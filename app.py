import os
import json
import time
import re
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Beautiful chart libraries with graceful imports
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt

# Optional imports
try:
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.transform import factor_cmap
    from bokeh.palettes import Category20, Viridis256
    from bokeh.embed import file_html
    from bokeh.resources import CDN
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    from plotnine import *
    PLOTNINE_AVAILABLE = True
except ImportError:
    PLOTNINE_AVAILABLE = False

from openai import OpenAI
from mcp_async_client import list_tools_sync, call_tool_sync
from visualization_intelligence import VisualizationIntelligence, get_smart_visualization_suggestions, analyze_query_for_chart_type

load_dotenv()

def auto_select_chart_type(df):
    """
    Intelligently select the best chart type based on data structure and content.
    """
    if df is None or df.empty:
        return "Bar Chart", "No data available"
    
    # Get data characteristics
    num_rows = len(df)
    num_cols = len(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Check for date/time columns (including string dates)
    potential_date_cols = []
    for col in categorical_cols:
        if any(keyword in col.lower() for keyword in ['date', 'time', 'month', 'year', 'day']):
            potential_date_cols.append(col)
    
    # Check for revenue/financial data
    revenue_cols = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'price', 'cost', 'profit']):
            revenue_cols.append(col)
    
    # Check for percentage/ratio data
    percentage_cols = []
    for col in numeric_cols:
        if df[col].max() <= 100 and df[col].min() >= 0:
            if any(keyword in col.lower() for keyword in ['percent', 'ratio', 'rate', '%']):
                percentage_cols.append(col)
    
    reasoning = []
    
    # Decision logic based on data characteristics
    
    # Time series data - Line Chart
    if (datetime_cols or potential_date_cols) and revenue_cols:
        reasoning.append("📈 Time series data detected with financial metrics")
        return "Line Chart", " | ".join(reasoning)
    
    # Monthly/temporal data - Line Chart
    if any(keyword in str(df.columns).lower() for keyword in ['month', 'quarter', 'year', 'week']):
        if revenue_cols:
            reasoning.append("📅 Temporal data with revenue metrics")
            return "Line Chart", " | ".join(reasoning)
    
    # Product/Category data with prices - Bar Chart
    if any(keyword in str(df.columns).lower() for keyword in ['product', 'category', 'brand']) and revenue_cols:
        reasoning.append("🛍️ Product/Category data with financial metrics")
        return "Bar Chart", " | ".join(reasoning)
    
    # Correlation analysis - Scatter Plot
    if len(numeric_cols) >= 2 and num_rows >= 10:
        reasoning.append("🔍 Multiple numeric variables for correlation analysis")
        return "Scatter Plot", " | ".join(reasoning)
    
    # Proportional data - Pie Chart
    if len(categorical_cols) == 1 and len(numeric_cols) == 1 and num_rows <= 12:
        reasoning.append("🥧 Single category with values, good for proportions")
        return "Pie Chart", " | ".join(reasoning)
    
    # Distribution analysis - Histogram/Distribution
    if len(numeric_cols) == 1 and num_rows >= 20:
        reasoning.append("📊 Single numeric variable with sufficient data points")
        return "Distribution", " | ".join(reasoning)
    
    # Comparison data - Bar Chart
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        reasoning.append("📊 Categorical data with numeric values for comparison")
        return "Bar Chart", " | ".join(reasoning)
    
    # Heatmap for correlation matrix
    if len(numeric_cols) >= 3:
        reasoning.append("🔥 Multiple numeric variables suitable for correlation heatmap")
        return "Heatmap", " | ".join(reasoning)
    
    # Default fallback
    reasoning.append("📊 Default choice for general data visualization")
    return "Bar Chart", " | ".join(reasoning)


def explain_chart_choice(df, chart_type, reasoning):
    """Display explanation for the auto-selected chart type."""
    st.info(f"""
    🤖 **Auto-Selected Chart Type: {chart_type}**
    
    **Reasoning:** {reasoning}
    
    **Data Analysis:**
    - Rows: {len(df)}
    - Columns: {len(df.columns)}
    - Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}
    - Categorical columns: {len(df.select_dtypes(include=['object', 'string']).columns)}
    - Date columns: {len(df.select_dtypes(include=['datetime64']).columns)}
    """)


def create_monthly_revenue_query(year):
    """Create a proper query for monthly revenue data."""
    return {
        "dimensions": [],
        "measures": ["sales.total_revenue"],
        "timeDimensions": [
            {
                "dimension": "sales.order_date",
                "dateRange": [f"{year}-01-01", f"{year}-12-31"],
                "granularity": "month"
            }
        ],
        "filters": [],
        "limit": 100
    }

def fix_time_dimension_query(base_query, time_dimension, date_range, granularity="month"):
    """Fix time dimension queries to ensure proper aggregation."""
    fixed_query = base_query.copy()
    
    # Ensure timeDimensions is properly structured
    if "timeDimensions" not in fixed_query or not fixed_query["timeDimensions"]:
        fixed_query["timeDimensions"] = []
    
    # Add or update the time dimension
    time_dim = {
        "dimension": time_dimension,
        "dateRange": date_range,
        "granularity": granularity
    }
    
    # Replace existing time dimension or add new one
    fixed_query["timeDimensions"] = [time_dim]
    
    return fixed_query

def handle_monthly_data_request(year="2023"):
    """Handle requests for monthly breakdown data."""
    try:
        # Create the monthly query
        monthly_query = create_monthly_revenue_query(year)
        
        # Display the query for debugging
        st.write("**Generated Query for Monthly Data:**")
        st.json(monthly_query)
        
        # Show the issue explanation
        st.info("""
        **Issue with Current Query Response:**
        
        The Lens2 API is returning aggregated data instead of monthly breakdown because:
        
        1. **Time Dimension Not Properly Processed**: The `timeDimensions` array in the response shows as empty `[]`
        2. **Granularity Not Applied**: The `month` granularity is not being applied to aggregate data by month
        3. **Possible Schema Issue**: The dimension `sales.order_date` might not support time aggregation
        
        **Solutions to Try:**
        """)
        
        # Alternative query approaches
        st.markdown("### 🔧 Alternative Query Approaches")
        
        tab1, tab2, tab3 = st.tabs(["📅 Add Date Dimension", "🔍 Check Schema", "📊 Manual Month Filter"])
        
        with tab1:
            st.markdown("**Try adding date as a regular dimension:**")
            alt_query1 = {
                "dimensions": ["sales.order_date"],
                "measures": ["sales.total_revenue"],
                "timeDimensions": [
                    {
                        "dimension": "sales.order_date",
                        "dateRange": [f"{year}-01-01", f"{year}-12-31"],
                        "granularity": "month"
                    }
                ],
                "filters": [],
                "limit": 100
            }
            st.json(alt_query1)
            
            if st.button("🚀 Try This Query", key="alt1"):
                st.code(f"""
Copy this into chat:
                
Please execute this query: {json.dumps(alt_query1, indent=2)}
                """)
        
        with tab2:
            st.markdown("**Check available dimensions and measures:**")
            schema_query = "Please get the Lens2 schema to see available dimensions and time dimensions"
            
            if st.button("📋 Get Schema", key="schema"):
                st.code(f"""
Copy this into chat:
                
{schema_query}
                """)
        
        with tab3:
            st.markdown("**Try individual month queries:**")
            
            months = [
                ("January", "2023-01-01", "2023-01-31"),
                ("February", "2023-02-01", "2023-02-28"),
                ("March", "2023-03-01", "2023-03-31"),
                ("April", "2023-04-01", "2023-04-30"),
                ("May", "2023-05-01", "2023-05-31"),
                ("June", "2023-06-01", "2023-06-30")
            ]
            
            selected_month = st.selectbox("Select Month:", options=[m[0] for m in months])
            
            if st.button("🗓️ Query Selected Month", key="month"):
                month_data = next(m for m in months if m[0] == selected_month)
                month_query = {
                    "dimensions": [],
                    "measures": ["sales.total_revenue"],
                    "filters": [
                        {
                            "member": "sales.order_date",
                            "operator": "inDateRange",
                            "values": [month_data[1], month_data[2]]
                        }
                    ],
                    "limit": 100
                }
                
                st.code(f"""
Copy this into chat:
                
Please execute this query for {selected_month}: {json.dumps(month_query, indent=2)}
                """)
        
        # Expected vs Actual Response
        st.markdown("### 📊 Expected vs Actual Response")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Expected Response (Monthly Data):**")
            expected_response = {
                "data": [
                    {"sales.order_date": "2023-01-01", "sales.total_revenue": "350000000"},
                    {"sales.order_date": "2023-02-01", "sales.total_revenue": "380000000"},
                    {"sales.order_date": "2023-03-01", "sales.total_revenue": "420000000"},
                    # ... more months
                ]
            }
            st.json(expected_response)
        
        with col2:
            st.markdown("**Actual Response (Aggregated):**")
            actual_response = {
                "data": [
                    {"sales.total_revenue": "4661006049.38"}
                ]
            }
            st.json(actual_response)
                
    except Exception as e:
        st.error(f"❌ Error executing monthly query: {str(e)}")
        logger.error(f"Monthly query error: {e}")

def execute_mcp_request(tool_name, arguments):
    """Execute an MCP request and return the response."""
    try:
        # This should integrate with your existing MCP client
        # For now, we'll use a placeholder
        st.info(f"Executing MCP tool: {tool_name} with args: {arguments}")
        
        # You'll need to integrate this with your actual MCP client
        # For example, if you have an mcp_client instance:
        # return mcp_client.call_tool(tool_name, arguments)
        
        return {"status": "placeholder", "data": []}
        
    except Exception as e:
        st.error(f"MCP request failed: {str(e)}")
        return None

st.set_page_config(page_title="Agentic MCP Chat (OpenAI)", layout="wide")
st.title("🤖 Agentic MCP Chat with Intelligent Visualization")
st.caption("Ask in natural language; the assistant will pick an MCP tool, build args, run it, and provide intelligent visualization recommendations.")

# --- Config / sidebar ---
DEFAULT_MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    # Tab selection
    tab_selection = st.radio(
        "Navigation",
        ["⚙️ Settings", "📊 Visualization"],
        index=0,
        horizontal=True
    )
    
    st.markdown("---")
    
    if tab_selection == "⚙️ Settings":
        st.header("Settings")
        mcp_url = st.text_input("MCP Server URL", value=DEFAULT_MCP_URL, help="Default MCP endpoint path is /mcp")
        model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
        temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        
        st.markdown("---")
        st.header("Visualization Options")
        auto_suggest = st.checkbox("Auto-suggest visualizations", value=True)
        show_chart_options = st.checkbox("Show advanced chart options", value=True)
        
        st.markdown("---")
        st.header("📊 Available Chart Libraries")
        st.write("✅ **Plotly** - Interactive charts")
        st.write("✅ **Seaborn** - Statistical plots") 
        st.write("✅ **Altair** - Declarative charts")
        if BOKEH_AVAILABLE:
            st.write("✅ **Bokeh** - Web-ready charts")
        else:
            st.write("❌ **Bokeh** - Not installed")
        if PLOTNINE_AVAILABLE:
            st.write("✅ **Plotnine** - ggplot2 style")
        else:
            st.write("❌ **Plotnine** - Not installed")
        
        st.markdown("---")
        if not OPENAI_API_KEY:
            st.error("Add OPENAI_API_KEY to your .env")
    
    elif tab_selection == "📊 Visualization":
        st.header("📊 Chart Configuration")
        
        # Initialize visualization session state
        if "viz_config" not in st.session_state:
            st.session_state.viz_config = {
                "chart_type": "auto",
                "library": "altair",
                "color_scheme": "default",
                "title": "",
                "show_legend": True,
                "interactive": True
            }
        
        # Chart Type Selection
        st.subheader("📈 Chart Type")
        chart_types = [
            "Auto (Smart Selection)", "Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", 
            "Pie Chart", "Donut Chart", "Heatmap", "Box Plot", "Violin Plot", 
            "Distribution", "Correlation", "Bubble Chart", "Treemap", "Horizontal Bar"
        ]
        selected_chart_type = st.selectbox(
            "Choose Chart Type:",
            chart_types,
            index=0,
            help="Auto will intelligently select the best chart type based on your data structure and content"
        )
        st.session_state.viz_config["chart_type"] = selected_chart_type
        
        # Library Selection
        st.subheader("🎨 Chart Library")
        library_options = ["altair", "plotly", "matplotlib", "seaborn"]
            
        selected_library = st.selectbox(
            "Choose Library:",
            library_options,
            index=library_options.index(st.session_state.viz_config["library"]),
            help="Select the charting library to use"
        )
        st.session_state.viz_config["library"] = selected_library
        
        # Color Schemes
        st.subheader("🎨 Styling")
        color_schemes = [
            "default", "viridis", "plasma", "coolwarm", "husl", 
            "Set1", "Set2", "Category10", "Category20"
        ]
        selected_color = st.selectbox(
            "Color Scheme:",
            color_schemes,
            index=color_schemes.index(st.session_state.viz_config["color_scheme"])
        )
        st.session_state.viz_config["color_scheme"] = selected_color
        
        # Chart Title
        chart_title = st.text_input(
            "Chart Title (optional):",
            value=st.session_state.viz_config["title"],
            placeholder="Enter custom chart title"
        )
        st.session_state.viz_config["title"] = chart_title
        
        # Chart Options
        st.subheader("⚙️ Chart Options")
        show_legend = st.checkbox(
            "Show Legend",
            value=st.session_state.viz_config["show_legend"]
        )
        st.session_state.viz_config["show_legend"] = show_legend
        
        interactive = st.checkbox(
            "Interactive Charts",
            value=st.session_state.viz_config["interactive"],
            help="Enable zoom, pan, and hover interactions"
        )
        st.session_state.viz_config["interactive"] = interactive
        
        st.markdown("---")
        
        # Create Chart Button
        if st.button("🎨 Create Chart", type="primary", width="stretch"):
            if st.session_state.last_data is not None:
                with st.spinner("Creating your visualization..."):
                    try:
                        # Get configuration
                        chart_type = st.session_state.viz_config["chart_type"]
                        library = st.session_state.viz_config["library"]
                        title = st.session_state.viz_config["title"] or "Generated Chart"
                        
                        # Convert data to DataFrame
                        df = pd.DataFrame(st.session_state.last_data) if isinstance(st.session_state.last_data, list) else st.session_state.last_data
                        
                        # Auto-select chart type if needed
                        if chart_type == "Auto (Smart Selection)":
                            auto_chart_type, reasoning = auto_select_chart_type(df)
                            explain_chart_choice(df, auto_chart_type, reasoning)
                            chart_type = auto_chart_type
                        
                        # Debug: Show data structure
                        with st.expander("📊 Data Preview & Analysis", expanded=False):
                            st.write(f"**Shape:** {df.shape}")
                            st.write(f"**Columns:** {list(df.columns)}")
                            st.write(f"**Data types:** {df.dtypes.to_dict()}")
                            st.dataframe(df.head(), width="stretch")
                        
                        # Handle different data types properly
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                        
                        # Smart column selection for charts
                        if chart_type.lower() in ["bar chart", "bar"]:
                            if len(numeric_cols) >= 1:
                                x_col = string_cols[0] if string_cols else df.columns[0]
                                y_col = numeric_cols[0]
                                
                                if library.lower() == "altair":
                                    chart = alt.Chart(df).mark_bar().encode(
                                        x=alt.X(x_col, title=x_col),
                                        y=alt.Y(y_col, title=y_col),
                                        color=alt.Color(y_col, scale=alt.Scale(scheme='viridis')),
                                        tooltip=[x_col, y_col]
                                    ).properties(
                                        title=title,
                                        width=600,
                                        height=400
                                    ).interactive()
                                    st.altair_chart(chart, width="stretch")
                                else:
                                    fig = px.bar(df, x=x_col, y=y_col, title=title, color=y_col)
                                    st.plotly_chart(fig, width="stretch")
                            else:
                                st.error("❌ Bar chart requires numeric data")
                                
                        elif chart_type.lower() in ["line chart", "line"]:
                            if len(numeric_cols) >= 1:
                                x_col = datetime_cols[0] if datetime_cols else (string_cols[0] if string_cols else df.columns[0])
                                y_col = numeric_cols[0]
                                
                                if library.lower() == "altair":
                                    chart = alt.Chart(df).mark_line(point=True).encode(
                                        x=alt.X(x_col, title=x_col),
                                        y=alt.Y(y_col, title=y_col),
                                        color=alt.value('steelblue'),
                                        tooltip=[x_col, y_col]
                                    ).properties(
                                        title=title,
                                        width=600,
                                        height=400
                                    ).interactive()
                                    st.altair_chart(chart, width="stretch")
                                else:
                                    fig = px.line(df, x=x_col, y=y_col, title=title)
                                    st.plotly_chart(fig, width="stretch")
                            else:
                                st.error("❌ Line chart requires numeric data")
                                
                        elif chart_type.lower() in ["scatter plot", "scatter"]:
                            if len(numeric_cols) >= 2:
                                x_col, y_col = numeric_cols[0], numeric_cols[1]
                                color_col = string_cols[0] if string_cols else None
                                
                                if library.lower() == "altair":
                                    chart = alt.Chart(df).mark_circle(size=100).encode(
                                        x=alt.X(x_col, title=x_col),
                                        y=alt.Y(y_col, title=y_col),
                                        color=alt.Color(color_col, title=color_col) if color_col else alt.value('steelblue'),
                                        tooltip=[x_col, y_col] + ([color_col] if color_col else [])
                                    ).properties(
                                        title=title,
                                        width=600,
                                        height=400
                                    ).interactive()
                                    st.altair_chart(chart, width="stretch")
                                else:
                                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
                                    st.plotly_chart(fig, width="stretch")
                            elif len(numeric_cols) >= 1:
                                x_col = string_cols[0] if string_cols else df.columns[0]
                                y_col = numeric_cols[0]
                                
                                if library.lower() == "altair":
                                    chart = alt.Chart(df).mark_circle(size=100).encode(
                                        x=alt.X(x_col, title=x_col),
                                        y=alt.Y(y_col, title=y_col),
                                        color=alt.Color(y_col, scale=alt.Scale(scheme='viridis')),
                                        tooltip=[x_col, y_col]
                                    ).properties(
                                        title=title,
                                        width=600,
                                        height=400
                                    ).interactive()
                                    st.altair_chart(chart, width="stretch")
                                else:
                                    fig = px.scatter(df, x=x_col, y=y_col, title=title)
                                    st.plotly_chart(fig, width="stretch")
                            else:
                                st.error("❌ Scatter plot requires numeric data")
                                
                        elif chart_type.lower() in ["pie chart", "pie"]:
                            if len(numeric_cols) > 0:
                                names_col = string_cols[0] if string_cols else None
                                values_col = numeric_cols[0]
                                
                                if library.lower() == "altair":
                                    if names_col:
                                        chart = alt.Chart(df).mark_arc().encode(
                                            theta=alt.Theta(values_col, title=values_col),
                                            color=alt.Color(names_col, title=names_col),
                                            tooltip=[names_col, values_col]
                                        ).properties(
                                            title=title,
                                            width=400,
                                            height=400
                                        )
                                        st.altair_chart(chart, width="stretch")
                                    else:
                                        st.error("❌ Pie chart needs categorical labels")
                                else:
                                    if names_col:
                                        fig = px.pie(df, names=names_col, values=values_col, title=title)
                                    else:
                                        fig = px.pie(df, values=values_col, title=title)
                                    st.plotly_chart(fig, width="stretch")
                            else:
                                st.error("❌ Pie chart requires numeric data")
                                
                        else:
                            # Default to bar chart
                            if len(numeric_cols) >= 1:
                                x_col = string_cols[0] if string_cols else df.columns[0]
                                y_col = numeric_cols[0]
                                
                                if library.lower() == "altair":
                                    chart = alt.Chart(df).mark_bar().encode(
                                        x=alt.X(x_col, title=x_col),
                                        y=alt.Y(y_col, title=y_col),
                                        color=alt.Color(y_col, scale=alt.Scale(scheme='viridis')),
                                        tooltip=[x_col, y_col]
                                    ).properties(
                                        title=title,
                                        width=600,
                                        height=400
                                    ).interactive()
                                    st.altair_chart(chart, width="stretch")
                                else:
                                    fig = px.bar(df, x=x_col, y=y_col, title=title)
                                    st.plotly_chart(fig, width="stretch")
                            else:
                                st.error("❌ Chart requires numeric data")
                        
                        st.success("✅ Chart created successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error creating chart: {str(e)}")
                        # Show debug information
                        with st.expander("🐛 Debug Info", expanded=False):
                            st.write("Data sample:")
                            st.dataframe(df.head() if 'df' in locals() else st.session_state.last_data)
                            st.write("Error details:", str(e))
            else:
                st.warning("⚠️ No data available. Please run a query first to get data for visualization.")
        
        # Quick Chart Buttons
        st.markdown("### 🚀 Quick Charts")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 Auto Chart", width="stretch", help="Automatically select the best chart type"):
                if st.session_state.last_data is not None:
                    try:
                        df = pd.DataFrame(st.session_state.last_data) if isinstance(st.session_state.last_data, list) else st.session_state.last_data
                        
                        # Auto-select chart type
                        auto_chart_type, reasoning = auto_select_chart_type(df)
                        explain_chart_choice(df, auto_chart_type, reasoning)
                        
                        # Create the auto-selected chart
                        library = st.session_state.viz_config["library"]
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                        
                        if auto_chart_type.lower() == "line chart" and len(numeric_cols) >= 1:
                            x_col = string_cols[0] if string_cols else df.columns[0]
                            y_col = numeric_cols[0]
                            if library.lower() == "altair":
                                chart = alt.Chart(df).mark_line(point=True).encode(
                                    x=alt.X(x_col, title=x_col),
                                    y=alt.Y(y_col, title=y_col),
                                    tooltip=[x_col, y_col]
                                ).properties(title=f"Auto-Selected: {auto_chart_type}").interactive()
                                st.altair_chart(chart, width="stretch")
                            else:
                                fig = px.line(df, x=x_col, y=y_col, title=f"Auto-Selected: {auto_chart_type}")
                                st.plotly_chart(fig, width="stretch")
                        elif auto_chart_type.lower() == "bar chart" and len(numeric_cols) >= 1:
                            x_col = string_cols[0] if string_cols else df.columns[0]
                            y_col = numeric_cols[0]
                            if library.lower() == "altair":
                                chart = alt.Chart(df).mark_bar().encode(
                                    x=alt.X(x_col, title=x_col),
                                    y=alt.Y(y_col, title=y_col),
                                    color=alt.Color(y_col, scale=alt.Scale(scheme='viridis')),
                                    tooltip=[x_col, y_col]
                                ).properties(title=f"Auto-Selected: {auto_chart_type}").interactive()
                                st.altair_chart(chart, width="stretch")
                            else:
                                fig = px.bar(df, x=x_col, y=y_col, title=f"Auto-Selected: {auto_chart_type}")
                                st.plotly_chart(fig, width="stretch")
                        
                        st.success(f"✅ {auto_chart_type} created automatically!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("No data available")
            
            if st.button("� Bar Chart", width="stretch"):
                if st.session_state.last_data is not None:
                    try:
                        df = pd.DataFrame(st.session_state.last_data) if isinstance(st.session_state.last_data, list) else st.session_state.last_data
                        
                        # Smart column selection
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                        
                        if len(numeric_cols) >= 1:
                            x_col = string_cols[0] if string_cols else df.columns[0]
                            y_col = numeric_cols[0]
                            
                            library = st.session_state.viz_config["library"]
                            if library.lower() == "altair":
                                chart = alt.Chart(df).mark_bar().encode(
                                    x=alt.X(x_col, title=x_col),
                                    y=alt.Y(y_col, title=y_col),
                                    color=alt.Color(y_col, scale=alt.Scale(scheme='viridis')),
                                    tooltip=[x_col, y_col]
                                ).properties(title="Bar Chart").interactive()
                                st.altair_chart(chart, width="stretch")
                            else:
                                fig = px.bar(df, x=x_col, y=y_col, title="Bar Chart")
                                st.plotly_chart(fig, width="stretch")
                        else:
                            st.error("❌ Bar chart requires numeric data")
                        
                        st.success("✅ Bar chart created!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("No data available")
        
        with col2:
            if st.button("� Line Chart", width="stretch"):
                if st.session_state.last_data is not None:
                    try:
                        df = pd.DataFrame(st.session_state.last_data) if isinstance(st.session_state.last_data, list) else st.session_state.last_data
                        
                        # Smart column selection for time series
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
                        string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                        
                        if len(numeric_cols) >= 1:
                            x_col = datetime_cols[0] if datetime_cols else (string_cols[0] if string_cols else df.columns[0])
                            y_col = numeric_cols[0]
                            
                            library = st.session_state.viz_config["library"]
                            if library.lower() == "altair":
                                chart = alt.Chart(df).mark_line(point=True).encode(
                                    x=alt.X(x_col, title=x_col),
                                    y=alt.Y(y_col, title=y_col),
                                    tooltip=[x_col, y_col]
                                ).properties(title="Line Chart").interactive()
                                st.altair_chart(chart, width="stretch")
                            else:
                                fig = px.line(df, x=x_col, y=y_col, title="Line Chart")
                                st.plotly_chart(fig, width="stretch")
                        else:
                            st.error("❌ Line chart requires numeric data")
                        
                        st.success("✅ Line chart created!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("No data available")
            
            if st.button("🔍 Scatter Plot", width="stretch"):
                if st.session_state.last_data is not None:
                    try:
                        df = pd.DataFrame(st.session_state.last_data) if isinstance(st.session_state.last_data, list) else st.session_state.last_data
                        
                        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        
                        if len(numeric_cols) >= 2:
                            library = st.session_state.viz_config["library"]
                            if library.lower() == "altair":
                                chart = alt.Chart(df).mark_circle(size=100).encode(
                                    x=alt.X(numeric_cols[0], title=numeric_cols[0]),
                                    y=alt.Y(numeric_cols[1], title=numeric_cols[1]),
                                    tooltip=[numeric_cols[0], numeric_cols[1]]
                                ).properties(title="Scatter Plot").interactive()
                                st.altair_chart(chart, width="stretch")
                            else:
                                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title="Scatter Plot")
                                st.plotly_chart(fig, width="stretch")
                        elif len(numeric_cols) >= 1:
                            string_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
                            x_col = string_cols[0] if string_cols else df.columns[0]
                            y_col = numeric_cols[0]
                            
                            library = st.session_state.viz_config["library"]
                            if library.lower() == "altair":
                                chart = alt.Chart(df).mark_circle(size=100).encode(
                                    x=alt.X(x_col, title=x_col),
                                    y=alt.Y(y_col, title=y_col),
                                    tooltip=[x_col, y_col]
                                ).properties(title="Scatter Plot").interactive()
                                st.altair_chart(chart, width="stretch")
                            else:
                                fig = px.scatter(df, x=x_col, y=y_col, title="Scatter Plot")
                                st.plotly_chart(fig, width="stretch")
                        else:
                            st.error("❌ Scatter plot requires numeric data")
                        
                        st.success("✅ Scatter plot created!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                else:
                    st.warning("No data available")
        
        st.markdown("---")
        
        # Monthly Data Analysis Section
        st.markdown("### 📅 Monthly Data Analysis")
        st.markdown("*Fix for time-based queries that return aggregated instead of monthly data*")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            year_input = st.selectbox(
                "Select Year for Monthly Analysis:",
                options=["2023", "2022", "2024"],
                index=0
            )
        
        with col2:
            if st.button("📊 Get Monthly Data", type="secondary", width="stretch"):
                handle_monthly_data_request(year_input)
        
        # Query Builder Section
        st.markdown("### 🔧 Query Builder")
        st.markdown("*Build time-dimension queries correctly*")
        
        with st.expander("Time Dimension Query Builder", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                time_dimension = st.text_input(
                    "Time Dimension:", 
                    value="sales.order_date",
                    help="The time dimension field name"
                )
                
                start_date = st.date_input(
                    "Start Date:",
                    value=pd.to_datetime("2023-01-01")
                )
                
            with col2:
                granularity = st.selectbox(
                    "Granularity:",
                    options=["month", "week", "day", "quarter", "year"],
                    index=0
                )
                
                end_date = st.date_input(
                    "End Date:",
                    value=pd.to_datetime("2023-12-31")
                )
            
            measure_input = st.text_input(
                "Measure:",
                value="sales.total_revenue",
                help="The measure to aggregate"
            )
            
            if st.button("🚀 Build & Execute Query", type="primary"):
                # Create the query
                query = {
                    "dimensions": [],
                    "measures": [measure_input],
                    "timeDimensions": [
                        {
                            "dimension": time_dimension,
                            "dateRange": [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
                            "granularity": granularity
                        }
                    ],
                    "filters": [],
                    "limit": 100
                }
                
                st.write("**Generated Query:**")
                st.json(query)
                
                # Copy to clipboard functionality
                st.code(f"""
Execute this query in your chat:
                
Please execute this query: {json.dumps(query, indent=2)}
                """)
                
                st.info("💡 Copy the query above and paste it in the chat to execute")
        
        st.markdown("---")
        
        # Data Debugging Section  
        st.markdown("### 🐛 Data Debugging")
        if st.session_state.last_data is not None:
            st.write("**Current Data Structure:**")
            df = pd.DataFrame(st.session_state.last_data) if isinstance(st.session_state.last_data, list) else st.session_state.last_data
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")
            
            with st.expander("View Raw Data", expanded=False):
                st.dataframe(df)
        else:
            st.info("💡 No data loaded yet. Run a query in the chat to see data here.")
        
        # Data Preview
        if st.session_state.last_data is not None:
            st.markdown("---")
            st.subheader("📋 Current Data Preview")
            
            # Data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", st.session_state.last_data.shape[0])
            with col2:
                st.metric("Columns", st.session_state.last_data.shape[1])
            with col3:
                numeric_cols = len(st.session_state.last_data.select_dtypes(include=['number']).columns)
                st.metric("Numeric Cols", numeric_cols)
            
            # Data types summary
            st.markdown("**Column Types:**")
            dtypes_df = pd.DataFrame({
                'Column': st.session_state.last_data.columns,
                'Type': st.session_state.last_data.dtypes.astype(str),
                'Non-Null': st.session_state.last_data.count()
            })
            st.dataframe(dtypes_df, width="stretch", height=200)
            
            # Sample data
            with st.expander("View Sample Data"):
                st.dataframe(st.session_state.last_data.head(10), width="stretch")
        else:
            st.markdown("---")
            st.info("💡 **Tip**: Run a data query in the chat to load data for visualization")
            
            # Show example queries
            with st.expander("📝 Example Data Queries"):
                st.markdown("""
                **Try these example queries in the chat:**
                
                • `Show me sales by product category`
                • `What are the top 10 customers by revenue?`
                • `Show monthly sales trends for the last year`
                • `Display product performance metrics`
                • `Get customer demographics breakdown`
                
                After running a query, return to this tab to create visualizations!
                """)
    
    # Set default values for variables that need to be accessible outside the tab
    if 'mcp_url' not in locals():
        mcp_url = DEFAULT_MCP_URL
    if 'model_name' not in locals():
        model_name = "gpt-4o-mini"
    if 'temp' not in locals():
        temp = 0.2
    if 'auto_suggest' not in locals():
        auto_suggest = True
    if 'show_chart_options' not in locals():
        show_chart_options = True

# Normalize URL to include /mcp if missing
if mcp_url and not mcp_url.rstrip("/").endswith("/mcp"):
    mcp_url = mcp_url.rstrip("/") + "/mcp"

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "oai_tools" not in st.session_state:
    st.session_state.oai_tools = None
if "raw_tools" not in st.session_state:
    st.session_state.raw_tools = []
if "last_data" not in st.session_state:
    st.session_state.last_data = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "viz_intelligence" not in st.session_state:
    st.session_state.viz_intelligence = VisualizationIntelligence()

# Forward declarations for functions used in sidebar
def create_chart_from_sidebar(df, chart_type, library="plotly", title=None):
    """Create chart from sidebar configuration - wrapper function."""
    try:
        # Import the functions here to avoid circular dependency
        if chart_type in ["bar", "line", "scatter", "pie", "area", "heatmap", "bubble", "treemap"]:
            # Simple chart creation using Plotly
            fig = None
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if chart_type == "bar" and categorical_cols and numeric_cols:
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0], title=title or "Bar Chart")
            elif chart_type == "line" and numeric_cols:
                if categorical_cols:
                    fig = px.line(df, x=categorical_cols[0], y=numeric_cols[0], title=title or "Line Chart")
                else:
                    fig = px.line(df, y=numeric_cols[0], title=title or "Line Chart")
            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=title or "Scatter Plot")
            elif chart_type == "pie" and categorical_cols and numeric_cols:
                fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0], title=title or "Pie Chart")
            
            if fig:
                st.plotly_chart(fig, width="stretch")
                return f"✅ Created {chart_type} chart"
            else:
                st.warning(f"Cannot create {chart_type} chart with current data structure")
                return f"⚠️ Cannot create {chart_type} chart"
        else:
            # For complex chart types, just show a simple version
            if numeric_cols:
                fig = px.bar(df, y=numeric_cols[0], title=title or f"{chart_type.title()} Chart")
                st.plotly_chart(fig, use_container_width=True)
                return f"✅ Created chart as bar chart (simplified)"
            else:
                st.warning("No numeric data available for visualization")
                return "⚠️ No numeric data available"
                
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return f"❌ Error creating {chart_type} chart: {e}"

# Forward declarations for functions used in sidebar - removed duplicate
# def create_chart_from_sidebar function was here but is now above

def _json_schema_or_default(schema):
    # Ensure a valid JSON schema object for OpenAI function parameters
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "additionalProperties": True}
    if "type" not in schema:
        schema["type"] = "object"
    if "properties" not in schema:
        schema["properties"] = {}
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = True
    return schema

def refresh_tools():
    """Pull tools from MCP and convert them to OpenAI function-calling schema."""
    raw = list_tools_sync(mcp_url)
    oai_tools = []
    for t in raw:
        name = t["name"]
        desc = t.get("description") or f"Tool {name}"
        schema = _json_schema_or_default(t.get("input_schema"))
        oai_tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": schema,
            }
        })
    st.session_state.raw_tools = raw
    st.session_state.oai_tools = oai_tools
    return oai_tools

def validate_load_query_args(args):
    """Validate and fix common issues with execute_load_query arguments according to DataOS Lens documentation."""
    if not isinstance(args, dict):
        return {"error": "Arguments must be a dictionary"}
    
    query = args.get("query", {})
    if not isinstance(query, dict):
        return {"error": "Query must be a dictionary object"}
    
    # Ensure required keys exist with proper defaults (DataOS Lens format)
    fixed_query = {
        "dimensions": query.get("dimensions", []),
        "measures": query.get("measures", []),
        "filters": query.get("filters", []),
        "timeDimensions": query.get("timeDimensions", []),
        "limit": query.get("limit", 100)
    }
    
    # Validate dimensions and measures are lists
    if not isinstance(fixed_query["dimensions"], list):
        fixed_query["dimensions"] = []
    if not isinstance(fixed_query["measures"], list):
        fixed_query["measures"] = []
    if not isinstance(fixed_query["filters"], list):
        fixed_query["filters"] = []
    if not isinstance(fixed_query["timeDimensions"], list):
        fixed_query["timeDimensions"] = []
    
    # Validate timeDimensions structure according to DataOS Lens docs
    for i, time_dim in enumerate(fixed_query["timeDimensions"]):
        if not isinstance(time_dim, dict):
            return {"error": f"timeDimensions[{i}] must be a dictionary"}
        
        # Required fields for timeDimensions
        if "dimension" not in time_dim:
            return {"error": f"timeDimensions[{i}] must include 'dimension' field"}
        
        # Ensure dateRange exists and is properly formatted
        if "dateRange" not in time_dim:
            time_dim["dateRange"] = ["2023-01-01", "2023-12-31"]
        elif not isinstance(time_dim["dateRange"], list) or len(time_dim["dateRange"]) != 2:
            return {"error": f"timeDimensions[{i}].dateRange must be a list with start and end dates"}
        
        # Ensure granularity exists
        if "granularity" not in time_dim:
            time_dim["granularity"] = "month"
        elif time_dim["granularity"] not in ["day", "week", "month", "quarter", "year"]:
            return {"error": f"timeDimensions[{i}].granularity must be one of: day, week, month, quarter, year"}
    
    # Convert old-style date filters to timeDimensions format for backward compatibility
    new_filters = []
    for filter_item in fixed_query["filters"]:
        if isinstance(filter_item, dict):
            # Check if it's a date filter that should be converted to timeDimensions
            if "member" in filter_item and "order_date" in filter_item.get("member", ""):
                # This is likely a date filter, suggest using timeDimensions instead
                continue  # Skip this filter, let timeDimensions handle it
            else:
                new_filters.append(filter_item)
    
    fixed_query["filters"] = new_filters
    
    # Ensure limit is an integer
    try:
        fixed_query["limit"] = int(fixed_query["limit"])
    except (ValueError, TypeError):
        fixed_query["limit"] = 100
    
    # Validation: Must have at least one measure
    if not fixed_query["measures"]:
        return {"error": "Query must include at least one measure"}
    
    return {"query": fixed_query}
    try:
        fixed_query["limit"] = int(fixed_query["limit"])
    except (ValueError, TypeError):
        fixed_query["limit"] = 100
    
    # Validation: Must have at least one measure
    if not fixed_query["measures"]:
        return {"error": "Query must include at least one measure. Use get_metadata to discover available measures in the current schema."}
    
    return {"query": fixed_query}

def parse_table_from_text(text):
    """Parse markdown table from assistant response and convert to DataFrame."""
    try:
        # Find markdown table pattern
        table_pattern = r'\|.*\|[\r\n]+\|.*\|[\r\n]+(?:\|.*\|[\r\n]+)+'
        table_match = re.search(table_pattern, text)
        
        if not table_match:
            return None
            
        table_text = table_match.group(0)
        lines = [line.strip() for line in table_text.split('\n') if line.strip()]
        
        if len(lines) < 3:  # Header, separator, at least one data row
            return None
            
        # Parse header
        header = [col.strip() for col in lines[0].split('|')[1:-1]]
        
        # Parse data rows (skip separator line)
        data = []
        for line in lines[2:]:
            row = [col.strip() for col in line.split('|')[1:-1]]
            if len(row) == len(header):
                data.append(row)
        
        if not data:
            return None
            
        df = pd.DataFrame(data, columns=header)
        
        # Try to convert numeric columns
        for col in df.columns:
            # Check if column contains currency values
            if df[col].astype(str).str.contains(r'[\$,]').any():
                df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True)
            
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Try to parse dates
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass  # Keep as string
        
        return df
        
    except Exception as e:
        st.error(f"Error parsing table: {e}")
        return None

def create_advanced_chart(df, chart_type="auto", title=None, user_query=""):
    """Create advanced charts with intelligent recommendations."""
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return
    
    try:
        # Get intelligent recommendations if auto mode
        if chart_type == "auto" and auto_suggest:
            recommendations = st.session_state.viz_intelligence.get_chart_recommendations(
                df, user_query, st.session_state.last_response
            )
            
            if recommendations:
                with st.expander("🧠 Intelligent Chart Recommendations", expanded=True):
                    primary_rec = recommendations[0]
                    st.markdown(f"**Recommended**: {primary_rec.name}")
                    st.markdown(f"*Why*: {primary_rec.analytical_value}")
                    
                    if len(recommendations) > 1:
                        st.markdown("**Alternatives**:")
                        for rec in recommendations[1:3]:
                            st.markdown(f"• {rec.name}: {rec.description}")
                
                # Use the primary recommendation
                chart_type = recommendations[0].chart_type
        
        # Chart creation section
        col1, col2 = st.columns([4, 1])
        
        with col2:
            if show_chart_options:
                st.markdown("**Chart Options**")
                
                # Chart type selection
                chart_types = ["auto", "bar", "line", "scatter", "area", "heatmap", "pie", "distribution", "boxplot", "violin", "correlation"]
                selected_chart = st.selectbox("Chart Type:", chart_types, index=0)
                
                # Library selection  
                library_options = ["altair", "plotly", "matplotlib", "seaborn"]
                selected_library = st.selectbox("Library:", library_options, index=0)
                
                if st.button("🎨 Create Chart"):
                    result = create_beautiful_chart(df, selected_chart, selected_library, user_query=user_query)
                    st.success(result)
                    return
        
        with col1:
            if title:
                st.subheader(f"📊 {title}")
            else:
                st.subheader("📊 Data Visualization")
            
            # Create the appropriate chart using beautiful chart system
            result = create_beautiful_chart(df, chart_type, "auto", user_query=user_query)
                
        # Show the underlying data
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        st.write("Data preview:")
        st.dataframe(df.head())

def create_plotly_chart(df, chart_type):
    """Create Plotly charts based on chart type."""
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        if chart_type == "line":
            if datetime_cols and numeric_cols:
                fig = px.line(df, x=datetime_cols[0], y=numeric_cols[0], 
                             title="Time Series Analysis")
            elif categorical_cols and numeric_cols:
                fig = px.line(df, x=categorical_cols[0], y=numeric_cols[0],
                             title="Trend Analysis")
            else:
                st.warning("Line chart needs time or categorical data with numeric values")
                return
        
        elif chart_type == "area":
            if datetime_cols and numeric_cols:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df[datetime_cols[0]], y=df[numeric_cols[0]],
                                       fill='tonexty', mode='lines',
                                       name=numeric_cols[0]))
                fig.update_layout(title="Area Chart - Magnitude Over Time")
            else:
                st.warning("Area chart needs time data with numeric values")
                return
        
        elif chart_type == "bar" or chart_type == "horizontal_bar":
            if categorical_cols and numeric_cols:
                orientation = 'h' if chart_type == "horizontal_bar" else 'v'
                fig = px.bar(df, x=categorical_cols[0] if orientation == 'v' else numeric_cols[0],
                           y=numeric_cols[0] if orientation == 'v' else categorical_cols[0],
                           orientation=orientation,
                           title="Categorical Comparison")
            else:
                st.warning("Bar chart needs categorical and numeric data")
                return
        
        elif chart_type == "pie" or chart_type == "donut":
            if categorical_cols and numeric_cols:
                fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0],
                           hole=0.3 if chart_type == "donut" else 0,
                           title="Proportional Distribution")
            else:
                st.warning("Pie chart needs categorical and numeric data")
                return
        
        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                color_col = categorical_cols[0] if categorical_cols else None
                size_col = numeric_cols[2] if len(numeric_cols) >= 3 else None
                
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               color=color_col, size=size_col,
                               title="Correlation Analysis",
                               trendline="ols")
            else:
                st.warning("Scatter plot needs at least 2 numeric columns")
                return
        
        elif chart_type == "bubble":
            if len(numeric_cols) >= 2:
                size_col = numeric_cols[2] if len(numeric_cols) >= 3 else numeric_cols[0]
                color_col = categorical_cols[0] if categorical_cols else None
                
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1],
                               size=size_col, color=color_col,
                               title="Multi-Variable Bubble Analysis")
            else:
                st.warning("Bubble chart needs at least 2 numeric columns")
                return
        
        elif chart_type == "treemap":
            if categorical_cols and numeric_cols:
                fig = px.treemap(df, path=[categorical_cols[0]], values=numeric_cols[0],
                               title="Hierarchical Proportions")
            else:
                st.warning("Treemap needs categorical and numeric data")
                return
        
        elif chart_type == "heatmap":
            if len(numeric_cols) >= 2:
                # Create correlation heatmap
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Correlation Heatmap")
            else:
                st.warning("Heatmap needs multiple numeric columns")
                return
        
        else:
            # Default to bar chart
            if categorical_cols and numeric_cols:
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0],
                           title="Data Visualization")
            else:
                st.warning("Cannot determine appropriate chart type for this data")
                return
        
        # Update layout for better appearance
        fig.update_layout(
            height=500,
            showlegend=True,
            font=dict(size=12),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating {chart_type} chart: {e}")
        st.dataframe(df.head())

# ===============================================================================
# BEAUTIFUL CHART LIBRARIES FUNCTIONS
# ===============================================================================

def create_seaborn_chart(df, chart_type="auto", title=None, user_query=""):
    """Create beautiful statistical charts with Seaborn."""
    
    # Set style for beautiful plots
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    if chart_type == "distribution":
        # Beautiful distribution plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            fig, axes = plt.subplots(1, len(numeric_cols[:3]), figsize=(15, 5))
            if len(numeric_cols[:3]) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:3]):
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
            
            plt.tight_layout()
            return fig
    
    elif chart_type == "correlation":
        # Beautiful correlation heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=.5, ax=ax)
            plt.title('Correlation Matrix')
            return fig
    
    elif chart_type == "scatter":
        # Beautiful scatter plot with regression
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(data=df, x=numeric_cols[0], y=numeric_cols[1], 
                       scatter_kws={'alpha':0.6}, line_kws={'color': 'red'}, ax=ax)
            plt.title(f'{numeric_cols[1]} vs {numeric_cols[0]} with Trend Line')
            return fig
    
    elif chart_type == "boxplot":
        # Beautiful box plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols and categorical_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=df, x=categorical_cols[0], y=numeric_cols[0], ax=ax)
            sns.stripplot(data=df, x=categorical_cols[0], y=numeric_cols[0], 
                         color='black', alpha=0.5, ax=ax)
            plt.xticks(rotation=45)
            plt.title(f'{numeric_cols[0]} Distribution by {categorical_cols[0]}')
            plt.tight_layout()
            return fig
    
    elif chart_type == "violin":
        # Beautiful violin plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols and categorical_cols:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.violinplot(data=df, x=categorical_cols[0], y=numeric_cols[0], ax=ax)
            plt.xticks(rotation=45)
            plt.title(f'{numeric_cols[0]} Distribution by {categorical_cols[0]}')
            plt.tight_layout()
            return fig
    
    return None

def create_altair_chart(df, chart_type="auto", title=None, user_query=""):
    """Create beautiful interactive charts with Altair."""
    
    if chart_type == "line":
        # Beautiful interactive line chart
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if date_cols and numeric_cols:
            chart = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X(date_cols[0], title=date_cols[0]),
                y=alt.Y(numeric_cols[0], title=numeric_cols[0]),
                tooltip=[date_cols[0], numeric_cols[0]]
            ).properties(
                width=600,
                height=400,
                title=title or f"{numeric_cols[0]} Over Time"
            ).interactive()
            
            return chart
    
    elif chart_type == "bar":
        # Beautiful interactive bar chart
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols and numeric_cols:
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X(categorical_cols[0], sort='-y', title=categorical_cols[0]),
                y=alt.Y(numeric_cols[0], title=numeric_cols[0]),
                color=alt.Color(categorical_cols[0], legend=None),
                tooltip=[categorical_cols[0], numeric_cols[0]]
            ).properties(
                width=600,
                height=400,
                title=title or f"{numeric_cols[0]} by {categorical_cols[0]}"
            ).interactive()
            
            return chart
    
    elif chart_type == "scatter":
        # Beautiful interactive scatter plot
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            chart = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X(numeric_cols[0], title=numeric_cols[0]),
                y=alt.Y(numeric_cols[1], title=numeric_cols[1]),
                color=alt.Color(numeric_cols[0], scale=alt.Scale(scheme='viridis')),
                tooltip=list(df.columns)
            ).properties(
                width=600,
                height=400,
                title=title or f"{numeric_cols[1]} vs {numeric_cols[0]}"
            ).interactive()
            
            return chart
    
    elif chart_type == "area":
        # Beautiful interactive area chart
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if date_cols and numeric_cols:
            chart = alt.Chart(df).mark_area(opacity=0.7).encode(
                x=alt.X(date_cols[0], title=date_cols[0]),
                y=alt.Y(numeric_cols[0], title=numeric_cols[0]),
                color=alt.value('#1f77b4'),
                tooltip=[date_cols[0], numeric_cols[0]]
            ).properties(
                width=600,
                height=400,
                title=title or f"Area Chart: {numeric_cols[0]}"
            ).interactive()
            
            return chart
    
    return None

def create_bokeh_chart(df, chart_type="auto", title=None, user_query=""):
    """Create beautiful interactive charts with Bokeh."""
    
    if not BOKEH_AVAILABLE:
        return None
    
    source = ColumnDataSource(df)
    
    if chart_type == "line":
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if date_cols and numeric_cols:
            p = figure(width=600, height=400, title=title or f"{numeric_cols[0]} Over Time",
                      x_axis_type='datetime')
            
            p.line(x=date_cols[0], y=numeric_cols[0], source=source, 
                  line_width=2, color='#2E86AB', legend_label=numeric_cols[0])
            p.circle(x=date_cols[0], y=numeric_cols[0], source=source, 
                    size=6, color='#A23B72', alpha=0.8)
            
            p.add_tools(HoverTool(tooltips=[
                (date_cols[0], f"@{date_cols[0]}{{%F}}"),
                (numeric_cols[0], f"@{numeric_cols[0]}{{0.2f}}")
            ]))
            
            return p
    
    elif chart_type == "bar":
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_cols and numeric_cols:
            p = figure(width=600, height=400, title=title or f"{numeric_cols[0]} by {categorical_cols[0]}",
                      x_range=df[categorical_cols[0]].unique())
            
            p.vbar(x=categorical_cols[0], top=numeric_cols[0], source=source,
                  width=0.8, color=factor_cmap(categorical_cols[0], palette=Category20[20], 
                                             factors=df[categorical_cols[0]].unique()))
            
            p.add_tools(HoverTool(tooltips=[
                (categorical_cols[0], f"@{categorical_cols[0]}"),
                (numeric_cols[0], f"@{numeric_cols[0]}{{0.2f}}")
            ]))
            
            p.xaxis.major_label_orientation = 45
            return p
    
    elif chart_type == "scatter":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            p = figure(width=600, height=400, title=title or f"{numeric_cols[1]} vs {numeric_cols[0]}")
            
            p.scatter(x=numeric_cols[0], y=numeric_cols[1], source=source,
                     size=8, alpha=0.8, color='#2E86AB')
            
            p.add_tools(HoverTool(tooltips=[
                (numeric_cols[0], f"@{numeric_cols[0]}{{0.2f}}"),
                (numeric_cols[1], f"@{numeric_cols[1]}{{0.2f}}")
            ]))
            
            return p
    
    return None

def display_seaborn_chart(fig):
    """Display Seaborn chart in Streamlit."""
    if fig:
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close(fig)  # Clean up memory

def display_bokeh_chart(p):
    """Display Bokeh chart in Streamlit."""
    if p:
        html = file_html(p, CDN, "chart")
        st.components.v1.html(html, height=450)

def create_beautiful_chart(df, chart_type="auto", library="altair", title=None, user_query=""):
    """Choose the best library for beautiful charts based on data and requirements."""
    
    if library == "auto":
        # Auto-select library based on chart type and data
        if chart_type in ["distribution", "correlation", "boxplot", "violin"]:
            library = "seaborn"  # Statistical plots
        elif chart_type in ["line", "bar", "scatter", "area"]:
            library = "altair"   # Interactive and clean
        elif chart_type == "heatmap":
            library = "plotly"   # Complex interactions
        else:
            library = "plotly"   # Default
    
    # Show library selection
    st.info(f"🎨 Creating {chart_type} chart with {library.title()}")
    
    # Create chart with selected library
    if library == "seaborn":
        fig = create_seaborn_chart(df, chart_type, title, user_query)
        if fig:
            display_seaborn_chart(fig)
            return f"✅ Created beautiful {chart_type} chart with Seaborn"
    
    elif library == "altair":
        chart = create_altair_chart(df, chart_type, title, user_query)
        if chart:
            st.altair_chart(chart, use_container_width=True)
            return f"✅ Created interactive {chart_type} chart with Altair"
    
    elif library == "bokeh" and BOKEH_AVAILABLE:
        p = create_bokeh_chart(df, chart_type, title, user_query)
        if p:
            display_bokeh_chart(p)
            return f"✅ Created interactive {chart_type} chart with Bokeh"
    
    else:  # Default to Plotly
        fig = create_plotly_chart(df, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            return f"✅ Created interactive {chart_type} chart with Plotly"
    
    return "❌ Could not create visualization"

# ===============================================================================

def detect_chart_request(user_message):
    """Detect if user is requesting a chart/visualization."""
    analysis = analyze_query_for_chart_type(user_message)
    return analysis['has_chart_request'] or analysis['is_generic_request']

def extract_chart_type(user_message):
    """Extract requested chart type from user message."""
    analysis = analyze_query_for_chart_type(user_message)
    if analysis['requested_types']:
        return analysis['requested_types'][0]  # Return first requested type
    return 'auto'

def detect_new_data_request(user_message):
    """Detect if user is asking for different data dimensions/measures that require a new query."""
    
    # Keywords that suggest new data requirements
    product_keywords = ['product', 'category', 'brand', 'item', 'merchandise']
    customer_keywords = ['customer', 'client', 'user', 'demographics', 'location', 'city', 'state']
    revenue_keywords = ['revenue', 'income', 'earnings', 'profit', 'money']
    time_keywords = ['monthly', 'quarterly', 'yearly', 'daily', 'trends', 'over time']
    
    user_lower = user_message.lower()
    
    # Check if asking for product-related data
    if any(keyword in user_lower for keyword in product_keywords):
        # If current data doesn't have product columns, need new query
        if st.session_state.last_data is None:
            return True
        current_columns = str(st.session_state.last_data.columns).lower()
        if not any(keyword in current_columns for keyword in ['product', 'category', 'brand']):
            return True
    
    # Check if asking for customer-related data  
    if any(keyword in user_lower for keyword in customer_keywords):
        if st.session_state.last_data is None:
            return True
        current_columns = str(st.session_state.last_data.columns).lower()
        if not any(keyword in current_columns for keyword in ['customer', 'client', 'city', 'state']):
            return True
    
    # Check if asking for revenue when we only have sales count
    if any(keyword in user_lower for keyword in revenue_keywords):
        if st.session_state.last_data is None:
            return True
        current_columns = str(st.session_state.last_data.columns).lower()
        if 'revenue' not in current_columns and 'total_revenue' not in current_columns:
            return True
    
    # Check if asking for time-based analysis
    if any(keyword in user_lower for keyword in time_keywords):
        if st.session_state.last_data is None:
            return True
        current_columns = str(st.session_state.last_data.columns).lower()
        if not any(keyword in current_columns for keyword in ['date', 'month', 'quarter', 'year', 'time']):
            return True
    
    return False

# UI: tool refresh + clear chat + schema info
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔄 Refresh tools"):
        try:
            n = len(refresh_tools())
            st.success(f"Loaded {n} tool specs")
        except Exception as e:
            st.error(f"Failed to load tools: {e}")
with col2:
    if st.button("🧹 Clear chat"):
        st.session_state.messages = []
        st.session_state.last_data = None
        st.session_state.last_query = ""
        st.session_state.last_response = ""
with col3:
    if st.button("📊 Schema Info"):
        # Show current endpoint info
        endpoint_url = os.getenv("LENS2_API_URL", "Not configured")
        if "sales360" in endpoint_url:
            schema_type = "🛍️ SALES360 (Product-focused)"
        elif "customer360" in endpoint_url:
            schema_type = "👥 CUSTOMER360 (Customer-focused)"
        else:
            schema_type = "❓ Unknown Schema"
        
        st.info(f"**Current Endpoint**: {schema_type}\n\n**URL**: {endpoint_url}")
with col4:
    if st.button("🎯 Get Smart Suggestions") and st.session_state.last_data is not None:
        suggestions = get_smart_visualization_suggestions(
            st.session_state.last_data, 
            st.session_state.last_query,
            st.session_state.last_response
        )
        st.info(suggestions)

# Auto-load once
if st.session_state.oai_tools is None:
    try:
        refresh_tools()
    except Exception as e:
        st.warning(f"Unable to load tools yet: {e}")

# Render chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Utility: convert MCP content blocks -> readable string for tool result
def mcp_blocks_to_text(blocks) -> str:
    if isinstance(blocks, list):
        out = []
        for b in blocks:
            # Common MCP shape: {"type":"text","text":"..."}
            if isinstance(b, dict):
                if b.get("type") == "text" and "text" in b:
                    out.append(b["text"])
                else:
                    out.append(json.dumps(b, ensure_ascii=False))
            else:
                # objects with attributes
                t = getattr(b, "type", None)
                txt = getattr(b, "text", None)
                if t == "text" and txt is not None:
                    out.append(str(txt))
                else:
                    try:
                        out.append(json.dumps(getattr(b, "__dict__", str(b)), ensure_ascii=False))
                    except Exception:
                        out.append(str(b))
        return "\n".join(out).strip()
    # Fallback
    try:
        return json.dumps(blocks, ensure_ascii=False)
    except Exception:
        return str(blocks)

# Load enhanced system prompt
def load_system_prompt():
    """Load the enhanced system prompt with schema detection instructions."""
    try:
        # Load the base system prompt
        with open("system.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
        
        # Add dynamic schema detection instructions
        schema_detection_prompt = """

## DYNAMIC SCHEMA ADAPTATION

**CRITICAL FIRST STEP**: Before answering any data queries, you MUST:

1. **Check Connection**: Use `get_connection_status` to verify connectivity
2. **Discover Schema**: Use `get_metadata` to understand the current schema structure
3. **Identify Schema Type**: Determine if this is sales360, customer360, or another schema variant
4. **Adapt Field Names**: Use the actual field names discovered from metadata, not hardcoded assumptions

**Common Schema Patterns to Expect:**

**SALES360 Schema (Product-focused):**
- Likely measures: `sales.total_revenue`, `sales.total_sales`, `product.*`
- Likely dimensions: `product.category`, `product.brand`, `product.name`, `sales.order_date`

**CUSTOMER360 Schema (Customer-focused):**
- Likely measures: `customer.total_customers`, `sales.total_sales`
- Likely dimensions: `customer.city`, `customer.status`, `customer.first_name`

**Field Name Discovery Rules:**
- NEVER assume field names exist without checking metadata first
- When user asks for "product category", look for fields like `product.category`, `product_category`, or similar
- When user asks for "revenue", look for fields like `sales.total_revenue`, `revenue`, `total_revenue`
- Adapt your queries based on what's actually available in the schema

**Error Handling:**
- If a field doesn't exist, suggest similar available fields from the metadata
- If schema discovery fails, inform the user and ask them to check their connection

Remember: Each DataOS endpoint may have different field names and structure. Always discover first, then query!
"""
        
        return base_prompt + schema_detection_prompt
        
    except FileNotFoundError:
        # Fallback if system.txt not found
        return """You are a Data Research Assistant with dynamic schema adaptation capabilities.

CRITICAL: Always use get_metadata tool first to discover available fields before constructing queries.

Your current endpoint supports dynamic schema detection. The schema could be sales360, customer360, or other variants.
Adapt your field names based on what you discover in the metadata."""

# OpenAI agent turn
def agent_turn(user_text: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Store the query for intelligent recommendations
    st.session_state.last_query = user_text

    # Check if user is requesting a chart
    if detect_chart_request(user_text):
        chart_type = extract_chart_type(user_text)
        
        # Check if user is asking for different data that requires a new query
        if detect_new_data_request(user_text):
            # Don't create chart yet, let the system fetch new data first
            st.info("🔍 Detected request for different data. I'll fetch the appropriate data first, then create your visualization.")
        else:
            # If we have previous data, create chart immediately
            if st.session_state.last_data is not None:
                create_advanced_chart(st.session_state.last_data, chart_type, 
                                    user_query=st.session_state.last_query)
                return f"I've created a {chart_type} chart based on the previous data. The visualization is displayed above with intelligent recommendations!"
            else:
                return "I don't have any recent tabular data to visualize. Please run a query that returns data first, then ask for a chart."

    # Build chat history for OpenAI
    msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    msgs.append({"role": "user", "content": user_text})

    system_prompt = load_system_prompt()
    tools = st.session_state.oai_tools or []

    # First pass: let the model decide whether to call a tool
    first = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        messages=[{"role": "system", "content": system_prompt}] + msgs,
        tools=tools,
        tool_choice="auto",
    )

    msg = first.choices[0].message
    tool_calls = msg.tool_calls or []

    # If the model didn't call a tool, return its text
    if not tool_calls:
        return msg.content or "(no response)"

    # Execute all tool calls with better error handling
    tool_result_msgs = []
    for tc in tool_calls:
        fn = tc.function
        tool_name = fn.name
        try:
            args = json.loads(fn.arguments) if fn.arguments else {}
        except Exception as e:
            args = {}
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"[tool_error] Invalid tool arguments JSON: {e}. Arguments received: {fn.arguments}"
            })
            continue

        try:
            # Special handling for execute_load_query
            if tool_name == "execute_load_query":
                validated_args = validate_load_query_args(args)
                if "error" in validated_args:
                    tool_result_msgs.append({
                        "role": "tool", 
                        "tool_call_id": tc.id,
                        "content": f"[tool_error] {validated_args['error']}"
                    })
                    continue
                
                # Show the query being executed for debugging
                query_json = json.dumps(validated_args, indent=2)
                st.info(f"**Executing Query:**\n```json\n{query_json}\n```")
                args = validated_args

            blocks = call_tool_sync(mcp_url, tool_name, args)
            text_result = mcp_blocks_to_text(blocks)
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": text_result[:120000],
            })
        except Exception as e:
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"[tool_error] {e}"
            })

    # Second pass: give the model the results so it can summarize/respond
    followup = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        messages=[{"role": "system", "content": system_prompt}] + msgs + [
            {"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls}
        ] + tool_result_msgs
    )

    final_response = followup.choices[0].message.content or "(no response)"
    
    # Store response for intelligent recommendations
    st.session_state.last_response = final_response
    
    # Store data for potential visualization
    df = parse_table_from_text(final_response)
    if df is not None:
        st.session_state.last_data = df
        
        # Auto-create visualization if enabled
        if auto_suggest:
            create_advanced_chart(df, "auto", user_query=user_text)

    return final_response

# Input box
user_msg = st.chat_input("Ask anything. I'll pick a tool, run it, and provide intelligent visualizations.")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Analyzing your query and preparing intelligent recommendations..."):
                time.sleep(0.25)
                reply = agent_turn(user_msg)
            placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            placeholder.error(f"Error: {e}")