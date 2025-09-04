import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Any

# Beautiful chart libraries with graceful imports
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from plotly.subplots import make_subplots

from prompts import get_guardrails_prompt

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Optional imports
try:
    from bokeh.embed import file_html
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category20, Viridis256
    from bokeh.plotting import figure
    from bokeh.resources import CDN
    from bokeh.transform import factor_cmap

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    from plotnine import *

    PLOTNINE_AVAILABLE = True
except ImportError:
    PLOTNINE_AVAILABLE = False

from openai import OpenAI

from mcp_async_client import call_tool_sync, list_tools_sync

# Add LIDA imports
try:
    from lida import Manager, TextGenerationConfig, llm

    LIDA_AVAILABLE = True
except ImportError:
    LIDA_AVAILABLE = False
    st.warning(
        "LIDA library not available. Intelligent auto-visualization will be disabled."
    )

load_dotenv()

# Initialize LIDA
lida_manager = None
if LIDA_AVAILABLE and os.getenv("OPENAI_API_KEY"):
    try:
        lida_manager = Manager(text_gen=llm("openai"))
    except Exception as e:
        st.warning(f"Failed to initialize LIDA: {str(e)}")
        lida_manager = None


def clean_response(response):
    cleaned_response = response.strip()
    if cleaned_response.startswith("```json"):
        cleaned_response = cleaned_response[7:]  # Remove ```json
    if cleaned_response.endswith("```"):
        cleaned_response = cleaned_response[:-3]  # Remove ```
    cleaned_response = cleaned_response.strip()
    return cleaned_response


def check_guardrails(query: str, schema: dict[str, Any]) -> dict[str, Any]:
    """
    Run all guardrail checks sequentially.
    Returns a dict with status and message.
    """
    logger.info("=== GUARDRAILS CHECK STARTED ===")
    logger.info(f"Query: {query}")
    logger.info(f"Schema keys: {list(schema.keys()) if schema else 'No schema'}")
    
    try:
        # Initialize OpenAI client
        logger.info("Initializing OpenAI client...")
        openai_model = OpenAI()
        
        # Generate guardrails prompt
        logger.info("Generating guardrails prompt...")
        prompt = get_guardrails_prompt({"question": query, "schema": schema})
        logger.info(f"Prompt generated, length: {len(prompt)} characters")
        
        # Make OpenAI API call
        logger.info("Making OpenAI API call...")
        response = openai_model.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1
        )
        logger.info("OpenAI API call completed successfully")
        
        # Extract response content
        logger.info("Extracting response content...")
        response_content = response.choices[0].message.content
        logger.info(f"Raw response content (first 200 chars): {repr(response_content[:200])}")
        logger.info(f"Raw response content (full): {repr(response_content)}")
        
        # Clean the response
        logger.info("Cleaning response...")
        cleaned_response = clean_response(response_content)
        logger.info(f"Cleaned response (first 200 chars): {repr(cleaned_response[:200])}")
        logger.info(f"Cleaned response (full): {repr(cleaned_response)}")
        
        # Parse JSON
        logger.info("Parsing JSON response...")
        result = json.loads(cleaned_response)
        logger.info(f"Parsed JSON result: {result}")
        
        # Check guardrails result
        status = result.get("status", "PASSED").upper()
        if status == "ERROR":
            logger.warning(f"Guardrails FAILED: {result.get('explanation')}")
            return {
                "status": "error",
                "message": result.get("explanation", "Invalid query. Please retry."),
                "code": "invalid_query",
            }
        
        logger.info(f"Guardrails PASSED: {result.get('explanation', 'No explanation provided')}")
        return {"status": "passed", "message": "Proceed to plan generation."}
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error(f"Failed to parse: {cleaned_response}")
        return {"status": "passed", "message": "Guardrails parsing failed, proceeding anyway."}
        
    except Exception as e:
        logger.error(f"Guardrails check failed with error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return {"status": "passed", "message": "Guardrails error, proceeding anyway."}
    
    finally:
        logger.info("=== GUARDRAILS CHECK COMPLETED ===")


def preprocess_lida_code(lida_code, df_name="data"):
    """
    Enhanced preprocessing for LIDA-generated code to handle common issues:
    - Remove top-level return statements
    - Handle function wrapping
    - Clean semicolons and whitespace
    - Extract function calls if needed
    """
    if not lida_code or not isinstance(lida_code, str):
        return "", False

    logger.info(f"🔧 Preprocessing LIDA code - Original length: {len(lida_code)}")
    logger.info(f"🔧 Original code:\n{lida_code}")

    # Clean the code
    clean_code = lida_code.strip()

    # Remove ALL return statements (more comprehensive approach)
    # This handles cases like "return plt", "return fig", "return chart", etc.
    return_patterns = [
        r"^\s*return\s+[^;]*;?\s*$",  # Return at start of line
        r"\n\s*return\s+[^;]*;?\s*$",  # Return at end after newline
        r"\s*return\s+[^;]*;?\s*$",  # Any trailing return
        r"return\s+plt\s*;?\s*",  # Specific plt return
        r"return\s+fig\s*;?\s*",  # Specific fig return
        r"return\s+chart\s*;?\s*",  # Specific chart return
        r"return\s+[a-zA-Z_][a-zA-Z0-9_]*\s*;?\s*$",  # Any variable return
    ]

    # Apply all patterns to remove return statements
    for pattern in return_patterns:
        before_count = clean_code.count("return")
        clean_code = re.sub(pattern, "", clean_code, flags=re.MULTILINE | re.IGNORECASE)
        after_count = clean_code.count("return")
        if before_count > after_count:
            logger.info(
                f"✅ Applied pattern '{pattern}' - removed {before_count - after_count} return statement(s)"
            )

    # Additional cleanup for any remaining return statements
    lines = clean_code.split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip lines that are just return statements
        stripped_line = line.strip()
        if re.match(r"^\s*return\s+.*$", stripped_line):
            logger.info(f"✅ Removed return line: '{stripped_line}'")
            continue

        # Handle semicolon-separated lines with return statements
        if ";" in line and "return" in line:
            # Split by semicolon and filter out return statements
            parts = line.split(";")
            filtered_parts = []
            for part in parts:
                part_stripped = part.strip()
                if not re.match(r"^\s*return\s+.*$", part_stripped):
                    filtered_parts.append(part)
                else:
                    logger.info(
                        f"✅ Removed return from semicolon line: '{part_stripped}'"
                    )

            # Rejoin the non-return parts
            if filtered_parts:
                cleaned_lines.append(";".join(filtered_parts))
        else:
            cleaned_lines.append(line)

    clean_code = "\n".join(cleaned_lines)

    # Remove trailing semicolons from all lines
    clean_code = re.sub(r";\s*$", "", clean_code, flags=re.MULTILINE)

    # Check if code is wrapped in a function
    is_function = bool(re.search(r"def\s+\w+\s*\([^)]*\):", clean_code))

    logger.info(f"🔧 Is function: {is_function}")

    if is_function:
        # Extract function name
        func_match = re.search(r"def\s+(\w+)\s*\([^)]*\):", clean_code)
        func_name = func_match.group(1) if func_match else "plot"

        # Add function call at the end
        clean_code += f"\n{func_name}({df_name})"
        logger.info(f"✅ Added function call: {func_name}({df_name})")

    # Final cleanup - remove any remaining isolated return statements
    clean_code = re.sub(r"\n\s*return\s+[^;]*;?\s*$", "", clean_code)
    clean_code = re.sub(r"^\s*return\s+[^;]*;?\s*\n", "", clean_code)

    # Add plt.show() if matplotlib is being used and no show() call exists
    if (
        "plt." in clean_code
        and "plt.show()" not in clean_code
        and "st.pyplot" not in clean_code
    ):
        # Don't add plt.show() as it might interfere with Streamlit
        pass

    final_code = clean_code.strip()
    logger.info(f"🔧 Final cleaned code length: {len(final_code)}")
    logger.info(f"🔧 Final cleaned code:\n{final_code}")

    return final_code, is_function


def fix_time_dimension_query(
    base_query, time_dimension, date_range, granularity="month"
):
    """Fix time dimension queries to ensure proper aggregation."""
    fixed_query = base_query.copy()

    # Ensure timeDimensions is properly structured
    if "timeDimensions" not in fixed_query or not fixed_query["timeDimensions"]:
        fixed_query["timeDimensions"] = []

    # Add or update the time dimension
    time_dim = {
        "dimension": time_dimension,
        "dateRange": date_range,
        "granularity": granularity,
    }

    # Replace existing time dimension or add new one
    fixed_query["timeDimensions"] = [time_dim]

    return fixed_query


st.set_page_config(page_title="Agentic MCP Chat (OpenAI)", layout="wide")
st.title("🤖 Agentic MCP Chat with Intelligent Visualization")
st.caption(
    "Ask in natural language; the assistant will pick an MCP tool, build args, run it, and provide intelligent visualization recommendations."
)

# --- Session state initialization (must be before sidebar) ---
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

# --- Config / sidebar ---
DEFAULT_MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    # Tab selection
    tab_selection = st.radio("Navigation", ["⚙️ Settings"], index=0, horizontal=True)

    st.markdown("---")

    if tab_selection == "⚙️ Settings":
        st.header("Settings")
        mcp_url = st.text_input(
            "MCP Server URL",
            value=DEFAULT_MCP_URL,
            help="Default MCP endpoint path is /mcp",
        )
        model_name = st.text_input("OpenAI model", value="gpt-4o")
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

    # Set default values for variables that need to be accessible outside the tab
    if "mcp_url" not in locals():
        mcp_url = DEFAULT_MCP_URL
    if "model_name" not in locals():
        model_name = "gpt-4o-mini"
    if "temp" not in locals():
        temp = 0.2
    if "auto_suggest" not in locals():
        auto_suggest = True
    if "show_chart_options" not in locals():
        show_chart_options = True

# Normalize URL to include /mcp if missing
if mcp_url and not mcp_url.rstrip("/").endswith("/mcp"):
    mcp_url = mcp_url.rstrip("/") + "/mcp"


# Forward declarations for functions used in sidebar
def create_chart_from_sidebar(df, chart_type, library="plotly", title=None):
    """Create chart from sidebar configuration - wrapper function."""
    try:
        # Import the functions here to avoid circular dependency
        if chart_type in [
            "bar",
            "line",
            "scatter",
            "pie",
            "area",
            "heatmap",
            "bubble",
            "treemap",
        ]:
            # Simple chart creation using Plotly
            fig = None
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

            if chart_type == "bar" and categorical_cols and numeric_cols:
                fig = px.bar(
                    df,
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title=title or "Bar Chart",
                )
            elif chart_type == "line" and numeric_cols:
                if categorical_cols:
                    fig = px.line(
                        df,
                        x=categorical_cols[0],
                        y=numeric_cols[0],
                        title=title or "Line Chart",
                    )
                else:
                    fig = px.line(df, y=numeric_cols[0], title=title or "Line Chart")
            elif chart_type == "scatter" and len(numeric_cols) >= 2:
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=title or "Scatter Plot",
                )
            elif chart_type == "pie" and categorical_cols and numeric_cols:
                fig = px.pie(
                    df,
                    values=numeric_cols[0],
                    names=categorical_cols[0],
                    title=title or "Pie Chart",
                )

            if fig:
                st.plotly_chart(fig, width="stretch")
                return f"✅ Created {chart_type} chart"
            else:
                st.warning(
                    f"Cannot create {chart_type} chart with current data structure"
                )
                return f"⚠️ Cannot create {chart_type} chart"
        else:
            # For complex chart types, just show a simple version
            if numeric_cols:
                fig = px.bar(
                    df, y=numeric_cols[0], title=title or f"{chart_type.title()} Chart"
                )
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
        oai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": schema,
                },
            }
        )
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
        "limit": query.get("limit", 100),
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
        elif (
            not isinstance(time_dim["dateRange"], list)
            or len(time_dim["dateRange"]) != 2
        ):
            return {
                "error": f"timeDimensions[{i}].dateRange must be a list with start and end dates"
            }

        # Ensure granularity exists
        if "granularity" not in time_dim:
            time_dim["granularity"] = "month"
        elif time_dim["granularity"] not in ["day", "week", "month", "quarter", "year"]:
            return {
                "error": f"timeDimensions[{i}].granularity must be one of: day, week, month, quarter, year"
            }

    # Convert old-style date filters to timeDimensions format for backward compatibility
    new_filters = []
    for filter_item in fixed_query["filters"]:
        if isinstance(filter_item, dict):
            # Check if it's a date filter that should be converted to timeDimensions
            if "member" in filter_item and "order_date" in filter_item.get(
                "member", ""
            ):
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
        return {
            "error": "Query must include at least one measure. Use get_metadata to discover available measures in the current schema."
        }

    return {"query": fixed_query}


def parse_table_from_text(text):
    """Enhanced table parsing from assistant response and convert to DataFrame."""
    try:
        # Find markdown table pattern
        table_pattern = r"\|.*\|[\r\n]+\|.*\|[\r\n]+(?:\|.*\|[\r\n]+)+"
        table_match = re.search(table_pattern, text)

        if not table_match:
            return None

        table_text = table_match.group(0)
        lines = [line.strip() for line in table_text.split("\n") if line.strip()]

        if len(lines) < 3:  # Header, separator, at least one data row
            return None

        # Parse header - clean up header names
        header_line = lines[0]
        header = []
        for col in header_line.split("|")[1:-1]:  # Remove first and last empty elements
            clean_header = col.strip()
            # Handle common header variations
            if clean_header.lower() in ["product category", "productcategory"]:
                clean_header = "Product Category"
            elif clean_header.lower() in ["average order value", "averageordervalue"]:
                clean_header = "Average Order Value"
            header.append(clean_header)

        # Parse data rows (skip separator line)
        data = []
        for line in lines[2:]:
            row_data = [col.strip() for col in line.split("|")[1:-1]]
            if len(row_data) == len(header):
                data.append(row_data)

        if not data:
            return None

        df = pd.DataFrame(data, columns=header)

        # Enhanced data type conversion
        for col in df.columns:
            # Handle currency values (remove $ and commas)
            if df[col].astype(str).str.contains(r"[\$,]").any():
                df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True)

            # Try to convert to numeric
            try:
                # Check if all values can be converted to float
                numeric_series = pd.to_numeric(df[col], errors="coerce")
                if (
                    not numeric_series.isna().all()
                ):  # If at least some values are numeric
                    df[col] = numeric_series
            except (ValueError, TypeError):
                # Try to parse dates
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except:
                    pass  # Keep as string

        # Log the parsing result
        print(
            f"Successfully parsed table with {len(df)} rows and columns: {list(df.columns)}"
        )
        print(f"Data types: {dict(df.dtypes)}")
        print(f"Sample data:\n{df.head()}")

        return df

    except Exception as e:
        print(f"Error parsing table: {e}")
        return None


def create_advanced_chart(df, chart_type="auto", title=None, user_query=""):
    """Create advanced charts using LIDA with COMPLETE AUTHORITY for beautiful visualizations."""
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return

    try:
        # LIDA FIRST AND ONLY - Give it complete freedom for beauty!
        if lida_manager:
            with st.spinner(
                "🎨 LIDA is creating the most beautiful visualization for your data..."
            ):
                # NO LIBRARY CONSTRAINTS - LIDA chooses the most beautiful approach
                lida_result = lida_manager.visualize(
                    data=df,
                    summary=f"Create the most beautiful, visually appealing, and professional-looking visualization for this data. User query: {user_query}",
                    # REMOVED: library="altair" - LIDA now has COMPLETE FREEDOM!
                )

                if lida_result:
                    st.success("✅ LIDA created a beautiful visualization!")

                    # Check if LIDA returned code that needs preprocessing
                    if hasattr(lida_result, "code") and lida_result.code:
                        logger.info(
                            f"🔍 LIDA chose to generate code: {lida_result.code[:100]}..."
                        )

                        # Preprocess the code to handle return statements
                        clean_code, is_function = preprocess_lida_code(
                            lida_result.code, "df"
                        )

                        if clean_code:
                            try:
                                # Execute with ALL libraries available for LIDA's beautiful choice
                                exec_globals = {
                                    "df": df,
                                    "data": df,
                                    "pd": pd,
                                    "plt": plt,
                                    "sns": sns,
                                    "px": px,
                                    "alt": alt,
                                    "np": np,
                                    "go": go if "go" in globals() else None,
                                    "make_subplots": make_subplots
                                    if "make_subplots" in globals()
                                    else None,
                                }
                                exec(clean_code, exec_globals)

                                # Try to display matplotlib plot
                                current_fig = plt.gcf()
                                if current_fig.get_axes():
                                    st.pyplot(current_fig)
                                    plt.clf()
                                else:
                                    st.warning("No plot was generated")
                            except Exception as exec_error:
                                logger.error(f"❌ Execution error: {exec_error}")
                                st.error(f"Error executing LIDA code: {exec_error}")
                        else:
                            st.warning("No valid code generated after preprocessing")

                    # Display the visualization (original LIDA chart if available)
                    elif hasattr(lida_result, "chart"):
                        st.altair_chart(lida_result.chart, width="stretch")

                    # Show LIDA's reasoning
                    if hasattr(lida_result, "summary"):
                        with st.expander(
                            "🎨 LIDA's Beautiful Creation", expanded=False
                        ):
                            st.write(lida_result.summary)
                else:
                    st.error("❌ LIDA could not create a visualization")
        else:
            st.error("❌ LIDA not available. Please check OpenAI API key in Settings.")
            st.info("💡 LIDA creates the most beautiful and appealing visualizations!")

        # Show the underlying data
        with st.expander("📋 View Raw Data"):
            st.dataframe(df, width="stretch")

    except Exception as e:
        st.error(f"Error creating chart: {e}")
        st.write("Data preview:")
        st.dataframe(df.head())


def create_plotly_chart(df, chart_type):
    """Create Plotly charts based on chart type."""
    try:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

        if chart_type == "line":
            if datetime_cols and numeric_cols:
                fig = px.line(
                    df,
                    x=datetime_cols[0],
                    y=numeric_cols[0],
                    title="Time Series Analysis",
                )
            elif categorical_cols and numeric_cols:
                fig = px.line(
                    df, x=categorical_cols[0], y=numeric_cols[0], title="Trend Analysis"
                )
            else:
                st.warning(
                    "Line chart needs time or categorical data with numeric values"
                )
                return

        elif chart_type == "area":
            if datetime_cols and numeric_cols:
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df[datetime_cols[0]],
                        y=df[numeric_cols[0]],
                        fill="tonexty",
                        mode="lines",
                        name=numeric_cols[0],
                    )
                )
                fig.update_layout(title="Area Chart - Magnitude Over Time")
            else:
                st.warning("Area chart needs time data with numeric values")
                return

        elif chart_type == "bar" or chart_type == "horizontal_bar":
            if categorical_cols and numeric_cols:
                orientation = "h" if chart_type == "horizontal_bar" else "v"
                fig = px.bar(
                    df,
                    x=categorical_cols[0] if orientation == "v" else numeric_cols[0],
                    y=numeric_cols[0] if orientation == "v" else categorical_cols[0],
                    orientation=orientation,
                    title="Categorical Comparison",
                )
            else:
                st.warning("Bar chart needs categorical and numeric data")
                return

        elif chart_type == "pie" or chart_type == "donut":
            if categorical_cols and numeric_cols:
                fig = px.pie(
                    df,
                    values=numeric_cols[0],
                    names=categorical_cols[0],
                    hole=0.3 if chart_type == "donut" else 0,
                    title="Proportional Distribution",
                )
            else:
                st.warning("Pie chart needs categorical and numeric data")
                return

        elif chart_type == "scatter":
            if len(numeric_cols) >= 2:
                color_col = categorical_cols[0] if categorical_cols else None
                size_col = numeric_cols[2] if len(numeric_cols) >= 3 else None

                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    color=color_col,
                    size=size_col,
                    title="Correlation Analysis",
                    trendline="ols",
                )
            else:
                st.warning("Scatter plot needs at least 2 numeric columns")
                return

        elif chart_type == "bubble":
            if len(numeric_cols) >= 2:
                size_col = (
                    numeric_cols[2] if len(numeric_cols) >= 3 else numeric_cols[0]
                )
                color_col = categorical_cols[0] if categorical_cols else None

                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    size=size_col,
                    color=color_col,
                    title="Multi-Variable Bubble Analysis",
                )
            else:
                st.warning("Bubble chart needs at least 2 numeric columns")
                return

        elif chart_type == "treemap":
            if categorical_cols and numeric_cols:
                fig = px.treemap(
                    df,
                    path=[categorical_cols[0]],
                    values=numeric_cols[0],
                    title="Hierarchical Proportions",
                )
            else:
                st.warning("Treemap needs categorical and numeric data")
                return

        elif chart_type == "heatmap":
            if len(numeric_cols) >= 2:
                # Create correlation heatmap
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    title="Correlation Heatmap",
                )
            else:
                st.warning("Heatmap needs multiple numeric columns")
                return

        else:
            # Default to bar chart
            if categorical_cols and numeric_cols:
                fig = px.bar(
                    df,
                    x=categorical_cols[0],
                    y=numeric_cols[0],
                    title="Data Visualization",
                )
            else:
                st.warning("Cannot determine appropriate chart type for this data")
                return

        # Update layout for better appearance
        fig.update_layout(
            height=500,
            showlegend=True,
            font=dict(size=12),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error creating {chart_type} chart: {e}")
        st.dataframe(df.head())


def get_best_library_for_chart_type(chart_type):
    """Get the best Python library for each visualization type based on best practices."""
    library_recommendations = {
        "bar": "seaborn",  # High-level API, beautiful default styles
        "line": "matplotlib",  # Flexible, publication-quality, time series
        "scatter": "plotly",  # Interactivity, zoom, selection
        "histogram": "seaborn",  # Statistical features, easy syntax
        "box": "seaborn",  # Statistical features, easy grouping
        "heatmap": "seaborn",  # Annotated, clustering, color palettes
        "pie": "plotly",  # Interactive, easy labels
        "donut": "plotly",  # Interactive, easy labels
        "area": "plotly",  # Stacked areas, interactivity
        "violin": "seaborn",  # Statistical, easy grouping
        "3d": "plotly",  # True 3D support, interactive
        "treemap": "plotly",  # Hierarchical, interactive
        "sunburst": "plotly",  # Hierarchical, interactive
        "facet": "altair",  # Declarative grammar, simple faceting
        "network": "plotly",  # NetworkX for data, Plotly for visualization
        "map": "plotly",  # Easy maps, choropleths, scatter geo
        "statistical": "seaborn",  # Built-in stats, regression, correlation
        "interactive": "plotly",  # Best overall for interactivity
        "waterfall": "plotly",  # Good support for waterfall charts
        "radar": "plotly",  # Good radar chart support
    }

    return library_recommendations.get(chart_type.lower(), "plotly")


def generate_lida_instructions_and_persona(df, user_query=""):
    """Generate intelligent instructions and persona for LIDA using LLM."""
    if not lida_manager:
        return [], ""

    try:
        # Analyze the data structure
        data_info = {
            "columns": list(df.columns),
            "numeric_cols": df.select_dtypes(include=["number"]).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=["object"]).columns.tolist(),
            "datetime_cols": df.select_dtypes(include=["datetime64"]).columns.tolist(),
            "shape": df.shape,
            "sample_data": df.head(2).to_dict("records"),
        }

        # Create instruction generation prompt
        instruction_prompt = f"""
Based on this data analysis request and dataset characteristics, generate 2-3 specific visualization instructions:

USER QUERY: "{user_query}"

DATASET INFO:
- Columns: {data_info["columns"]}
- Numeric columns: {data_info["numeric_cols"]}
- Categorical columns: {data_info["categorical_cols"]}
- Data shape: {data_info["shape"]}
- Sample: {data_info["sample_data"]}

Generate 2-3 specific instructions to make the visualization more beautiful and insightful. Focus on:
1. Visual aesthetics (colors, styling, layout)
2. Data clarity (labels, formatting, readability)
3. Professional appearance (titles, legends, annotations)

Return only the instructions as a simple list, one per line, starting with "- ".
"""

        # Create persona generation prompt
        persona_prompt = f"""
Based on this data analysis request, generate a specific persona who would be interested in this visualization:

USER QUERY: "{user_query}"
DATASET CONTEXT: Data about {", ".join(data_info["columns"][:3])}

Generate a specific persona (role, goals, context) who would want to see this data visualization. 
Be specific about their role, why they need this data, and what decisions they might make.

Return only the persona description in 1-2 sentences, focusing on their role and data needs.
"""

        # Generate instructions using OpenAI
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Generate instructions
            instructions_response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[{"role": "user", "content": instruction_prompt}],
                max_tokens=200,
            )

            instructions_text = instructions_response.choices[0].message.content
            instructions = [
                line.strip("- ").strip()
                for line in instructions_text.split("\n")
                if line.strip() and line.strip().startswith("-")
            ]

            # Generate persona
            persona_response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.8,
                messages=[{"role": "user", "content": persona_prompt}],
                max_tokens=100,
            )

            persona = persona_response.choices[0].message.content.strip()

            logger.info(f"🎭 Generated persona: {persona}")
            logger.info(f"📝 Generated instructions: {instructions}")

            return instructions[:3], persona  # Limit to 3 instructions

        except Exception as llm_error:
            logger.warning(f"LLM generation failed: {llm_error}")
            # Fallback to basic instructions
            return [
                "use professional color palette with high contrast",
                "add clear axis labels and title",
                "ensure the chart is publication-ready with proper formatting",
            ], f"data analyst interested in {user_query.lower()}"

    except Exception as e:
        logger.error(f"Error generating instructions/persona: {e}")
        return [], ""


def enhanced_visualization_pipeline(df, user_query="", enable_lida=True):
    """
    Enhanced visualization pipeline with LIDA having COMPLETE AUTHORITY over library selection
    for creating the most beautiful and appealing visualizations.
    """
    results = {
        "lida_success": False,
        "fallback_used": False,
        "chart_created": False,
        "chart_description": "",
    }

    if df is None or df.empty:
        results["chart_description"] = "No data available"
        return results

    # LIDA FIRST AND ONLY CHOICE - Give it complete freedom!
    if enable_lida and lida_manager is not None:
        try:
            logger.info(
                "� LIDA has COMPLETE AUTHORITY to create beautiful visualizations..."
            )
            logger.info(f"� User query: {user_query}")

            # Create LIDA summary
            summary = lida_manager.summarize(df)

            # Generate goal - let LIDA choose EVERYTHING (library, chart type, style)
            if user_query:
                # Enhanced goal that gives LIDA maximum freedom
                enhanced_goal = user_query

                # Only add specific instructions if user explicitly requests a chart type
                chart_type_keywords = [
                    "pie chart",
                    "bar chart",
                    "line chart",
                    "scatter plot",
                    "histogram",
                    "heatmap",
                    "treemap",
                    "donut chart",
                    "area chart",
                    "box plot",
                    "violin plot",
                    "sunburst",
                    "waterfall",
                    "radar",
                ]

                user_requested_specific_chart = any(
                    keyword in user_query.lower() for keyword in chart_type_keywords
                )

                if user_requested_specific_chart:
                    # User specifically requested a chart type - honor that
                    if "pie" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing pie chart with proper colors and styling."
                    elif "bar" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing bar chart with proper colors and styling."
                    elif "line" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing line chart with proper colors and styling."
                    elif "scatter" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing scatter plot with proper colors and styling."
                    elif "histogram" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing histogram with proper colors and styling."
                    elif "heatmap" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing heatmap with proper colors and styling."
                    elif "treemap" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing treemap with proper colors and styling."
                    elif "donut" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing donut chart with proper colors and styling."
                    elif "area" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing area chart with proper colors and styling."
                    elif "box" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing box plot with proper colors and styling."
                    elif "violin" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing violin plot with proper colors and styling."
                    elif "sunburst" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing sunburst chart with proper colors and styling."
                    elif "waterfall" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing waterfall chart with proper colors and styling."
                    elif "radar" in user_query.lower():
                        enhanced_goal += ". Create a beautiful, visually appealing radar chart with proper colors and styling."
                    # Add more specific chart types as needed
                else:
                    # No specific chart type requested - let LIDA choose completely!
                    enhanced_goal += ". Create the most beautiful, visually appealing, and professional-looking visualization for this data. Choose the best chart type, library, colors, and styling to make it stunning and insightful."

                goal = enhanced_goal
            else:
                # No user query - let LIDA choose completely
                goals = lida_manager.goals(summary, n=1)
                goal = (
                    goals[0]
                    if goals
                    else "Create the most beautiful and visually appealing visualization for this data"
                )

            logger.info(f"� Final goal for LIDA: {goal}")

            # Generate visualization - LIDA HAS COMPLETE FREEDOM!
            # NO LIBRARY CONSTRAINTS - LIDA chooses the most beautiful approach
            visualizations = lida_manager.visualize(summary=summary, goal=goal)

            if visualizations:
                for viz in visualizations:
                    if hasattr(viz, "code") and viz.code:
                        # Preprocess and execute - let LIDA's choice shine!
                        clean_code, is_function = preprocess_lida_code(viz.code, "data")

                        if clean_code:
                            # Execute with ALL libraries available for LIDA's choice
                            exec_globals = {
                                "data": df,
                                "df": df,
                                "pd": pd,
                                "plt": plt,
                                "sns": sns,
                                "px": px,
                                "alt": alt,
                                "np": np,
                                # Add more libraries if available
                                "go": go if "go" in globals() else None,
                                "make_subplots": make_subplots
                                if "make_subplots" in globals()
                                else None,
                            }

                            exec(clean_code, exec_globals)

                            # Check if plot was created
                            current_fig = plt.gcf()
                            if current_fig.get_axes():
                                st.pyplot(current_fig)
                                plt.clf()

                                results["lida_success"] = True
                                results["chart_created"] = True
                                results["chart_description"] = (
                                    f"LIDA created a beautiful visualization: {goal}"
                                )
                                return results

                    elif hasattr(viz, "chart"):
                        # LIDA chose Altair - display it beautifully
                        st.altair_chart(viz.chart, width="stretch")
                        results["lida_success"] = True
                        results["chart_created"] = True
                        results["chart_description"] = (
                            f"LIDA created a beautiful Altair visualization: {goal}"
                        )
                        return results

        except Exception as e:
            logger.warning(f"LIDA failed: {e}")

    # MINIMAL FALLBACK - Only if LIDA is completely unavailable
    results["fallback_used"] = True
    logger.info("🔄 LIDA unavailable, using minimal fallback...")

    try:
        # Simple, clean fallback - don't compete with LIDA's beauty
        st.info(
            "💡 For the most beautiful visualizations, please ensure LIDA is properly configured with your OpenAI API key."
        )

        # Show data table as clean fallback
        st.dataframe(df, width="stretch")
        results["chart_description"] = (
            "Data table displayed (LIDA not available for beautiful visualizations)"
        )

    except Exception as e:
        logger.error(f"Minimal fallback failed: {e}")
        results["chart_description"] = "Unable to display data"

    return results


def debug_lida_integration(df, user_query=""):
    """Debug LIDA integration and return diagnostic information."""
    debug_info = {"success": False, "steps": [], "errors": [], "raw_code": None}

    try:
        debug_info["steps"].append("✅ Starting LIDA debug")

        if lida_manager is None:
            debug_info["errors"].append("❌ LIDA manager not initialized")
            return debug_info

        debug_info["steps"].append("✅ LIDA manager available")

        # Test summary generation
        try:
            summary = lida_manager.summarize(df)
            debug_info["steps"].append("✅ Summary generated")
        except Exception as e:
            debug_info["errors"].append(f"❌ Summary generation failed: {e}")
            return debug_info

        # Test goal generation
        try:
            goals = lida_manager.goals(summary, n=1)
            goal = goals[0] if goals else user_query or "Show data distribution"
            debug_info["steps"].append("✅ Goals generated")
        except Exception as e:
            debug_info["errors"].append(f"❌ Goal generation failed: {e}")
            return debug_info

        # Test visualization generation
        try:
            visualizations = lida_manager.visualize(summary=summary, goal=goal)
            debug_info["steps"].append("✅ Visualizations generated")

            if visualizations:
                viz = visualizations[0]
                if hasattr(viz, "code") and viz.code:
                    debug_info["raw_code"] = viz.code
                    debug_info["steps"].append("✅ Code extracted from visualization")

                    # Test preprocessing
                    try:
                        clean_code, is_function = preprocess_lida_code(viz.code, "data")
                        debug_info["steps"].append("✅ Code preprocessing successful")

                        if clean_code:
                            debug_info["steps"].append("✅ Clean code generated")

                            # Test execution
                            try:
                                exec_globals = {
                                    "data": df,
                                    "df": df,
                                    "pd": pd,
                                    "plt": plt,
                                    "sns": sns,
                                    "px": px,
                                    "alt": alt,
                                    "np": np,
                                }
                                exec(clean_code, exec_globals)
                                debug_info["steps"].append(
                                    "✅ Code execution successful"
                                )
                                debug_info["success"] = True

                            except Exception as e:
                                debug_info["errors"].append(
                                    f"❌ Code execution failed: {e}"
                                )
                        else:
                            debug_info["errors"].append("❌ No clean code generated")

                    except Exception as e:
                        debug_info["errors"].append(
                            f"❌ Code preprocessing failed: {e}"
                        )
                else:
                    debug_info["errors"].append("❌ No code in visualization")
            else:
                debug_info["errors"].append("❌ No visualizations generated")

        except Exception as e:
            debug_info["errors"].append(f"❌ Visualization generation failed: {e}")

    except Exception as e:
        debug_info["errors"].append(f"❌ Unexpected error: {e}")

    return debug_info


def create_simple_fallback_chart(df):
    """Create a simple fallback chart when LIDA fails."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        print(f"Creating fallback chart for DataFrame with columns: {list(df.columns)}")
        print(f"Data types: {dict(df.dtypes)}")

        # Find numeric and categorical columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        print(f"Numeric columns: {numeric_cols}")
        print(f"Categorical columns: {categorical_cols}")

        # Create appropriate chart based on data structure
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            # Bar chart: categorical vs numeric
            fig = px.bar(
                df,
                x=categorical_cols[0],
                y=numeric_cols[0],
                title=f"{numeric_cols[0]} by {categorical_cols[0]}",
            )
            # Fix: Use update_layout instead of update_xaxis for tickangle
            fig.update_layout(xaxis_tickangle=45)
            return fig

        elif len(numeric_cols) >= 2:
            # Scatter plot: numeric vs numeric
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"{numeric_cols[1]} vs {numeric_cols[0]}",
            )
            return fig

        elif len(numeric_cols) == 1:
            # Histogram for single numeric column
            fig = px.histogram(
                df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}"
            )
            return fig

        else:
            # Fallback: show data summary as text
            return None

    except Exception as e:
        print(f"Error creating fallback chart: {e}")
        return None


def run_lida_diagnostics():
    """Run comprehensive LIDA diagnostics."""
    st.markdown("### 🔍 LIDA Integration Diagnostics")

    # Test 1: LIDA availability
    if lida_manager is None:
        st.error("❌ LIDA Manager is not initialized")
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OPENAI_API_KEY not found in environment")
        else:
            st.info("✅ OPENAI_API_KEY found")
        return False
    else:
        st.success("✅ LIDA Manager initialized")

    # Test 2: Create sample data
    sample_df = pd.DataFrame(
        {
            "Category": ["A", "B", "C", "D"],
            "Value": [10, 20, 15, 25],
            "Score": [0.8, 0.6, 0.9, 0.7],
        }
    )

    st.write("**Sample Data:**")
    st.dataframe(sample_df)

    # Test 3: Run debug pipeline
    st.write("**Running Debug Pipeline:**")
    debug_info = debug_lida_integration(sample_df, "Show category performance")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Steps Completed:**")
        for step in debug_info.get("steps", []):
            st.write(f"  {step}")

    with col2:
        st.write("**Errors Found:**")
        for error in debug_info.get("errors", []):
            st.write(f"  {error}")

    # Test 4: Show raw code if available
    if debug_info.get("raw_code"):
        st.write("**Generated Code:**")
        st.code(debug_info["raw_code"], language="python")

        # Analyze the code
        code_issues = []
        if "return" in debug_info["raw_code"]:
            code_issues.append("⚠️ Contains return statements")
        if ";" in debug_info["raw_code"]:
            code_issues.append("⚠️ Contains semicolons")
        if not any(
            lib in debug_info["raw_code"] for lib in ["plt", "px", "alt", "sns"]
        ):
            code_issues.append("⚠️ No recognized plotting library")

        if code_issues:
            st.write("**Code Issues:**")
            for issue in code_issues:
                st.write(f"  {issue}")
        else:
            st.success("✅ Code looks syntactically correct")

    return debug_info.get("success", False)


def show_manual_chart_options(df):
    """Show manual chart creation options as a fallback."""
    st.markdown("### 🛠️ Manual Chart Options")

    col1, col2 = st.columns(2)

    with col1:
        chart_type = st.selectbox(
            "Chart Type",
            ["bar", "line", "scatter", "pie", "histogram", "box", "heatmap"],
            help="Select the type of chart to create",
        )

    with col2:
        library = st.selectbox(
            "Chart Library",
            ["plotly", "altair"],
            help="Choose the visualization library",
        )

    if st.button("🎨 Create Chart"):
        if library == "plotly":
            fig = create_plotly_chart(df, chart_type)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.success(f"✅ Created {chart_type} chart with Plotly")
        elif library == "altair":
            # Simple Altair charts
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist()

                if chart_type == "bar" and categorical_cols and numeric_cols:
                    chart = (
                        alt.Chart(df)
                        .mark_bar()
                        .encode(x=categorical_cols[0], y=numeric_cols[0])
                        .properties(width=600, height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)
                elif chart_type == "scatter" and len(numeric_cols) >= 2:
                    chart = (
                        alt.Chart(df)
                        .mark_circle()
                        .encode(x=numeric_cols[0], y=numeric_cols[1])
                        .properties(width=600, height=400)
                    )
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning("Chart type not supported with current data structure")
            except Exception as e:
                st.error(f"Error creating Altair chart: {e}")


def create_simple_chart(df, chart_type="auto"):
    """Create simple charts as fallback when LIDA is not available."""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if chart_type == "auto":
            # Auto-select based on data
            if len(numeric_cols) >= 2:
                chart_type = "scatter"
            elif numeric_cols and categorical_cols:
                chart_type = "bar"
            elif numeric_cols:
                chart_type = "histogram"
            else:
                chart_type = "bar"

        # Create simple Plotly chart
        fig = create_plotly_chart(df, chart_type)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            return f"✅ Created simple {chart_type} chart"
        else:
            st.warning("Could not create chart with available data")
            return None

    except Exception as e:
        st.error(f"Error creating simple chart: {e}")
        return None


# ===============================================================================

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
        st.info(
            "Smart suggestions are now handled automatically by LIDA when you query data!"
        )

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
                        out.append(
                            json.dumps(
                                getattr(b, "__dict__", str(b)), ensure_ascii=False
                            )
                        )
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


def agent_turn(user_text: str) -> str:
    """Agent turn that processes user requests naturally via system prompt."""
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    logger.info(f"=== AGENT TURN STARTED FOR: {user_text} ===")

    # Load schema
    try:
        with open("./schema.json", "r", encoding="utf-8") as f:
            schema = json.load(f)
        logger.info(f"Schema loaded successfully with {len(schema)} keys")
    except Exception as e:
        schema = {}
        logger.warning(f"Could not load schema: {e}")

    # Run guardrails check (single validation)
    logger.info("Calling guardrails check...")
    guard_dict = check_guardrails(user_text, schema)
    logger.info(f"Guardrails result: {guard_dict}")

    if guard_dict.get("status") == "error":
        return f"🚫 {guard_dict.get('message', 'Your query was blocked by guardrails.')}"

    logger.info("Guardrails passed, proceeding with agent processing...")

    # Let the system prompt handle the interaction flow naturally
    execution_result = execute_query(user_text, schema)
    return execution_result


def execute_query(user_text: str, schema: dict) -> str:
    """Execute the actual query after plan confirmation."""
    logger.info(f"=== EXECUTING QUERY: {user_text} ===")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Store the query for LIDA visualization
    st.session_state.last_query = user_text

    # Build chat history for OpenAI
    msgs = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
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
            tool_result_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"[tool_error] Invalid tool arguments JSON: {e}. Arguments received: {fn.arguments}",
                }
            )
            continue

        try:
            # Special handling for execute_load_query
            if tool_name == "execute_load_query":
                validated_args = validate_load_query_args(args)
                if "error" in validated_args:
                    tool_result_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"[tool_error] {validated_args['error']}",
                        }
                    )
                    continue

                # Show the query being executed for debugging
                query_json = json.dumps(validated_args, indent=2)
                st.info(f"**Executing Query:**\n```json\n{query_json}\n```")
                args = validated_args

            blocks = call_tool_sync(mcp_url, tool_name, args)
            text_result = mcp_blocks_to_text(blocks)
            tool_result_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": text_result[:120000],
                }
            )
        except Exception as e:
            tool_result_msgs.append(
                {"role": "tool", "tool_call_id": tc.id, "content": f"[tool_error] {e}"}
            )

    # Second pass: give the model the results so it can summarize/respond
    followup = client.chat.completions.create(
        model=model_name,
        temperature=temp,
        messages=[{"role": "system", "content": system_prompt}]
        + msgs
        + [
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            }
        ]
        + tool_result_msgs,
    )

    final_response = followup.choices[0].message.content or "(no response)"

    # Store response for potential visualization
    st.session_state.last_response = final_response

    # Parse data from response and create LIDA visualization
    df = parse_table_from_text(final_response)
    if df is not None:
        st.session_state.last_data = df

        # Auto-create LIDA visualization if enabled
        if auto_suggest:
            st.markdown("---")
            st.markdown("### 🤖 LIDA Auto-Visualization")
            result = enhanced_visualization_pipeline(df, user_query=user_text)
            if result:
                st.success(result)
            else:
                # Fallback to simple chart
                st.info("Falling back to simple visualization...")
                create_simple_chart(df, "auto")

    return final_response


# Input box
user_msg = st.chat_input(
    "Ask anything. I'll show you my execution plan and then proceed with the analysis."
)
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("🎨 Analyzing your query and creating beautiful visualizations..."):
            time.sleep(0.25)
            reply = agent_turn(user_msg)

        st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
