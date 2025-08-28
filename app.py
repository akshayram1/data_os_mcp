import os
import json
import time
import re
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from openai import OpenAI
from mcp_async_client import list_tools_sync, call_tool_sync

load_dotenv()

st.set_page_config(page_title="Agentic MCP Chat (OpenAI)", layout="centered")
st.title("🤖 Agentic MCP Chat (OpenAI)")
st.caption("Ask in natural language; the assistant will pick an MCP tool, build args, run it, and summarize the result.")

# --- Config / sidebar ---
DEFAULT_MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.header("Settings")
    mcp_url = st.text_input("MCP Server URL", value=DEFAULT_MCP_URL, help="Default MCP endpoint path is /mcp")
    model_name = st.text_input("OpenAI model", value="gpt-4o-mini")
    temp = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.markdown("---")
    if not OPENAI_API_KEY:
        st.error("Add OPENAI_API_KEY to your .env")

# Normalize URL to include /mcp if missing
if mcp_url and not mcp_url.rstrip("/").endswith("/mcp"):
    mcp_url = mcp_url.rstrip("/") + "/mcp"

# --- Session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role":"user"/"assistant","content":"..."}]
if "oai_tools" not in st.session_state:
    st.session_state.oai_tools = None  # OpenAI tool spec list
if "raw_tools" not in st.session_state:
    st.session_state.raw_tools = []  # MCP tools list
if "last_data" not in st.session_state:
    st.session_state.last_data = None  # Store last retrieved data for visualization
if "pending_chart" not in st.session_state:
    st.session_state.pending_chart = False  # Flag to indicate user wants a chart

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
    """
    Validate and fix common issues with execute_load_query arguments.
    """
    if not isinstance(args, dict):
        return {"error": "Arguments must be a dictionary"}
    
    query = args.get("query", {})
    if not isinstance(query, dict):
        return {"error": "Query must be a dictionary object"}
    
    # Ensure required keys exist with proper defaults
    fixed_query = {
        "dimensions": query.get("dimensions", []),
        "measures": query.get("measures", []),
        "filters": query.get("filters", []),
        "limit": query.get("limit", 100)
    }
    
    # Validate dimensions and measures are lists
    if not isinstance(fixed_query["dimensions"], list):
        fixed_query["dimensions"] = []
    if not isinstance(fixed_query["measures"], list):
        fixed_query["measures"] = []
    if not isinstance(fixed_query["filters"], list):
        fixed_query["filters"] = []
    
    # Ensure limit is an integer
    try:
        fixed_query["limit"] = int(fixed_query["limit"])
    except (ValueError, TypeError):
        fixed_query["limit"] = 100
    
    # Validation: Must have at least one measure
    if not fixed_query["measures"]:
        return {"error": "Query must include at least one measure. Available measures: customer.total_customers, proposal.total_proposal, sales.total_sales, sales.total_revenue"}
    
    return {"query": fixed_query}

def parse_table_from_text(text):
    """
    Parse markdown table from assistant response and convert to DataFrame.
    """
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

def create_chart(df, chart_type="auto"):
    """
    Create a chart based on the DataFrame and chart type.
    """
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return
    
    try:
        # Auto-detect chart type based on data
        if chart_type == "auto":
            # If we have datetime column, use line chart
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                chart_type = "line"
            elif len(numeric_cols) >= 1:
                chart_type = "bar"
            else:
                chart_type = "bar"
        
        st.subheader("📊 Data Visualization")
        
        # Create the appropriate chart
        if chart_type == "line":
            # For line charts, use the first datetime/text column as x-axis
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                x_col = datetime_cols[0]
            else:
                x_col = df.columns[0]
            
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                chart_data = df.set_index(x_col)[numeric_cols]
                st.line_chart(chart_data)
            else:
                st.warning("No numeric data found for line chart")
                
        elif chart_type == "bar":
            # For bar charts, use first column as categories
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                if len(df.columns) > 1:
                    chart_data = df.set_index(df.columns[0])[numeric_cols]
                else:
                    chart_data = df[numeric_cols]
                st.bar_chart(chart_data)
            else:
                st.warning("No numeric data found for bar chart")
                
        elif chart_type == "pie":
            # For pie charts, need exactly 2 columns (labels and values)
            if len(df.columns) >= 2:
                numeric_col = df.select_dtypes(include=['number']).columns[0]
                label_col = df.columns[0] if df.columns[0] != numeric_col else df.columns[1]
                
                import plotly.express as px
                fig = px.pie(df, values=numeric_col, names=label_col)
                st.plotly_chart(fig)
            else:
                st.warning("Pie chart needs at least 2 columns (categories and values)")
                
        # Show the underlying data
        with st.expander("📋 View Raw Data"):
            st.dataframe(df)
            
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        st.write("Data preview:")
        st.dataframe(df.head())

def detect_chart_request(user_message):
    """
    Detect if user is requesting a chart/visualization.
    """
    chart_keywords = ['chart', 'graph', 'plot', 'visualize', 'visualization', 'show chart', 'create chart', 'bar chart', 'line chart', 'pie chart']
    return any(keyword in user_message.lower() for keyword in chart_keywords)

def extract_chart_type(user_message):
    """
    Extract requested chart type from user message.
    """
    message_lower = user_message.lower()
    if 'line chart' in message_lower or 'line graph' in message_lower:
        return 'line'
    elif 'bar chart' in message_lower or 'bar graph' in message_lower:
        return 'bar'
    elif 'pie chart' in message_lower or 'pie graph' in message_lower:
        return 'pie'
    else:
        return 'auto'

# UI: tool refresh + clear chat
col1, col2 = st.columns(2)
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
        st.session_state.pending_chart = False

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

# Load system prompt from system.txt
def load_system_prompt():
    try:
        with open("system.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    except FileNotFoundError:
        base_prompt = """You are a helpful assistant connected to an MCP server. Use available tools to answer user queries.

CRITICAL QUERY CONSTRUCTION RULES:

For execute_load_query tool, you MUST follow this exact structure:

{
  "query": {
    "dimensions": [],
    "measures": ["table.measure_name"],
    "filters": [],
    "limit": 100
  }
}

MANDATORY REQUIREMENTS:
1. Always include at least one measure - Never leave measures array empty
2. Use exact field names from schema - "customer.total_customers", not "total_customers"
3. Dimensions can be empty for total aggregations
4. Available measures: customer.total_customers, proposal.total_proposal, sales.total_sales, sales.total_revenue

For "total customer count", use:
{
  "query": {
    "dimensions": [],
    "measures": ["customer.total_customers"], 
    "filters": [],
    "limit": 1
  }
}"""
    except Exception as e:
        base_prompt = f"Error loading system prompt: {e}"
    
    # Add visualization instructions
    visualization_prompt = """

VISUALIZATION INSTRUCTIONS:
- When you return tabular data (tables with rows and columns), always end your response by asking: "Would you like me to create a chart or visualization for this data? You can specify: bar chart, line chart, or pie chart."
- If the user asks for a chart/visualization, respond with: "I'll create that visualization for you based on the data from our previous query."
- Do not attempt to create visualizations yourself - the system will handle chart generation when the user requests it.
"""
    
    return base_prompt + visualization_prompt

# --- OpenAI agent turn ---
def agent_turn(user_text: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Check if user is requesting a chart
    if detect_chart_request(user_text):
        st.session_state.pending_chart = True
        chart_type = extract_chart_type(user_text)
        
        # If we have previous data, create chart immediately
        if st.session_state.last_data is not None:
            create_chart(st.session_state.last_data, chart_type)
            return f"I've created a {chart_type} chart based on the previous data. You can view it above!"
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
            # Special handling for execute_load_query to provide better feedback
            if tool_name == "execute_load_query":
                # Validate and fix the query structure before calling
                validated_args = validate_load_query_args(args)
                if "error" in validated_args:
                    tool_result_msgs.append({
                        "role": "tool", 
                        "tool_call_id": tc.id,
                        "content": f"[tool_error] {validated_args['error']}"
                    })
                    continue
                
                # Use the validated arguments
                args = validated_args
                st.write(f"Debug - Validated query: {args}")

            blocks = call_tool_sync(mcp_url, tool_name, args)
            text_result = mcp_blocks_to_text(blocks)
            tool_result_msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": text_result[:120000],  # keep it reasonable
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
    
    # Store data for potential visualization
    df = parse_table_from_text(final_response)
    if df is not None:
        st.session_state.last_data = df

    return final_response

# --- Input box ---
user_msg = st.chat_input("Ask anything. I'll pick a tool and run it.")
if user_msg:
    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        try:
            with st.spinner("Thinking, choosing a tool, and running it..."):
                time.sleep(0.25)
                reply = agent_turn(user_msg)
            placeholder.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            placeholder.error(f"Error: {e}")