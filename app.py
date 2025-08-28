import os
import json
import time
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

# --- OpenAI agent turn ---
def agent_turn(user_text: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in .env")

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build chat history for OpenAI
    msgs = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    msgs.append({"role": "user", "content": user_text})

    system_prompt = (
        "You are a tools-first assistant connected to an MCP server.\n"
        "- If a tool can answer, call exactly one function with a STRICT JSON argument body that matches its schema.\n"
        "- If you need clarification, ask a brief follow-up.\n"
        "- If no tool applies, answer concisely.\n"
        "Be accurate and terse."
    )

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

    # Execute all tool calls (we encourage only one, but handle N just in case)
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
                "content": f"[tool_error] Invalid tool arguments: {e}"
            })
            continue

        try:
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

    return followup.choices[0].message.content or "(no response)"

# --- Input box ---
user_msg = st.chat_input("Ask anything. I’ll pick a tool and run it.")
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
