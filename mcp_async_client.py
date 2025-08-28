import asyncio
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client  # correct for mcp 1.13.x


class AsyncMCPClient:
    def __init__(self, url: str):
        # Accept either ...:8000 or ...:8000/mcp; normalize to include /mcp
        self.url = url if url.rstrip("/").endswith("/mcp") else url.rstrip("/") + "/mcp"

    # NOTE: not async — returns the async context manager itself
    def _session(self):
        return streamablehttp_client(self.url)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return tools with name, description, and input schema (if provided)."""
        async with self._session() as ctx:
            # SDK 1.13.x returns a tuple (read_stream, write_stream, [optional metadata])
            if isinstance(ctx, (list, tuple)):
                if len(ctx) >= 2:
                    read_stream, write_stream = ctx[0], ctx[1]
                else:
                    raise RuntimeError("Unexpected streamablehttp_client context shape")
            else:
                read_stream, write_stream = getattr(ctx, "read", None), getattr(ctx, "write", None)

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_resp = await session.list_tools()
                tools: List[Dict[str, Any]] = []
                for t in tools_resp.tools:
                    tools.append(
                        {
                            "name": t.name,
                            "description": getattr(t, "description", None),
                            # Anthropic expects 'input_schema' (snake_case)
                            "input_schema": getattr(t, "inputSchema", None) or getattr(t, "input_schema", None),
                        }
                    )
                return tools

    async def call_tool(self, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Call a tool and return the server's content payload (MCP content blocks)."""
        params = params or {}
        async with self._session() as ctx:
            if isinstance(ctx, (list, tuple)):
                if len(ctx) >= 2:
                    read_stream, write_stream = ctx[0], ctx[1]
                else:
                    raise RuntimeError("Unexpected streamablehttp_client context shape")
            else:
                read_stream, write_stream = getattr(ctx, "read", None), getattr(ctx, "write", None)

            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, params)
                return result.content


# Convenience sync wrappers for Streamlit
def list_tools_sync(url: str) -> List[Dict[str, Any]]:
    return asyncio.run(AsyncMCPClient(url).list_tools())


def call_tool_sync(url: str, tool_name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return asyncio.run(AsyncMCPClient(url).call_tool(tool_name, params))
