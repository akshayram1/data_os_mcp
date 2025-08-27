import os
import json
import inspect
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
from data_products.lens_utils import LensUtils


# Initialize MCP server with HTTP transport, host/port will be set later
PORT = int(os.environ.get("PORT", 8000))
mcp = FastMCP("dataos-lens", host="0.0.0.0", port=PORT)

# In-memory storage for credentials (per server instance)
_session_config = {}

def _get_configured_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get stored credentials."""
    lens_url = _session_config.get("lens_api_url")
    secret = _session_config.get("lens_api_secret")
    return lens_url, secret

@mcp.tool()
def configure_dataos(
    lens_api_url: str,
    lens_api_secret: str
) -> Dict[str, Any]:
    """
    Configure DataOS credentials. Call this once when setting up the client.
    Args:
        lens_api_url: DataOS Lens2 API URL (e.g., https://your-dataos.com/lens2/api/public:product-name/v2)
        lens_api_secret: DataOS API secret key
    Returns:
        Configuration status and connection test result
    """
    try:
        # Test connection first
        import requests
        meta_url = f"{lens_api_url.rstrip('/')}/meta"
        headers = {"apikey": lens_api_secret}
        response = requests.get(meta_url, headers=headers, timeout=10)

        # Debugging output for troubleshooting
        print("Requesting URL:", meta_url)
        print("Using headers:", headers)
        print("Response code:", response.status_code)
        print("Response body:", response.text[:250])

        # If 401, try with Authorization Bearer header
        if response.status_code == 401:
            headers = {"Authorization": f"Bearer {lens_api_secret}"}
            response = requests.get(meta_url, headers=headers, timeout=10)
            print("Retry with Bearer header. Response code:", response.status_code)
            print("Response body:", response.text[:250])

        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Connection test failed: HTTP {response.status_code} {response.text[:200]}"
            }

        # Store credentials in memory and prefer Bearer auth for queries
        _session_config["lens_api_url"] = lens_api_url.rstrip("/")
        _session_config["lens_api_secret"] = lens_api_secret
        # Use Authorization: Bearer <secret> for /load POSTs per project convention
        _session_config["auth_header_type"] = "bearer"

        return {
            "success": True,
            "message": "DataOS credentials configured successfully",
            "lens_url": lens_api_url
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Configuration failed: {str(e)}"
        }

@mcp.tool()
def get_metadata(
    include_schema: bool = True,
    include_dimensions: bool = False
) -> Dict[str, Any]:
    """
    Fetch metadata from DataOS Lens2 /meta endpoint using configured credentials.
    Args:
        include_schema: Include full schema information
        include_dimensions: Include extracted dimensions list
    Returns:
        Schema metadata and optionally dimensions
    """
    lens_url, secret = _get_configured_credentials()
    auth_type = _session_config.get("auth_header_type", "apikey")
    if not lens_url or not secret:
        return {
            "error": "DataOS not configured. Please call configure_dataos() first with your credentials."
        }
    try:
        import requests
        meta_url = f"{lens_url}/meta"
        if auth_type == "apikey":
            headers = {"apikey": secret}
        else:
            headers = {"Authorization": f"Bearer {secret}"}
        response = requests.get(meta_url, headers=headers, timeout=30)
        print("Requesting URL:", meta_url)
        print("Using headers:", headers)
        print("Response code:", response.status_code)
        print("Response body:", response.text[:250])
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        data = response.json()
        result = {"success": True}
        if include_schema:
            result["schema"] = data
        if include_dimensions:
            dimensions = []
            if "tables" in data:
                for table in data["tables"]:
                    if "dimensions" in table:
                        for dim in table["dimensions"]:
                            dimensions.append(dim.get("name", ""))
            result["dimensions"] = dimensions
        return result
    except Exception as e:
        return {"error": str(e)}

# NOTE: execute_query tool removed — use `execute_graphql` for all queries (GraphQL endpoint).

@mcp.tool()
def get_connection_status() -> Dict[str, Any]:
    """Check if DataOS credentials are configured and working."""
    lens_url, secret = _get_configured_credentials()
    if not lens_url or not secret:
        return {
            "configured": False,
            "message": "No credentials configured"
        }
    return {
        "configured": True,
        "lens_url": lens_url,
        "message": "Credentials are configured"
    }


@mcp.tool()
def execute_graphql(
    query: str,
    variables: Optional[dict] = None
) -> Dict[str, Any]:
    """
    Execute a GraphQL query against the configured Lens `/graphql` endpoint.
    Args:
        query: GraphQL query string
        variables: Optional variables dict
    Returns:
        Parsed JSON response or error
    """
    lens_url, secret = _get_configured_credentials()
    if not lens_url or not secret:
        return {"error": "DataOS not configured. Please call configure_dataos() first with your credentials."}
    try:
        lens_utils = LensUtils(lensurl=lens_url, secret=secret)
        result = lens_utils.execute_graphql(query, variables, auth_type="bearer")
        return {"success": True, "data": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def list_tools() -> Dict[str, Any]:
    """
    List available MCP tools and short descriptions.
    Returns:
        A dict with a `tools` key containing a list of tool metadata.
    """


    tools: List[Dict[str, str]] = []

    # Prefer using the MCP instance's own listing method if provided
    try:
        if hasattr(mcp, "list_tools") and callable(getattr(mcp, "list_tools")):
            try:
                # call the method; it may return a coroutine
                candidate = mcp.list_tools()
                import asyncio
                if inspect.iscoroutine(candidate):
                    candidate = asyncio.run(candidate)

                res = candidate
                # Normalize common return shapes
                if isinstance(res, dict) and "tools" in res:
                    return res
                if isinstance(res, (list, tuple)):
                    import re
                    normalized = []
                    for item in res:
                        name = None
                        desc = ""
                        if isinstance(item, dict):
                            name = item.get("name") or item.get("id") or item.get("tool")
                            desc = item.get("description", "")
                        else:
                            # try common attributes
                            for attr in ("name", "id", "tool", "title"):
                                name = getattr(item, attr, None)
                                if name:
                                    break
                            # description candidates
                            desc = getattr(item, "description", None) or getattr(item, "doc", None) or getattr(item, "__doc__", None) or ""
                            # fallback: parse repr like "name='foo'"
                            if not name:
                                s = repr(item)
                                m = re.search(r"name=['\"]([^'\"]+)['\"]", s)
                                if m:
                                    name = m.group(1)
                        if not name:
                            name = str(item)
                        normalized.append({"name": str(name), "description": str(desc) if desc else ""})
                    return {"tools": normalized}
            except Exception:
                # If the MCP's method fails, continue to fallback methods
                pass

    except Exception:
        # Defensive: ignore and fall back
        pass

    # Try common registry attributes used by MCP implementations
    registry = getattr(mcp, "tools", None) or getattr(mcp, "_tools", None) or getattr(mcp, "registry", None)
    if registry:
        # If registry is dict-like
        if isinstance(registry, dict):
            for name, meta in registry.items():
                desc = ""
                if callable(meta):
                    desc = (meta.__doc__ or "").splitlines()[0] if meta.__doc__ else ""
                elif isinstance(meta, dict):
                    desc = meta.get("description", "")
                else:
                    desc = getattr(meta, "description", "") or ""
                tools.append({"name": name, "description": desc})
        else:
            # Try to iterate registry entries (name, meta)
            try:
                for entry in registry:
                    try:
                        name, meta = entry
                    except Exception:
                        continue
                    desc = ""
                    if callable(meta):
                        desc = (meta.__doc__ or "").splitlines()[0] if meta.__doc__ else ""
                    elif isinstance(meta, dict):
                        desc = meta.get("description", "")
                    else:
                        desc = getattr(meta, "description", "") or ""
                    tools.append({"name": name, "description": desc})
            except Exception:
                # Fall through to global scan
                pass

    # Fallback: scan global functions decorated as MCP tools (common decorator attributes)
    if not tools:
        for nm, obj in globals().items():
            if inspect.isfunction(obj):
                if getattr(obj, "__mcp_tool__", False) or getattr(obj, "_mcp_tool", False) or getattr(obj, "_is_mcp_tool", False):
                    desc = (obj.__doc__ or "").splitlines()[0] if obj.__doc__ else ""
                    tools.append({"name": nm, "description": desc})

    return {"tools": tools}

# Run the server
if __name__ == "__main__":
    mcp.run(transport="streamable-http")