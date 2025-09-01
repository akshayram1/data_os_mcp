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

def validate_and_fix_load_query(query_dict):
    """
    Validate and fix common issues with load query structure.
    Supports the full DataOS Lens query structure including timeDimensions.
    """
    if not isinstance(query_dict, dict):
        return {"error": "Query must be a dictionary"}
    
    # Ensure required keys exist with proper defaults
    fixed_query = {
        "dimensions": query_dict.get("dimensions", []),
        "measures": query_dict.get("measures", []),
        "filters": query_dict.get("filters", []),
        "timeDimensions": query_dict.get("timeDimensions", []),  # CRITICAL: Add timeDimensions support
        "limit": query_dict.get("limit", 100)
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
    
    # Validate timeDimensions structure
    for i, time_dim in enumerate(fixed_query["timeDimensions"]):
        if not isinstance(time_dim, dict):
            return {"error": f"timeDimensions[{i}] must be a dictionary"}
        
        # Required fields for timeDimensions
        if "dimension" not in time_dim:
            return {"error": f"timeDimensions[{i}] must include 'dimension' field"}
        
        # Optional fields with defaults
        if "dateRange" not in time_dim:
            time_dim["dateRange"] = ["2023-01-01", "2023-12-31"]  # Default range
        
        if "granularity" not in time_dim:
            time_dim["granularity"] = "month"  # Default granularity
        
        # Validate dateRange is a list of 2 strings
        if not isinstance(time_dim["dateRange"], list) or len(time_dim["dateRange"]) != 2:
            return {"error": f"timeDimensions[{i}].dateRange must be a list with start and end dates"}
    
    # Ensure limit is an integer
    try:
        fixed_query["limit"] = int(fixed_query["limit"])
    except (ValueError, TypeError):
        fixed_query["limit"] = 100
    
    # Validation: Must have at least one measure
    if not fixed_query["measures"]:
        return {"error": "Query must include at least one measure"}
    
    # Add debug logging for timeDimensions
    if fixed_query["timeDimensions"]:
        print(f"DEBUG: Processing timeDimensions: {fixed_query['timeDimensions']}")
    
    return fixed_query

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
def execute_load_query(query: dict) -> Dict[str, Any]:
    """
    Execute a query against the configured Lens /load endpoint.
    Args:
        query: Query object with dimensions, measures, filters, and limit
              Example: {
                  "dimensions": [],
                  "measures": ["customer.total_customers"],
                  "filters": [],
                  "limit": 100
              }
    Returns:
        Parsed JSON response or error
    """
    lens_url, secret = _get_configured_credentials()
    if not lens_url or not secret:
        return {"error": "DataOS not configured. Please call configure_dataos() first with your credentials."}
    
    # Validate and fix query structure
    validated_query = validate_and_fix_load_query(query)
    if "error" in validated_query:
        return validated_query
    
    try:
        import requests
        import json
        
        # Convert validated query dict to JSON string
        query_json = json.dumps(validated_query)
        
        # Build the load URL with query parameter
        load_url = f"{lens_url}/load"
        
        # Get auth type from session config
        auth_type = _session_config.get("auth_header_type", "apikey")
        if auth_type == "bearer":
            headers = {"Authorization": f"Bearer {secret}"}
        else:
            headers = {"apikey": secret}
        
        # Add content type for JSON
        headers["Content-Type"] = "application/json"
        
        # Make the request with query as URL parameter
        params = {"query": query_json}
        response = requests.get(load_url, headers=headers, params=params, timeout=120)
        
        print("Requesting URL:", load_url)
        print("Validated Query:", validated_query)
        print("Query params:", params)
        print("Using headers:", {k: v if k != "Authorization" else "***" for k, v in headers.items()})
        print("Response code:", response.status_code)
        print("Response body:", response.text[:500])
        
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}: {response.text[:200]}"}
        
        data = response.json()
        return {"success": True, "data": data, "query_used": validated_query}
        
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