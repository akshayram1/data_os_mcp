## Available Tools

The MCP server exposes several tools for interacting with DataOS:

### 1. `configure_dataos`
Configure DataOS credentials for the client. Call this once to set up the API URL and secret key.

**Arguments:**
- `lens_api_url`: DataOS Lens2 API URL (e.g., https://your-dataos.com/lens2/api/public:product-name/v2)
- `lens_api_secret`: DataOS API secret key

### 2. `get_metadata`
Fetch metadata from DataOS Lens2 `/meta` endpoint using configured credentials.

**Arguments:**
- `include_schema` (bool): Include full schema information
- `include_dimensions` (bool): Include extracted dimensions list

### 3. `get_connection_status`
Check if DataOS credentials are configured and working.

### 4. `execute_graphql`
Execute a GraphQL query against the configured Lens `/graphql` endpoint.

**Arguments:**
- `query`: GraphQL query string
- `variables`: Optional variables dict

### 5. `list_tools`
List available MCP tools and short descriptions.

---
# data_os_mcp

## How to Run the Application with `uv`

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency management and running the server.

### 1. Install `uv`

If you don't have `uv` installed, run:

```powershell
pip install uv
```

### 2. Sync Dependencies

Install all required dependencies as specified in `pyproject.toml`:

```powershell
uv sync
```

This will create a virtual environment and install all packages.

### 3. Run the Application

Start the server using:

```powershell
uv run python server.py
```

This will launch the MCP server on the default port (8000). You can change the port by setting the `PORT` environment variable.

### 4. (Optional) Activate the Virtual Environment

If you want to activate the environment manually:

**On Linux/macOS:**
```bash
source .venv/bin/activate
```

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```bash
python server.py
```
