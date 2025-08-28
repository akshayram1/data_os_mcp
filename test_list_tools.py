import requests

# Change this URL if your server runs on a different port
SERVER_URL = "http://localhost:8000"

# Endpoint for the list_tools tool
LIST_TOOLS_ENDPOINT = f"{SERVER_URL}/tools/list_tools"

if __name__ == "__main__":
    try:
        response = requests.post(LIST_TOOLS_ENDPOINT, json={})
        if response.status_code == 200:
            print("Available tools:")
            print(response.json())
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")
