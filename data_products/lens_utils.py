from typing import Optional, Any
import requests
import json


class LensUtils:
    def __init__(self, lensurl: str, secret: str):
        # lensurl should already include the Lens api base (e.g. .../v2)
        self.lensurl = lensurl.rstrip('/')
        self.secret = secret

    def get_results(self, query: dict, lens_name: Optional[str] = None, auth_type: str = "apikey", format: str = "json") -> Any:
        """
        Execute a query against the Lens `/load` endpoint.

        Args:
            query: The query object (dict). Will be wrapped in a top-level `query` key in the POST body.
            lens_name: Optional lens name (unused for now, kept for compatibility).
            auth_type: Either 'apikey' or 'bearer' to control the header type.
            format: Desired output format (currently only 'json' uses response.json()).

        Returns:
            Parsed JSON response (if format=='json') or raw text.
        """
        url = f"{self.lensurl}/load"
        headers = {"Content-Type": "application/json"}
        if auth_type and auth_type.lower() == "bearer":
            headers["Authorization"] = f"Bearer {self.secret}"
        else:
            # default to apikey header if not specified
            headers["apikey"] = self.secret

        body = {"query": query}

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

        if resp.status_code >= 400:
            # return structured error for caller
            try:
                return {"error": True, "status_code": resp.status_code, "body": resp.text}
            except Exception:
                return {"error": True, "status_code": resp.status_code, "body": resp.text}

        if format == "json":
            try:
                return resp.json()
            except Exception:
                return {"error": "Invalid JSON response", "body": resp.text}
        else:
            return resp.text

    def fetch_data(self, endpoint: str) -> Any:
        # convenience helper to GET from a relative endpoint on the lens (e.g. '/meta')
        url = f"{self.lensurl.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {"apikey": self.secret}
        try:
            resp = requests.get(url, headers=headers, timeout=30)
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
        if resp.status_code != 200:
            return {"error": True, "status_code": resp.status_code, "body": resp.text}
        try:
            return resp.json()
        except Exception:
            return resp.text

    def validate_query(self, query: dict) -> bool:
        # Basic validation: must be a dict and contain either measures or dimensions or timeDimensions
        if not isinstance(query, dict):
            return False
        if any(k in query for k in ("measures", "dimensions", "timeDimensions")):
            return True
        return False

    def execute_graphql(self, gql: str, variables: Optional[dict] = None, auth_type: str = "bearer") -> Any:
        """
        Execute a GraphQL query against the Lens `/graphql` endpoint.

        Args:
            gql: GraphQL query string.
            variables: Optional variables dict.
            auth_type: 'bearer' or 'apikey' to select header type.

        Returns:
            Parsed JSON response or an error dict.
        """
        url = f"{self.lensurl}/graphql"
        headers = {"Content-Type": "application/json"}
        if auth_type and auth_type.lower() == "bearer":
            headers["Authorization"] = f"Bearer {self.secret}"
        else:
            headers["apikey"] = self.secret

        body = {"query": gql}
        if variables:
            body["variables"] = variables

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=60)
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

        if resp.status_code >= 400:
            try:
                return {"error": True, "status_code": resp.status_code, "body": resp.text}
            except Exception:
                return {"error": True, "status_code": resp.status_code, "body": resp.text}

        try:
            return resp.json()
        except Exception:
            return {"error": "Invalid JSON response", "body": resp.text}