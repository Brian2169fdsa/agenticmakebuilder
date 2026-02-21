"""
Make.com REST Client — Pure httpx, no MCP dependency.

Connects agenticmakebuilder to the Make.com API for creating, activating,
monitoring, and managing scenarios programmatically.

Reads MAKECOM_API_KEY, MAKECOM_TEAM_ID, MAKECOM_ORG_ID from environment.
Raises MakecomError if credentials not configured or API calls fail.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import httpx


class MakecomError(Exception):
    """Raised when a Make.com API call fails or credentials are missing."""

    def __init__(self, message: str, status_code: int = 0, body: str = ""):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class MakecomClient:
    """Lightweight Make.com REST client using httpx."""

    def __init__(self, api_key: str, team_id: int, org_id: int, base_url: str = ""):
        self.api_key = api_key
        self.team_id = team_id
        self.org_id = org_id
        self.base_url = (base_url or "https://us2.make.com/api/v2").rstrip("/")
        self.headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json_body: Optional[dict] = None,
        timeout: int = 30,
    ) -> dict | list:
        """Execute an HTTP request against the Make.com API."""
        url = f"{self.base_url}{path}"
        resp = httpx.request(
            method,
            url,
            headers=self.headers,
            params=params,
            json=json_body,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            raise MakecomError(
                f"Make.com API {method} {path} failed: {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text,
            )
        if resp.status_code == 204 or not resp.text.strip():
            return {}
        return resp.json()

    # ── Scenarios ─────────────────────────────────────────────

    def create_scenario(
        self,
        name: str,
        blueprint: dict,
        scheduling: Optional[dict] = None,
        folder_id: Optional[int] = None,
    ) -> dict:
        """Create a new scenario in Make.com.

        Args:
            name: Scenario name.
            blueprint: Make.com blueprint dict (will be JSON-stringified).
            scheduling: Scheduling config (default: indefinitely, 900s interval).
            folder_id: Optional folder ID to place the scenario in.

        Returns:
            Full scenario object with id, name, etc.
        """
        body = {
            "teamId": self.team_id,
            "blueprint": json.dumps(blueprint),
            "scheduling": scheduling or {"type": "indefinitely", "interval": 900},
            "confirmed": True,
        }
        if folder_id:
            body["folderId"] = folder_id

        result = self._request("POST", "/scenarios", json_body=body)
        scenario = result.get("scenario", result)
        # Inject the name via update since create doesn't always accept name
        if name and scenario.get("id"):
            try:
                self.update_scenario(scenario["id"], name=name)
                scenario["name"] = name
            except MakecomError:
                pass
        return scenario

    def get_scenario(self, scenario_id: int) -> dict:
        """Get a scenario by ID."""
        result = self._request("GET", f"/scenarios/{scenario_id}")
        return result.get("scenario", result)

    def update_scenario(
        self,
        scenario_id: int,
        blueprint: Optional[dict] = None,
        scheduling: Optional[dict] = None,
        name: Optional[str] = None,
    ) -> dict:
        """Update a scenario's blueprint, scheduling, or name."""
        body = {}
        if blueprint is not None:
            body["blueprint"] = json.dumps(blueprint)
        if scheduling is not None:
            body["scheduling"] = scheduling
        if name is not None:
            body["name"] = name
        result = self._request("PATCH", f"/scenarios/{scenario_id}", json_body=body)
        return result.get("scenario", result)

    def activate_scenario(self, scenario_id: int) -> dict:
        """Activate (start) a scenario."""
        result = self._request("POST", f"/scenarios/{scenario_id}/start")
        return result.get("scenario", result)

    def deactivate_scenario(self, scenario_id: int) -> dict:
        """Deactivate (stop) a scenario."""
        result = self._request("POST", f"/scenarios/{scenario_id}/stop")
        return result.get("scenario", result)

    def delete_scenario(self, scenario_id: int) -> bool:
        """Delete a scenario. Returns True on success."""
        self._request("DELETE", f"/scenarios/{scenario_id}")
        return True

    def list_scenarios(
        self, team_id: Optional[int] = None, folder_id: Optional[int] = None, limit: int = 20
    ) -> list:
        """List scenarios for a team."""
        params = {"teamId": team_id or self.team_id, "pg[limit]": limit}
        if folder_id:
            params["folderId"] = folder_id
        result = self._request("GET", "/scenarios", params=params)
        return result.get("scenarios", result) if isinstance(result, dict) else result

    def run_scenario(self, scenario_id: int, data: Optional[dict] = None) -> dict:
        """Trigger a single execution of a scenario."""
        body = {"scenarioId": scenario_id, "responsive": True}
        if data:
            body["data"] = data
        return self._request("POST", f"/scenarios/{scenario_id}/run", json_body=body)

    # ── Executions ────────────────────────────────────────────

    def get_executions(self, scenario_id: int, limit: int = 10) -> list:
        """Get recent executions for a scenario."""
        params = {"pg[limit]": limit}
        result = self._request("GET", f"/scenarios/{scenario_id}/logs", params=params)
        return result.get("scenarioLogs", result) if isinstance(result, dict) else result

    def get_execution_detail(self, scenario_id: int, execution_id: str) -> dict:
        """Get detailed result of a specific execution."""
        result = self._request(
            "GET", f"/scenarios/{scenario_id}/logs/{execution_id}"
        )
        return result.get("scenarioLog", result) if isinstance(result, dict) else result

    # ── Connections ────────────────────────────────────────────

    def list_connections(self, team_id: Optional[int] = None) -> list:
        """List connections for a team."""
        params = {"teamId": team_id or self.team_id}
        result = self._request("GET", "/connections", params=params)
        return result.get("connections", result) if isinstance(result, dict) else result

    # ── Webhooks ──────────────────────────────────────────────

    def create_webhook(self, name: str, team_id: Optional[int] = None) -> dict:
        """Create a custom webhook."""
        body = {
            "teamId": team_id or self.team_id,
            "name": name,
            "typeName": "gateway-CustomWebHook",
        }
        result = self._request("POST", "/hooks", json_body=body)
        return result.get("hook", result)

    def get_webhook(self, hook_id: int) -> dict:
        """Get a webhook by ID."""
        result = self._request("GET", f"/hooks/{hook_id}")
        return result.get("hook", result)

    # ── Folders ───────────────────────────────────────────────

    def list_folders(self, team_id: Optional[int] = None) -> list:
        """List folders for a team."""
        params = {"teamId": team_id or self.team_id}
        result = self._request("GET", "/folders", params=params)
        return result.get("folders", result) if isinstance(result, dict) else result

    def get_team_id(self) -> int:
        """Return configured team ID."""
        return self.team_id


# ── Singleton ─────────────────────────────────────────────────

_client: Optional[MakecomClient] = None


def get_makecom_client() -> MakecomClient:
    """Return shared MakecomClient instance.

    Raises MakecomError if MAKECOM_API_KEY is not configured.
    """
    global _client
    if _client is not None:
        return _client

    api_key = os.environ.get("MAKECOM_API_KEY", "").strip()
    if not api_key:
        raise MakecomError("MAKECOM_API_KEY not configured")

    team_id = int(os.environ.get("MAKECOM_TEAM_ID", "0"))
    org_id = int(os.environ.get("MAKECOM_ORG_ID", "0"))
    base_url = os.environ.get("MAKECOM_API_BASE", "").strip()

    _client = MakecomClient(
        api_key=api_key,
        team_id=team_id,
        org_id=org_id,
        base_url=base_url,
    )
    return _client
