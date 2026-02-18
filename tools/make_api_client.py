"""
Make.com API Client

Wrapper for the Make.com REST API.
Used by the integration tester and live monitor.

Configuration:
    MAKE_API_KEY    — Make.com API key (required)
    MAKE_TEAM_ID    — Team ID (required)
    MAKE_API_BASE   — API base URL (default: https://us1.make.com/api/v2)

Make.com API docs: https://developers.make.com/api-documentation
"""

import json
import os
import time
import urllib.request
import urllib.error
from typing import Optional


MAKE_API_BASE = os.environ.get("MAKE_API_BASE", "https://us1.make.com/api/v2")
MAX_POLL_SECONDS = 60
POLL_INTERVAL = 3


class MakeAPIError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Make.com API error {status_code}: {message}")


class MakeAPIClient:
    """
    Make.com REST API client.
    All methods raise MakeAPIError on non-2xx responses.
    """

    def __init__(self, api_key: str = None, team_id: int = None, base_url: str = None):
        self.api_key = api_key or os.environ.get("MAKE_API_KEY", "")
        self.team_id = team_id or int(os.environ.get("MAKE_TEAM_ID", "0"))
        self.base_url = (base_url or MAKE_API_BASE).rstrip("/")

    def _request(self, method: str, path: str, body: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode() if body else None
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode()[:500]
            raise MakeAPIError(e.code, body_text)
        except Exception as ex:
            raise MakeAPIError(0, str(ex))

    # ─── Scenarios ───

    def create_scenario(self, blueprint: dict, name: str = None) -> dict:
        """Create a new scenario from a Make.com blueprint JSON."""
        payload = {
            "teamId": self.team_id,
            "blueprint": json.dumps(blueprint),
            "scheduling": {"type": "indefinitely"},
        }
        if name:
            payload["name"] = name
        return self._request("POST", "/scenarios", payload)

    def get_scenario(self, scenario_id: int) -> dict:
        return self._request("GET", f"/scenarios/{scenario_id}")

    def delete_scenario(self, scenario_id: int) -> dict:
        return self._request("DELETE", f"/scenarios/{scenario_id}")

    def activate_scenario(self, scenario_id: int) -> dict:
        return self._request("POST", f"/scenarios/{scenario_id}/start")

    def deactivate_scenario(self, scenario_id: int) -> dict:
        return self._request("POST", f"/scenarios/{scenario_id}/stop")

    def list_scenarios(self, folder_id: int = None) -> dict:
        path = f"/scenarios?teamId={self.team_id}"
        if folder_id:
            path += f"&folderId={folder_id}"
        return self._request("GET", path)

    # ─── Executions ───

    def run_scenario(self, scenario_id: int, data: dict = None) -> dict:
        """Trigger a manual scenario run. Returns execution info."""
        payload = {}
        if data:
            payload["data"] = data
        return self._request("POST", f"/scenarios/{scenario_id}/run", payload)

    def get_execution(self, execution_id: str) -> dict:
        return self._request("GET", f"/executions/{execution_id}")

    def list_executions(self, scenario_id: int, limit: int = 20) -> dict:
        return self._request(
            "GET", f"/scenarios/{scenario_id}/executions?limit={limit}"
        )

    def wait_for_execution(
        self, execution_id: str, timeout: int = MAX_POLL_SECONDS
    ) -> dict:
        """
        Poll until execution completes or timeout.
        Returns final execution dict.
        """
        start = time.time()
        while time.time() - start < timeout:
            execution = self.get_execution(execution_id)
            status = execution.get("status", "")
            if status in ("success", "failed", "warning"):
                return execution
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"Execution {execution_id} did not complete in {timeout}s")

    # ─── Execution history for monitor ───

    def get_execution_history(
        self, scenario_id: int, limit: int = 50
    ) -> list:
        """
        Get recent execution history for a scenario.
        Returns list of execution summaries.
        """
        result = self.list_executions(scenario_id, limit=limit)
        return result.get("executions", result.get("data", []))

    def get_all_scenario_stats(self) -> list:
        """
        Get execution stats for all scenarios in the team.
        Used by live monitor for health checks.
        """
        scenarios = self.list_scenarios()
        scenario_list = scenarios.get("scenarios", scenarios.get("data", []))
        stats = []
        for s in scenario_list:
            sid = s.get("id")
            if not sid:
                continue
            try:
                history = self.get_execution_history(sid, limit=20)
                stats.append({
                    "scenario_id": sid,
                    "name": s.get("name", ""),
                    "is_active": s.get("isActive", False),
                    "recent_executions": history,
                })
            except MakeAPIError:
                pass
        return stats

    # ─── Connections ───

    def list_connections(self) -> dict:
        return self._request("GET", f"/connections?teamId={self.team_id}")

    def is_configured(self) -> bool:
        """Check if API key and team ID are set."""
        return bool(self.api_key) and self.team_id > 0


def get_client() -> MakeAPIClient:
    """Get a configured Make.com API client from environment."""
    return MakeAPIClient()


if __name__ == "__main__":
    print("=== Make API Client Self-Check ===\n")

    print("Test 1: Client instantiation from env")
    os.environ["MAKE_API_KEY"] = "test-key"
    os.environ["MAKE_TEAM_ID"] = "12345"
    client = get_client()
    assert client.api_key == "test-key"
    assert client.team_id == 12345
    print("  [OK]")

    print("Test 2: is_configured returns True when key + team set")
    assert client.is_configured()
    print("  [OK]")

    print("Test 3: is_configured returns False when missing")
    empty_client = MakeAPIClient(api_key="", team_id=0)
    assert not empty_client.is_configured()
    print("  [OK]")

    print("Test 4: MakeAPIError has correct attributes")
    err = MakeAPIError(404, "Not found")
    assert err.status_code == 404
    assert "404" in str(err)
    print("  [OK]")

    print("\n=== All Make API client checks passed ===")
    print("Note: Live API calls require MAKE_API_KEY and MAKE_TEAM_ID set to real values.")
