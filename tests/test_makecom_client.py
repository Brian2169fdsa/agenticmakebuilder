"""
Tests for Make.com REST client.

All tests mock httpx to avoid real Make.com API calls.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestMakecomClient:
    def test_client_raises_without_api_key(self):
        """get_makecom_client raises MakecomError when MAKECOM_API_KEY not set."""
        import tools.makecom_client as mod
        mod._client = None
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MAKECOM_API_KEY", None)
            from tools.makecom_client import MakecomError
            with pytest.raises(MakecomError, match="not configured"):
                mod.get_makecom_client()

    def test_client_created_with_env(self):
        """get_makecom_client returns MakecomClient when env vars set."""
        import tools.makecom_client as mod
        mod._client = None
        with patch.dict(os.environ, {
            "MAKECOM_API_KEY": "test-key-123",
            "MAKECOM_TEAM_ID": "999",
            "MAKECOM_ORG_ID": "888",
        }):
            client = mod.get_makecom_client()
            assert client is not None
            assert client.api_key == "test-key-123"
            assert client.team_id == 999
            assert client.org_id == 888
        mod._client = None

    def test_create_scenario_success(self):
        """create_scenario sends POST and returns scenario dict."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"scenario": {"id": 42, "name": "Test"}}'
        mock_resp.json.return_value = {"scenario": {"id": 42, "name": "Test"}}

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            result = client.create_scenario("Test", {"flow": []})
            assert result["id"] == 42
            assert mock_req.call_count >= 1

    def test_get_scenario_success(self):
        """get_scenario returns scenario dict."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"scenario": {"id": 42, "isActive": true}}'
        mock_resp.json.return_value = {"scenario": {"id": 42, "isActive": True}}

        with patch("httpx.request", return_value=mock_resp):
            result = client.get_scenario(42)
            assert result["id"] == 42
            assert result["isActive"] is True

    def test_activate_scenario(self):
        """activate_scenario sends POST to start endpoint."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"scenario": {"id": 42, "isActive": true}}'
        mock_resp.json.return_value = {"scenario": {"id": 42, "isActive": True}}

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            result = client.activate_scenario(42)
            assert result["isActive"] is True
            call_args = mock_req.call_args
            assert call_args[0][0] == "POST"
            assert "/scenarios/42/start" in call_args[0][1]

    def test_deactivate_scenario(self):
        """deactivate_scenario sends POST to stop endpoint."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"scenario": {"id": 42, "isActive": false}}'
        mock_resp.json.return_value = {"scenario": {"id": 42, "isActive": False}}

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            result = client.deactivate_scenario(42)
            assert result["isActive"] is False
            call_args = mock_req.call_args
            assert "/scenarios/42/stop" in call_args[0][1]

    def test_delete_scenario(self):
        """delete_scenario returns True on success."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 204
        mock_resp.text = ""

        with patch("httpx.request", return_value=mock_resp):
            assert client.delete_scenario(42) is True

    def test_list_scenarios(self):
        """list_scenarios returns list of scenario dicts."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"scenarios": [{"id": 1}, {"id": 2}]}'
        mock_resp.json.return_value = {"scenarios": [{"id": 1}, {"id": 2}]}

        with patch("httpx.request", return_value=mock_resp):
            result = client.list_scenarios()
            assert len(result) == 2

    def test_get_executions(self):
        """get_executions returns list of execution logs."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"scenarioLogs": [{"id": "ex1", "status": 1}]}'
        mock_resp.json.return_value = {"scenarioLogs": [{"id": "ex1", "status": 1}]}

        with patch("httpx.request", return_value=mock_resp):
            result = client.get_executions(42, limit=5)
            assert len(result) == 1
            assert result[0]["status"] == 1

    def test_error_raises_makecom_error(self):
        """MakecomError raised on non-2xx response."""
        from tools.makecom_client import MakecomClient, MakecomError

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch("httpx.request", return_value=mock_resp):
            with pytest.raises(MakecomError) as exc:
                client.get_scenario(42)
            assert exc.value.status_code == 403

    def test_create_webhook(self):
        """create_webhook sends POST to hooks endpoint."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"hook": {"id": 55, "url": "https://hook.test/abc"}}'
        mock_resp.json.return_value = {"hook": {"id": 55, "url": "https://hook.test/abc"}}

        with patch("httpx.request", return_value=mock_resp):
            result = client.create_webhook("test-hook")
            assert result["id"] == 55
            assert "hook.test" in result["url"]

    def test_run_scenario(self):
        """run_scenario sends POST to run endpoint."""
        from tools.makecom_client import MakecomClient

        client = MakecomClient("key", 100, 200)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = '{"executionId": "exec-123"}'
        mock_resp.json.return_value = {"executionId": "exec-123"}

        with patch("httpx.request", return_value=mock_resp) as mock_req:
            result = client.run_scenario(42, data={"key": "val"})
            assert result["executionId"] == "exec-123"
            call_args = mock_req.call_args
            assert "/scenarios/42/run" in call_args[0][1]

    def test_get_team_id(self):
        """get_team_id returns configured team_id."""
        from tools.makecom_client import MakecomClient
        client = MakecomClient("key", 999, 888)
        assert client.get_team_id() == 999
