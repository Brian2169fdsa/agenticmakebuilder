"""
Tests for Supabase integration layer.

All tests mock httpx to avoid real Supabase calls.
"""

import os
from unittest.mock import MagicMock, patch

import pytest


# ── SupabaseClient tests ─────────────────────────────────────

class TestSupabaseClient:
    def test_client_returns_none_without_env(self):
        """get_supabase_client returns None when SUPABASE_URL not set."""
        import tools.supabase_client as mod
        mod._client = None  # Reset singleton
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SUPABASE_URL", None)
            os.environ.pop("SUPABASE_SERVICE_KEY", None)
            result = mod.get_supabase_client()
            assert result is None

    def test_client_created_with_env(self):
        """get_supabase_client returns SupabaseClient when env vars set."""
        import tools.supabase_client as mod
        mod._client = None  # Reset singleton
        with patch.dict(os.environ, {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_SERVICE_KEY": "test-key-123",
        }):
            client = mod.get_supabase_client()
            assert client is not None
            assert client.base_url == "https://test.supabase.co"
            assert client.headers["apikey"] == "test-key-123"
        mod._client = None  # Reset for other tests

    def test_select_success(self):
        """select() returns parsed JSON on 200."""
        from tools.supabase_client import SupabaseClient

        client = SupabaseClient("https://test.supabase.co", "key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "1", "current_stage": "build"}]

        with patch("httpx.get", return_value=mock_resp) as mock_get:
            result = client.select("project_agent_state", filters={"project_id": "abc"})
            assert result == [{"id": "1", "current_stage": "build"}]
            mock_get.assert_called_once()

    def test_select_raises_on_error(self):
        """select() raises SupabaseError on 4xx/5xx."""
        from tools.supabase_client import SupabaseClient, SupabaseError

        client = SupabaseClient("https://test.supabase.co", "key")
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not found"

        with patch("httpx.get", return_value=mock_resp):
            with pytest.raises(SupabaseError) as exc:
                client.select("nonexistent_table")
            assert exc.value.status_code == 404

    def test_insert_success(self):
        """insert() returns inserted rows."""
        from tools.supabase_client import SupabaseClient

        client = SupabaseClient("https://test.supabase.co", "key")
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = [{"id": "new-1"}]

        with patch("httpx.post", return_value=mock_resp):
            result = client.insert("project_activity", {"action_type": "test"})
            assert result == [{"id": "new-1"}]

    def test_upsert_success(self):
        """upsert() returns upserted rows with merge-duplicates header."""
        from tools.supabase_client import SupabaseClient

        client = SupabaseClient("https://test.supabase.co", "key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"project_id": "abc", "current_stage": "verify"}]

        with patch("httpx.post", return_value=mock_resp) as mock_post:
            result = client.upsert(
                "project_agent_state",
                {"project_id": "abc", "current_stage": "verify"},
                on_conflict="project_id",
            )
            assert result[0]["current_stage"] == "verify"
            call_headers = mock_post.call_args.kwargs.get("headers", {})
            assert "merge-duplicates" in call_headers.get("Prefer", "")

    def test_update_success(self):
        """update() returns updated rows."""
        from tools.supabase_client import SupabaseClient

        client = SupabaseClient("https://test.supabase.co", "key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"id": "1", "status": "deployed"}]

        with patch("httpx.patch", return_value=mock_resp):
            result = client.update("deployments", {"id": "1"}, {"status": "deployed"})
            assert result[0]["status"] == "deployed"


# ── pipeline_sync tests ──────────────────────────────────────

class TestPipelineSync:
    def test_sync_project_state_success(self):
        """sync_project_state upserts to Supabase and returns row."""
        from tools.pipeline_sync import sync_project_state
        mock_client = MagicMock()
        mock_client.upsert.return_value = [{"project_id": "abc", "current_stage": "build"}]

        with patch("tools.pipeline_sync.get_supabase_client", return_value=mock_client):
            result = sync_project_state("abc", "build", "builder")
            assert result["current_stage"] == "build"
            mock_client.upsert.assert_called_once()

    def test_sync_project_state_no_client(self):
        """sync_project_state returns None when Supabase not configured."""
        from tools.pipeline_sync import sync_project_state

        with patch("tools.pipeline_sync.get_supabase_client", return_value=None):
            result = sync_project_state("abc", "build", "builder")
            assert result is None

    def test_sync_build_verification_success(self):
        """sync_build_verification inserts verification result."""
        from tools.pipeline_sync import sync_build_verification
        mock_client = MagicMock()
        mock_client.insert.return_value = [{"project_id": "abc", "passed": True}]

        with patch("tools.pipeline_sync.get_supabase_client", return_value=mock_client):
            result = sync_build_verification("abc", 92.5, True)
            assert result["passed"] is True
            mock_client.insert.assert_called_once()

    def test_sync_project_financials_calculates_margin(self):
        """sync_project_financials calculates margin correctly."""
        from tools.pipeline_sync import sync_project_financials
        mock_client = MagicMock()
        mock_client.upsert.return_value = [{"project_id": "abc"}]

        with patch("tools.pipeline_sync.get_supabase_client", return_value=mock_client):
            result = sync_project_financials("abc", revenue=1000.0, api_cost=200.0)
            assert result is not None
            call_data = mock_client.upsert.call_args[0][1]
            assert call_data["margin"] == 800.0
            assert call_data["margin_pct"] == 80.0

    def test_sync_activity_success(self):
        """sync_activity inserts to project_activity."""
        from tools.pipeline_sync import sync_activity
        mock_client = MagicMock()
        mock_client.insert.return_value = [{"id": "new"}]

        with patch("tools.pipeline_sync.get_supabase_client", return_value=mock_client):
            result = sync_activity("abc", "build_started", "Build plan generated")
            assert result is not None
            mock_client.insert.assert_called_once()

    def test_get_project_stage_returns_stage(self):
        """get_project_stage queries and returns current stage."""
        from tools.pipeline_sync import get_project_stage
        mock_client = MagicMock()
        mock_client.select.return_value = [{"current_stage": "verify"}]

        with patch("tools.pipeline_sync.get_supabase_client", return_value=mock_client):
            result = get_project_stage("abc")
            assert result == "verify"

    def test_get_project_stage_not_found(self):
        """get_project_stage returns None when no rows found."""
        from tools.pipeline_sync import get_project_stage
        mock_client = MagicMock()
        mock_client.select.return_value = []

        with patch("tools.pipeline_sync.get_supabase_client", return_value=mock_client):
            result = get_project_stage("abc")
            assert result is None

    def test_sync_handles_exception_gracefully(self):
        """sync functions return None on exception, never raise."""
        from tools.pipeline_sync import sync_project_state
        mock_client = MagicMock()
        mock_client.upsert.side_effect = Exception("Connection refused")

        with patch("tools.pipeline_sync.get_supabase_client", return_value=mock_client):
            result = sync_project_state("abc", "build", "builder")
            assert result is None


# ── notification_sender tests ────────────────────────────────

class TestNotificationSender:
    def test_send_notification_success(self):
        """send_notification inserts to department_notifications."""
        from tools.notification_sender import send_notification
        mock_client = MagicMock()
        mock_client.insert.return_value = [{"id": "notif-1"}]

        with patch("tools.notification_sender.get_supabase_client", return_value=mock_client):
            result = send_notification(
                department="operations",
                type="warning",
                title="Test",
                message="Test message",
                reference_type="project",
            )
            assert result is not None
            mock_client.insert.assert_called_once()

    def test_send_stall_alert(self):
        """send_stall_alert formats message correctly."""
        from tools.notification_sender import send_stall_alert
        mock_client = MagicMock()
        mock_client.insert.return_value = [{"id": "notif-2"}]

        with patch("tools.notification_sender.get_supabase_client", return_value=mock_client):
            result = send_stall_alert("proj-123", days_stalled=3.5)
            assert result is not None
            call_data = mock_client.insert.call_args[0][1]
            assert "3.5 days" in call_data[0]["title"]
            assert call_data[0]["department"] == "operations"

    def test_send_cost_alert_negative_margin(self):
        """send_cost_alert sends alert-level for negative margin."""
        from tools.notification_sender import send_cost_alert
        mock_client = MagicMock()
        mock_client.insert.return_value = [{"id": "notif-3"}]

        with patch("tools.notification_sender.get_supabase_client", return_value=mock_client):
            result = send_cost_alert("proj-123", margin_pct=-5.0)
            assert result is not None
            call_data = mock_client.insert.call_args[0][1]
            assert call_data[0]["type"] == "alert"
            assert "Negative margin" in call_data[0]["title"]

    def test_send_cost_alert_low_margin(self):
        """send_cost_alert sends warning-level for low positive margin."""
        from tools.notification_sender import send_cost_alert
        mock_client = MagicMock()
        mock_client.insert.return_value = [{"id": "notif-4"}]

        with patch("tools.notification_sender.get_supabase_client", return_value=mock_client):
            result = send_cost_alert("proj-123", margin_pct=12.0)
            assert result is not None
            call_data = mock_client.insert.call_args[0][1]
            assert call_data[0]["type"] == "warning"

    def test_notification_no_client(self):
        """send_notification returns None when Supabase not configured."""
        from tools.notification_sender import send_notification

        with patch("tools.notification_sender.get_supabase_client", return_value=None):
            result = send_notification("ops", "info", "T", "M", "project")
            assert result is None
