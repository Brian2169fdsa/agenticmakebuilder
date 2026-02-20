"""
End-to-End Integration Tests.

Exercises the complete pipeline using TestClient.
Uses mocked DB but real app logic.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app import app
from db.session import get_db


@pytest.fixture
def mock_db():
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []
    session.execute.return_value.scalar.return_value = 0
    session.execute.return_value.fetchone.return_value = None
    session.query.return_value.filter.return_value.first.return_value = None
    return session


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    app.dependency_overrides.clear()


class TestE2EPipeline:
    def test_health_to_admin_flow(self, client):
        """Verify core health and admin endpoints work end-to-end."""
        # Step 1: Health check
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True

        # Step 2: System status
        r = client.get("/admin/system-status")
        assert r.status_code == 200
        assert r.json()["status"] == "operational"

        # Step 3: Pipeline dashboard
        r = client.get("/pipeline/dashboard")
        assert r.status_code == 200
        assert "stages" in r.json()

        # Step 4: Daily briefing
        r = client.post("/briefing/daily")
        assert r.status_code == 200
        assert "date" in r.json()

    @patch("tools.embedding_engine.embed_document")
    def test_learn_outcome_to_insights(self, mock_embed, client):
        """Record a learning outcome then query insights."""
        # Step 1: Record outcome
        r = client.post("/learn/outcome", json={
            "scenario_id": "e2e-scn-001",
            "project_id": "e2e-proj-001",
            "outcome": "success",
            "execution_count": 10,
            "avg_duration_ms": 300,
            "error_patterns": [],
        })
        assert r.status_code == 200
        assert r.json()["learned"] is True

        # Step 2: Query insights
        r = client.get("/learn/insights", params={"description": "webhook automation"})
        assert r.status_code == 200
        assert "recommendation" in r.json()
        assert "risk_level" in r.json()

        # Step 3: Get summary
        r = client.get("/learn/summary")
        assert r.status_code == 200
        assert "total_outcomes" in r.json()

    def test_api_key_lifecycle(self, client, mock_db):
        """Create, list, and revoke an API key."""
        # Mock the INSERT RETURNING
        row = MagicMock()
        row.id = "00000000-0000-0000-0000-000000000099"
        row.created_at = MagicMock()
        row.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        mock_db.execute.return_value.fetchone.return_value = row

        # Step 1: Create key
        r = client.post("/auth/keys", json={"name": "E2E Test Key", "tenant_id": "default"})
        assert r.status_code == 200
        data = r.json()
        assert data["raw_key"].startswith("mab_")
        key_id = data["key_id"]

        # Step 2: List keys
        mock_db.execute.return_value.fetchall.return_value = []
        r = client.get("/auth/keys")
        assert r.status_code == 200
        assert "keys" in r.json()

        # Step 3: Revoke key
        r = client.delete(f"/auth/keys/{key_id}")
        assert r.status_code == 200
        assert r.json()["revoked"] is True

    def test_event_bus_subscribe_and_list(self, client, mock_db):
        """Subscribe to events, list subscriptions, check event types."""
        # Step 1: Get event types
        r = client.get("/events/types")
        assert r.status_code == 200
        assert len(r.json()["event_types"]) >= 10

        # Step 2: Subscribe
        row = MagicMock()
        row.id = "00000000-0000-0000-0000-000000000099"
        mock_db.execute.return_value.fetchone.return_value = row

        r = client.post("/events/subscribe", json={
            "event_type": "project.deployed",
            "target_url": "https://example.com/test-hook",
        })
        assert r.status_code == 200
        assert "subscription_id" in r.json()

        # Step 3: List subscriptions
        mock_db.execute.return_value.fetchall.return_value = []
        r = client.get("/events/subscriptions")
        assert r.status_code == 200
        assert "subscriptions" in r.json()

    def test_job_queue_lifecycle(self, client, mock_db):
        """Enqueue, check status, cancel a job."""
        with patch("tools.job_queue.enqueue_job", return_value="e2e-job-001"):
            r = client.post("/jobs/enqueue", json={"job_type": "verify", "payload": {}})
            assert r.status_code == 200
            job_id = r.json()["job_id"]

        with patch("tools.job_queue.get_job", return_value={
            "id": job_id, "job_type": "verify", "status": "pending",
            "payload": {}, "result": None, "error_message": None,
            "tenant_id": "default", "project_id": None, "priority": 5,
            "created_at": "2026-01-01T00:00:00+00:00",
            "started_at": None, "completed_at": None,
            "retry_count": 0, "max_retries": 3,
        }):
            r = client.get(f"/jobs/{job_id}")
            assert r.status_code == 200
            assert r.json()["status"] == "pending"

        with patch("tools.job_queue.cancel_job", return_value=True):
            r = client.delete(f"/jobs/{job_id}/cancel")
            assert r.status_code == 200
            assert r.json()["cancelled"] is True

    def test_tenant_lifecycle(self, client, mock_db):
        """Create, get, update, and delete a tenant."""
        # Create
        r = client.post("/tenants", json={"id": "e2e-test", "name": "E2E Tenant"})
        assert r.status_code == 200
        assert r.json()["created"] is True

        # Get (mock return)
        tenant = MagicMock()
        tenant.id = "e2e-test"
        tenant.name = "E2E Tenant"
        tenant.plan = "standard"
        tenant.rate_limit_per_minute = 60
        tenant.max_projects = 50
        tenant.max_api_keys = 5
        tenant.features = ["plan", "verify"]
        tenant.active = True
        tenant.created_at = MagicMock()
        tenant.created_at.isoformat.return_value = "2026-01-01T00:00:00"

        call_count = [0]
        def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.fetchone.return_value = tenant
            else:
                result.scalar.return_value = 0
                result.fetchone.return_value = None
            return result
        mock_db.execute = MagicMock(side_effect=execute_side_effect)

        r = client.get("/tenants/e2e-test")
        assert r.status_code == 200
        assert r.json()["name"] == "E2E Tenant"

        # Reset mock
        mock_db.execute = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        mock_db.execute.return_value.scalar.return_value = 0
        mock_db.execute.return_value.fetchone.return_value = None

        # Update
        r = client.patch("/tenants/e2e-test", json={"plan": "professional"})
        assert r.status_code == 200
        assert r.json()["updated"] is True

        # Delete
        r = client.delete("/tenants/e2e-test")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_full_verify_flow(self, client):
        """Verify a blueprint and check confidence."""
        r = client.post("/verify", json={
            "blueprint": {"name": "E2E Test Scenario", "flow": []},
            "project_id": "e2e-proj-001",
        })
        assert r.status_code == 200
        data = r.json()
        assert "confidence_score" in data
        assert "passed" in data

    @patch("tools.job_queue.enqueue_job", return_value="async-e2e-001")
    def test_async_plan_flow(self, mock_enqueue, client):
        """POST /plan?async_mode=true returns a job to poll."""
        r = client.post("/plan?async_mode=true", json={
            "original_request": "Build webhook automation",
            "customer_name": "E2E Corp",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == "async-e2e-001"
        assert data["status"] == "pending"
        assert "/jobs/" in data["poll_url"]
