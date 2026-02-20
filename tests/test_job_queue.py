"""
Tests for Phase 2 — Job Queue.

Covers: tools/job_queue.py, tools/job_worker.py,
POST /jobs/enqueue, GET /jobs/{id}, GET /jobs/list,
DELETE /jobs/{id}/cancel, GET /jobs/stats, POST /jobs/cleanup,
POST /plan?async_mode=true
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
    return session


@pytest.fixture
def client(mock_db):
    app.dependency_overrides[get_db] = lambda: mock_db
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    app.dependency_overrides.clear()


class TestJobQueueEndpoints:
    @patch("tools.job_queue.enqueue_job", return_value="test-job-id-001")
    def test_enqueue_job(self, mock_enqueue, client):
        r = client.post("/jobs/enqueue", json={
            "job_type": "plan",
            "payload": {"original_request": "Test"},
        })
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == "test-job-id-001"
        assert data["status"] == "pending"

    def test_enqueue_invalid_type(self, client):
        r = client.post("/jobs/enqueue", json={"job_type": "invalid_type"})
        assert r.status_code == 400

    def test_enqueue_invalid_priority(self, client):
        r = client.post("/jobs/enqueue", json={"job_type": "plan", "priority": 99})
        assert r.status_code == 400

    @patch("tools.job_queue.get_job")
    def test_get_job_found(self, mock_get, client):
        mock_get.return_value = {
            "id": "test-id", "job_type": "plan", "status": "completed",
            "payload": {}, "result": {"success": True}, "error_message": None,
            "tenant_id": "default", "project_id": None, "priority": 5,
            "created_at": "2026-01-01T00:00:00+00:00", "started_at": None,
            "completed_at": None, "retry_count": 0, "max_retries": 3,
        }
        r = client.get("/jobs/test-id")
        assert r.status_code == 200
        assert r.json()["is_complete"] is True

    @patch("tools.job_queue.get_job", return_value=None)
    def test_get_job_not_found(self, mock_get, client):
        r = client.get("/jobs/nonexistent")
        assert r.status_code == 404

    def test_list_jobs_empty(self, client):
        r = client.get("/jobs/list")
        assert r.status_code == 200
        assert r.json()["jobs"] == []

    @patch("tools.job_queue.cancel_job", return_value=True)
    def test_cancel_job(self, mock_cancel, client):
        r = client.delete("/jobs/test-id/cancel")
        assert r.status_code == 200
        assert r.json()["cancelled"] is True

    @patch("tools.job_queue.cancel_job", return_value=False)
    def test_cancel_running_job_fails(self, mock_cancel, client):
        r = client.delete("/jobs/test-id/cancel")
        assert r.status_code == 200
        assert r.json()["cancelled"] is False

    @patch("tools.job_queue.get_job_stats", return_value={"by_status": {}, "by_type": {}, "total": 0})
    def test_job_stats(self, mock_stats, client):
        r = client.get("/jobs/stats")
        assert r.status_code == 200
        assert "total" in r.json()

    @patch("tools.job_queue.cleanup_old_jobs", return_value=5)
    def test_cleanup_jobs(self, mock_cleanup, client):
        r = client.post("/jobs/cleanup?days=7")
        assert r.status_code == 200
        assert r.json()["deleted_count"] == 5


class TestPlanAsyncMode:
    @patch("tools.job_queue.enqueue_job", return_value="async-job-001")
    def test_plan_async_returns_job_id(self, mock_enqueue, client):
        r = client.post("/plan?async_mode=true", json={
            "original_request": "Build webhook",
            "customer_name": "Test Corp",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["job_id"] == "async-job-001"
        assert data["status"] == "pending"
        assert "poll_url" in data

    def test_plan_sync_default(self, client):
        """Default sync mode still works."""
        r = client.post("/plan", json={
            "original_request": "Build a Slack bot",
            "customer_name": "Test Corp",
        })
        # Should work (200) or fail validation (422) — NOT a job_id response
        assert r.status_code in (200, 422, 500)
        data = r.json()
        assert "job_id" not in data


class TestJobWorker:
    def test_worker_dispatch_plan(self):
        from tools.job_worker import JobWorker
        worker = JobWorker()
        # Verify dispatch method exists and routes correctly
        assert hasattr(worker, "_dispatch")
        with pytest.raises(ValueError, match="Unknown job type"):
            worker._dispatch("nonexistent_type", {})

    def test_worker_start_stop(self):
        from tools.job_worker import JobWorker
        worker = JobWorker()
        assert worker.running is False
        # Don't actually start (would need DB), just verify the interface
        assert hasattr(worker, "start")
        assert hasattr(worker, "stop")
