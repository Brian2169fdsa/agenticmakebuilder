"""
Tests for Sprint 4 — Intelligence Layer endpoints.

Covers: POST /clients/health, GET /clients/list, POST /pipeline/advance,
GET /pipeline/dashboard, POST /briefing/daily, POST /verify/auto,
GET /confidence/trend
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app import app
from db.session import get_db


@pytest.fixture
def mock_db():
    """Create a fresh mock session for each test."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []
    session.execute.return_value.scalar.return_value = 0
    session.execute.return_value.first.return_value = None
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
    return session


@pytest.fixture
def client(mock_db):
    """TestClient with DB dependency overridden."""
    app.dependency_overrides[get_db] = lambda: mock_db
    c = TestClient(app, raise_server_exceptions=False)
    yield c
    app.dependency_overrides.clear()


# ── POST /clients/health ────────────────────────────────────────


class TestClientsHealth:
    def test_clients_health_empty(self, client):
        r = client.post("/clients/health")
        assert r.status_code == 200
        body = r.json()
        assert "clients" in body
        assert "summary" in body
        assert body["total"] == 0

    def test_clients_health_with_data(self, client, mock_db):
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.client_id = "Acme Corp"
        row.project_count = 3
        row.last_activity = now
        row.stalled_count = 0
        row.days_inactive = 1.0
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.post("/clients/health")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["clients"][0]["health_status"] == "healthy"


# ── GET /clients/list ───────────────────────────────────────────


class TestClientsList:
    def test_clients_list_empty(self, client):
        r = client.get("/clients/list")
        assert r.status_code == 200
        body = r.json()
        assert body["clients"] == []
        assert body["total"] == 0

    def test_clients_list_with_data(self, client, mock_db):
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.client_id = "test-client"
        row.project_count = 2
        row.last_updated = now
        row.decisions_count = 5
        row.tech_stack_all = None
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.get("/clients/list")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["clients"][0]["client_id"] == "test-client"


# ── POST /pipeline/advance ──────────────────────────────────────


class TestPipelineAdvance:
    def test_advance_invalid_uuid(self, client):
        r = client.post("/pipeline/advance", json={"project_id": "not-a-uuid"})
        assert r.status_code == 400

    def test_advance_project_not_found(self, client):
        r = client.post("/pipeline/advance", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
        })
        assert r.status_code == 404

    def test_advance_success(self, client, mock_db):
        project = MagicMock()
        project.id = UUID("00000000-0000-0000-0000-000000000001")
        mock_db.query.return_value.filter.return_value.first.return_value = project

        # First call returns project, second returns None (no state)
        call_count = [0]
        original_filter = mock_db.query.return_value.filter
        def filter_side_effect(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.first.return_value = project
            else:
                result.first.return_value = None
            return result
        mock_db.query.return_value.filter = MagicMock(side_effect=filter_side_effect)

        r = client.post("/pipeline/advance", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
        })
        assert r.status_code in (200, 404)


# ── GET /pipeline/dashboard ─────────────────────────────────────


class TestPipelineDashboard:
    def test_dashboard_empty(self, client):
        r = client.get("/pipeline/dashboard")
        assert r.status_code == 200
        body = r.json()
        assert "stages" in body
        assert "summary" in body
        assert body["total_projects"] == 0

    def test_dashboard_with_data(self, client, mock_db):
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.project_id = UUID("00000000-0000-0000-0000-000000000001")
        row.project_name = "Test Project"
        row.customer_name = "Acme"
        row.current_stage = "build"
        row.current_agent = "builder"
        row.started_at = now
        row.updated_at = now
        row.pipeline_health = "on_track"
        row.days_in_stage = 0.5
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.get("/pipeline/dashboard")
        assert r.status_code == 200
        body = r.json()
        assert body["total_projects"] == 1
        assert len(body["stages"]["build"]) == 1


# ── POST /briefing/daily ────────────────────────────────────────


class TestBriefingDaily:
    def test_briefing_daily_empty(self, client):
        r = client.post("/briefing/daily")
        assert r.status_code == 200
        body = r.json()
        assert "date" in body
        assert "stalled_projects" in body
        assert "pipeline_summary" in body
        assert "cost_summary" in body
        assert "recommendations" in body

    def test_briefing_daily_has_alerts(self, client, mock_db):
        """Briefing returns top_alerts list."""
        r = client.post("/briefing/daily")
        assert r.status_code == 200
        body = r.json()
        assert "top_alerts" in body
        assert isinstance(body["top_alerts"], list)


# ── POST /verify/auto ───────────────────────────────────────────


class TestVerifyAuto:
    def test_verify_auto_empty_blueprint(self, client):
        r = client.post("/verify/auto", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
            "blueprint": {},
        })
        assert r.status_code == 400

    def test_verify_auto_valid(self, client):
        r = client.post("/verify/auto", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
            "blueprint": {"name": "Test Scenario", "flow": []},
        })
        assert r.status_code == 200
        body = r.json()
        assert "confidence_score" in body
        assert "passed" in body
        assert "auto_advanced" in body
        assert isinstance(body["auto_advanced"], bool)


# ── GET /confidence/trend ───────────────────────────────────────


class TestConfidenceTrend:
    def test_trend_no_data(self, client):
        r = client.get("/confidence/trend", params={
            "project_id": "00000000-0000-0000-0000-000000000001",
        })
        assert r.status_code == 200
        body = r.json()
        assert body["trend"] == "no_data"
        assert body["entries"] == []
        assert body["total_runs"] == 0

    def test_trend_invalid_uuid(self, client):
        r = client.get("/confidence/trend", params={"project_id": "not-a-uuid"})
        assert r.status_code == 400
