"""
Tests for deployment endpoints — Make.com integration.

Mocks MakecomClient and DB to test endpoint behavior.
"""

import json
import os
import shutil
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import app
from db.session import get_db

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.json")


@pytest.fixture(autouse=True)
def clean_embeddings():
    """Save and restore embeddings.json so learn tests don't pollute it."""
    backup = None
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH) as f:
            backup = f.read()
    yield
    if backup is not None:
        with open(EMBEDDINGS_PATH, "w") as f:
            f.write(backup)


@pytest.fixture
def mock_db():
    """Create a fresh mock session for each test."""
    session = MagicMock()
    session.execute.return_value.fetchall.return_value = []
    session.execute.return_value.scalar.return_value = 0
    session.execute.return_value.first.return_value = None
    session.query.return_value.filter.return_value.first.return_value = None
    session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
    session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
    session.query.return_value.order_by.return_value.limit.return_value.all.return_value = []
    return session


@pytest.fixture
def client(mock_db):
    """TestClient with DB dependency overridden."""
    app.dependency_overrides[get_db] = lambda: mock_db
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestDeployMakecom:
    def test_deploy_makecom_project_not_found(self, client):
        """Returns 404 when project_id doesn't exist."""
        resp = client.post("/deploy/makecom", json={
            "project_id": "00000000-0000-0000-0000-000000000099",
            "blueprint": {"processing_steps": []},
        })
        assert resp.status_code == 404

    def test_deploy_makecom_simulation_mode(self, client, mock_db):
        """Returns simulation response when MAKECOM_API_KEY not set."""
        # Mock project exists
        mock_project = MagicMock()
        mock_project.id = "00000000-0000-0000-0000-000000000001"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_project

        resp = client.post("/deploy/makecom", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
            "blueprint": {"processing_steps": [{"description": "webhook trigger"}]},
            "scenario_name": "Test Deploy",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["simulation"] is True
        assert data["deployed"] is False
        assert data["blueprint_valid"] is True

    def test_deploy_makecom_missing_blueprint(self, client):
        """Returns 422 when blueprint not provided."""
        resp = client.post("/deploy/makecom", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
        })
        assert resp.status_code == 422


class TestDeployRun:
    def test_deploy_run_not_configured(self, client):
        """Returns error when MAKECOM_API_KEY not set."""
        resp = client.post("/deploy/run", json={"scenario_id": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "not_configured"

    def test_deploy_run_with_data(self, client):
        """Passes data parameter correctly."""
        resp = client.post("/deploy/run", json={
            "scenario_id": 42, "data": {"key": "value"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("error") == "not_configured"


class TestDeployList:
    def test_deploy_list_empty(self, client):
        """Returns empty list when no deployments."""
        resp = client.get("/deploy/list")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["deployments"] == []

    def test_deploy_list_with_project_filter(self, client):
        """Filters by project_id."""
        resp = client.get("/deploy/list?project_id=00000000-0000-0000-0000-000000000001")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["deployments"], list)


class TestDeployTeardown:
    def test_deploy_teardown_not_configured(self, client):
        """Returns error when MAKECOM_API_KEY not set."""
        resp = client.post("/deploy/teardown", json={
            "scenario_id": 42,
            "project_id": "00000000-0000-0000-0000-000000000001",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is False
        assert data["error"] == "not_configured"


class TestDeployMonitor:
    def test_deploy_monitor_empty(self, client):
        """Returns monitoring summary with no active deployments."""
        resp = client.get("/deploy/monitor")
        assert resp.status_code == 200
        data = resp.json()
        assert "monitored_count" in data
        assert "checked_at" in data


class TestWebhookMakecom:
    def test_webhook_success_event(self, client):
        """Processes successful execution event."""
        resp = client.post("/webhook/makecom", json={
            "scenario_id": "42",
            "execution_id": "exec-123",
            "status": "success",
            "duration_ms": 1500,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["received"] is True
        assert data["processed"] is True

    def test_webhook_error_event(self, client):
        """Processes error execution event."""
        resp = client.post("/webhook/makecom", json={
            "scenario_id": "42",
            "execution_id": "exec-err",
            "status": "error",
            "error_message": "Module timeout",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["received"] is True
        assert data["status"] == "error"

    def test_webhook_minimal_payload(self, client):
        """Handles minimal payload without crashing."""
        resp = client.post("/webhook/makecom", json={"scenario_id": "99"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["received"] is True


class TestLearnOutcome:
    def test_learn_outcome_success(self, client):
        """Records deployment outcome and returns learned=True."""
        resp = client.post("/learn/outcome", json={
            "scenario_id": "42",
            "project_id": "proj-123",
            "outcome": "success",
            "execution_count": 10,
            "avg_duration_ms": 500.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["learned"] is True
        assert data["embedded"] is True

    def test_learn_outcome_with_errors(self, client):
        """Includes error patterns in learning summary."""
        resp = client.post("/learn/outcome", json={
            "scenario_id": "42",
            "project_id": "proj-456",
            "outcome": "failure",
            "error_patterns": ["timeout", "auth_expired"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "timeout" in data["learning_summary"]


class TestLearnInsights:
    def test_learn_insights_no_data(self, client):
        """Returns risk assessment even with no embedding data."""
        resp = client.get("/learn/insights?description=webhook+slack+automation")
        assert resp.status_code == 200
        data = resp.json()
        assert "recommendation" in data
        assert "risk_level" in data
        assert "similar_outcomes" in data

    def test_learn_insights_with_prior_data(self, client):
        """Returns insights after embedding learning data."""
        client.post("/learn/outcome", json={
            "scenario_id": "42", "project_id": "proj-ins",
            "outcome": "success", "execution_count": 20,
        })
        resp = client.get("/learn/insights?description=scenario+execution")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["similar_outcomes"], list)
