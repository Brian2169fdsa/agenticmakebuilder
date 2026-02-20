"""
Tests for Phase 5 — Learning Feedback Loop.

Covers: POST /learn/outcome, GET /learn/insights, GET /learn/summary
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


class TestLearnOutcome:
    @patch("tools.embedding_engine.embed_document")
    def test_post_outcome_success(self, mock_embed, client):
        r = client.post("/learn/outcome", json={
            "scenario_id": "scn-001",
            "project_id": "proj-001",
            "outcome": "success",
            "execution_count": 10,
            "avg_duration_ms": 450,
            "error_patterns": [],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["learned"] is True
        assert data["embedded"] is True
        assert "learning_summary" in data

    @patch("tools.embedding_engine.embed_document")
    def test_post_outcome_failure(self, mock_embed, client):
        r = client.post("/learn/outcome", json={
            "scenario_id": "scn-002",
            "project_id": "proj-002",
            "outcome": "failure",
            "error_patterns": ["timeout", "rate_limit"],
        })
        assert r.status_code == 200
        assert r.json()["learned"] is True

    @patch("tools.embedding_engine.embed_document")
    def test_post_outcome_partial(self, mock_embed, client):
        r = client.post("/learn/outcome", json={
            "scenario_id": "scn-003",
            "project_id": "proj-003",
            "outcome": "partial",
        })
        assert r.status_code == 200


class TestLearnInsights:
    def test_insights_returns_recommendation(self, client):
        r = client.get("/learn/insights", params={"description": "webhook automation"})
        assert r.status_code == 200
        data = r.json()
        assert "recommendation" in data
        assert "risk_level" in data
        assert "similar_outcomes" in data

    def test_insights_missing_description(self, client):
        r = client.get("/learn/insights")
        assert r.status_code == 422

    @patch("tools.embedding_engine.find_similar")
    def test_insights_risk_low(self, mock_similar, client):
        mock_similar.return_value = [
            {"id": "p1", "score": 0.9, "metadata": {"outcome": "success"}},
            {"id": "p2", "score": 0.8, "metadata": {"outcome": "success"}},
        ]
        r = client.get("/learn/insights", params={"description": "test"})
        assert r.status_code == 200
        assert r.json()["risk_level"] == "low"

    @patch("tools.embedding_engine.find_similar")
    def test_insights_risk_high(self, mock_similar, client):
        mock_similar.return_value = [
            {"id": "p1", "score": 0.9, "metadata": {"outcome": "failure"}},
            {"id": "p2", "score": 0.8, "metadata": {"outcome": "failure"}},
        ]
        r = client.get("/learn/insights", params={"description": "test"})
        assert r.status_code == 200
        assert r.json()["risk_level"] == "high"


class TestLearnSummary:
    def test_summary_empty(self, client):
        r = client.get("/learn/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total_outcomes"] == 0
        assert data["success_rate"] == 0

    def test_summary_with_data(self, client, mock_db):
        row1 = MagicMock()
        row1.outcome = "success"
        row1.cnt = 5
        row1.avg_exec = 10.0
        row1.avg_dur = 300.0
        row2 = MagicMock()
        row2.outcome = "failure"
        row2.cnt = 2
        row2.avg_exec = 3.0
        row2.avg_dur = 150.0

        # First call returns outcome stats, second returns error patterns
        call_count = [0]
        def execute_side_effect(*args, **kwargs):
            call_count[0] += 1
            result = MagicMock()
            if call_count[0] == 1:
                result.fetchall.return_value = [row1, row2]
            else:
                result.fetchall.return_value = []
            return result
        mock_db.execute = MagicMock(side_effect=execute_side_effect)

        r = client.get("/learn/summary")
        assert r.status_code == 200
        data = r.json()
        assert data["total_outcomes"] == 7
        assert data["success_count"] == 5
        assert data["failure_count"] == 2
