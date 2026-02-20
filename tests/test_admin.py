"""
Tests for Sprint 6 — Admin Control Plane endpoints.

Covers: POST /admin/reset-project, POST /admin/bulk-verify,
GET /admin/system-status, POST /admin/reindex, GET /admin/audit-log
"""

from datetime import datetime, timezone
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


# ── POST /admin/reset-project ───────────────────────────────────


class TestAdminResetProject:
    def test_reset_invalid_uuid(self, client):
        r = client.post("/admin/reset-project", json={
            "project_id": "not-a-uuid",
        })
        assert r.status_code == 400

    def test_reset_project_not_found(self, client):
        r = client.post("/admin/reset-project", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
        })
        assert r.status_code == 404

    def test_reset_invalid_stage(self, client, mock_db):
        project = MagicMock()
        project.id = UUID("00000000-0000-0000-0000-000000000001")
        mock_db.query.return_value.filter.return_value.first.return_value = project

        r = client.post("/admin/reset-project", json={
            "project_id": "00000000-0000-0000-0000-000000000001",
            "target_stage": "nonexistent",
        })
        assert r.status_code == 400


# ── POST /admin/bulk-verify ─────────────────────────────────────


class TestAdminBulkVerify:
    def test_bulk_verify_empty_list(self, client):
        r = client.post("/admin/bulk-verify", json={
            "project_ids": [],
            "blueprint": {"name": "Test"},
        })
        assert r.status_code == 400

    def test_bulk_verify_too_many(self, client):
        ids = [f"00000000-0000-0000-0000-00000000{i:04d}" for i in range(11)]
        r = client.post("/admin/bulk-verify", json={
            "project_ids": ids,
            "blueprint": {"name": "Test"},
        })
        assert r.status_code == 400

    def test_bulk_verify_single_project(self, client):
        r = client.post("/admin/bulk-verify", json={
            "project_ids": ["00000000-0000-0000-0000-000000000001"],
            "blueprint": {"name": "Test Scenario", "flow": []},
        })
        assert r.status_code == 200
        body = r.json()
        assert "results" in body
        assert len(body["results"]) == 1
        assert "confidence_score" in body["results"][0]
        assert "passed" in body["results"][0]


# ── GET /admin/system-status ────────────────────────────────────


class TestAdminSystemStatus:
    def test_system_status(self, client, mock_db):
        # Mock row count queries
        count_mock = MagicMock()
        count_mock.cnt = 5
        mock_db.execute.return_value.first.return_value = count_mock

        r = client.get("/admin/system-status")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "operational"
        assert "db" in body
        assert "embeddings" in body
        assert "registry" in body
        assert "pipeline" in body
        assert "costs" in body
        assert "personas" in body

    def test_system_status_has_table_counts(self, client):
        r = client.get("/admin/system-status")
        assert r.status_code == 200
        body = r.json()
        # Should have counts for key tables
        assert "projects" in body["db"]
        assert "builds" in body["db"]


# ── POST /admin/reindex ─────────────────────────────────────────


class TestAdminReindex:
    def test_reindex_empty(self, client):
        r = client.post("/admin/reindex")
        assert r.status_code == 200
        body = r.json()
        assert body["reindexed_count"] == 0
        assert body["skipped_count"] == 0
        assert "duration_ms" in body

    def test_reindex_with_data(self, client, mock_db):
        row = MagicMock()
        row.client_id = "test-client"
        row.project_id = "00000000-0000-0000-0000-000000000001"
        row.key_decisions = ["Use webhook"]
        row.tech_stack = ["Make.com"]
        row.failure_patterns = []
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.post("/admin/reindex")
        assert r.status_code == 200
        body = r.json()
        assert body["total_records"] == 1


# ── GET /admin/audit-log ────────────────────────────────────────


class TestAdminAuditLog:
    def test_audit_log_empty(self, client):
        r = client.get("/admin/audit-log")
        assert r.status_code == 200
        body = r.json()
        assert body["entries"] == []
        assert body["total"] == 0
        assert body["limit"] == 50

    def test_audit_log_with_data(self, client, mock_db):
        now = datetime.now(timezone.utc)
        row = MagicMock()
        row.from_agent = "assessor"
        row.to_agent = "builder"
        row.project_id = UUID("00000000-0000-0000-0000-000000000001")
        row.context_bundle = {"outcome": "success"}
        row.created_at = now
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.get("/admin/audit-log")
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 1
        assert body["entries"][0]["from_agent"] == "assessor"
        assert body["entries"][0]["outcome"] == "success"

    def test_audit_log_custom_limit(self, client):
        r = client.get("/admin/audit-log", params={"limit": 10})
        assert r.status_code == 200
        assert r.json()["limit"] == 10
