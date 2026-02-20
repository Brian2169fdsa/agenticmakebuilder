"""
Tests for Phase 7 — Multi-Tenant Management.

Covers: GET /tenants, GET /tenants/{id}, POST /tenants,
PATCH /tenants/{id}, DELETE /tenants/{id}, GET /tenants/{id}/usage
"""

from unittest.mock import MagicMock
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


class TestListTenants:
    def test_list_empty(self, client):
        r = client.get("/tenants")
        assert r.status_code == 200
        assert r.json()["tenants"] == []
        assert r.json()["total"] == 0

    def test_list_with_data(self, client, mock_db):
        row = MagicMock()
        row.id = "default"
        row.name = "Default Tenant"
        row.plan = "enterprise"
        row.rate_limit_per_minute = 60
        row.max_projects = 50
        row.max_api_keys = 5
        row.features = ["plan", "verify"]
        row.active = True
        row.created_at = MagicMock()
        row.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.get("/tenants")
        assert r.status_code == 200
        assert r.json()["total"] == 1
        assert r.json()["tenants"][0]["id"] == "default"


class TestGetTenant:
    def test_get_not_found(self, client):
        r = client.get("/tenants/nonexistent")
        assert r.status_code == 404

    def test_get_found(self, client, mock_db):
        tenant = MagicMock()
        tenant.id = "default"
        tenant.name = "Default Tenant"
        tenant.plan = "enterprise"
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
                result.scalar.return_value = 3
                result.fetchone.return_value = None
            return result
        mock_db.execute = MagicMock(side_effect=execute_side_effect)

        r = client.get("/tenants/default")
        assert r.status_code == 200
        data = r.json()
        assert data["id"] == "default"
        assert data["plan"] == "enterprise"


class TestCreateTenant:
    def test_create_success(self, client):
        r = client.post("/tenants", json={
            "id": "new-tenant",
            "name": "New Tenant",
            "plan": "standard",
        })
        assert r.status_code == 200
        assert r.json()["created"] is True
        assert r.json()["tenant_id"] == "new-tenant"


class TestUpdateTenant:
    def test_update_plan(self, client):
        r = client.patch("/tenants/default", json={"plan": "professional"})
        assert r.status_code == 200
        assert r.json()["updated"] is True

    def test_update_no_fields(self, client):
        r = client.patch("/tenants/default", json={})
        assert r.status_code == 400


class TestDeleteTenant:
    def test_delete_success(self, client):
        r = client.delete("/tenants/test-tenant")
        assert r.status_code == 200
        assert r.json()["deleted"] is True

    def test_delete_default_blocked(self, client):
        r = client.delete("/tenants/default")
        assert r.status_code == 400


class TestTenantUsage:
    def test_usage_returns_counts(self, client):
        r = client.get("/tenants/default/usage")
        assert r.status_code == 200
        data = r.json()
        assert data["tenant_id"] == "default"
        assert "project_count" in data
        assert "api_key_count" in data
        assert "jobs_today" in data
        assert "subscriptions" in data
        assert "plan_limits" in data
