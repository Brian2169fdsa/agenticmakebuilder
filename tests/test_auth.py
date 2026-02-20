"""
Tests for Phase 1 — Authentication Layer.

Covers: tools/auth.py, POST /auth/keys, GET /auth/keys,
DELETE /auth/keys/{key_id}, POST /auth/keys/rotate
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


class TestAuthUtils:
    def test_generate_api_key_format(self):
        from tools.auth import generate_api_key
        raw, hashed = generate_api_key()
        assert raw.startswith("mab_")
        assert len(raw) > 20
        assert raw != hashed

    def test_generate_api_key_custom_prefix(self):
        from tools.auth import generate_api_key
        raw, hashed = generate_api_key(prefix="test")
        assert raw.startswith("test_")

    def test_hash_key_deterministic(self):
        from tools.auth import hash_key
        h1 = hash_key("test-key-123")
        h2 = hash_key("test-key-123")
        assert h1 == h2

    def test_hash_key_different_inputs(self):
        from tools.auth import hash_key
        h1 = hash_key("key-a")
        h2 = hash_key("key-b")
        assert h1 != h2

    def test_auth_disabled_returns_default(self):
        """When AUTH_ENABLED=false, get_api_key returns default permissions."""
        import asyncio
        from tools.auth import get_api_key
        # AUTH_ENABLED defaults to false in test environment
        result = asyncio.get_event_loop().run_until_complete(get_api_key(None))
        assert result["tenant_id"] == "default"
        assert "read" in result["permissions"]
        assert "write" in result["permissions"]

    @patch("tools.auth.AUTH_ENABLED", True)
    def test_auth_enabled_requires_key(self):
        import asyncio
        from tools.auth import get_api_key
        with pytest.raises(Exception) as exc:
            asyncio.get_event_loop().run_until_complete(get_api_key(None))
        assert "401" in str(exc.value) or "API key required" in str(exc.value)

    @patch("tools.auth.AUTH_ENABLED", True)
    @patch("tools.auth.MASTER_KEY", "super-secret-master")
    def test_master_key_bypasses(self):
        import asyncio
        from tools.auth import get_api_key
        result = asyncio.get_event_loop().run_until_complete(get_api_key("super-secret-master"))
        assert result["tenant_id"] == "master"
        assert "admin" in result["permissions"]


class TestAuthEndpoints:
    def test_post_auth_keys_creates_key(self, client, mock_db):
        row = MagicMock()
        row.id = "00000000-0000-0000-0000-000000000001"
        row.created_at = MagicMock()
        row.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        mock_db.execute.return_value.fetchone.return_value = row

        r = client.post("/auth/keys", json={"name": "Test Key"})
        assert r.status_code == 200
        data = r.json()
        assert "raw_key" in data
        assert data["raw_key"].startswith("mab_")
        assert "key_id" in data
        assert "warning" in data

    def test_get_auth_keys_empty(self, client):
        r = client.get("/auth/keys")
        assert r.status_code == 200
        assert r.json()["keys"] == []
        assert r.json()["total"] == 0

    def test_get_auth_keys_with_data(self, client, mock_db):
        row = MagicMock()
        row.id = "00000000-0000-0000-0000-000000000001"
        row.key_prefix = "mab_abc"
        row.name = "Test"
        row.tenant_id = "default"
        row.permissions = ["read", "write"]
        row.rate_limit_per_minute = 60
        row.last_used_at = None
        row.expires_at = None
        row.revoked = False
        row.created_at = MagicMock()
        row.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.get("/auth/keys")
        assert r.status_code == 200
        assert r.json()["total"] == 1
        assert r.json()["keys"][0]["name"] == "Test"

    def test_delete_auth_keys_revokes(self, client):
        r = client.delete("/auth/keys/00000000-0000-0000-0000-000000000001")
        assert r.status_code == 200
        assert r.json()["revoked"] is True

    def test_delete_auth_keys_invalid_id(self, client):
        r = client.delete("/auth/keys/not-a-uuid")
        assert r.status_code == 400

    def test_rotate_key_not_found(self, client):
        r = client.post("/auth/keys/rotate", json={"key_id": "00000000-0000-0000-0000-000000000001"})
        assert r.status_code == 404
