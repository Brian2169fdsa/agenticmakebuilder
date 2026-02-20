"""
Tests for Phase 6 — Webhook Event Bus.

Covers: tools/event_bus.py, POST /events/subscribe,
DELETE /events/subscriptions/{id}, GET /events/subscriptions,
GET /events/deliveries, POST /events/test, GET /events/types
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


class TestEventTypes:
    def test_get_event_types(self, client):
        r = client.get("/events/types")
        assert r.status_code == 200
        data = r.json()
        assert "event_types" in data
        assert "project.created" in data["event_types"]
        assert "project.deployed" in data["event_types"]
        assert "execution.failure" in data["event_types"]
        assert data["total"] > 0

    def test_event_types_constant(self):
        from tools.event_bus import EVENT_TYPES
        assert isinstance(EVENT_TYPES, list)
        assert len(EVENT_TYPES) >= 10


class TestEventSubscriptions:
    def test_subscribe_valid_event(self, client, mock_db):
        row = MagicMock()
        row.id = "00000000-0000-0000-0000-000000000001"
        mock_db.execute.return_value.fetchone.return_value = row

        r = client.post("/events/subscribe", json={
            "event_type": "project.deployed",
            "target_url": "https://example.com/webhook",
        })
        assert r.status_code == 200
        data = r.json()
        assert "subscription_id" in data
        assert data["event_type"] == "project.deployed"

    def test_subscribe_invalid_event_type(self, client):
        r = client.post("/events/subscribe", json={
            "event_type": "nonexistent.event",
            "target_url": "https://example.com/webhook",
        })
        assert r.status_code == 422

    def test_unsubscribe(self, client):
        r = client.delete("/events/subscriptions/00000000-0000-0000-0000-000000000001")
        assert r.status_code == 200
        assert r.json()["deactivated"] is True

    def test_list_subscriptions_empty(self, client):
        r = client.get("/events/subscriptions")
        assert r.status_code == 200
        assert r.json()["subscriptions"] == []
        assert r.json()["total"] == 0

    def test_list_subscriptions_with_data(self, client, mock_db):
        row = MagicMock()
        row.id = "00000000-0000-0000-0000-000000000001"
        row.event_type = "project.deployed"
        row.target_url = "https://example.com/hook"
        row.tenant_id = "default"
        row.active = True
        row.failure_count = 0
        row.last_triggered_at = None
        row.created_at = MagicMock()
        row.created_at.isoformat.return_value = "2026-01-01T00:00:00"
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.get("/events/subscriptions")
        assert r.status_code == 200
        assert r.json()["total"] == 1


class TestEventDeliveries:
    def test_list_deliveries_empty(self, client):
        r = client.get("/events/deliveries")
        assert r.status_code == 200
        assert r.json()["deliveries"] == []

    def test_list_deliveries_with_data(self, client, mock_db):
        row = MagicMock()
        row.id = "00000000-0000-0000-0000-000000000001"
        row.subscription_id = "00000000-0000-0000-0000-000000000002"
        row.event_type = "project.deployed"
        row.response_status = 200
        row.success = True
        row.duration_ms = 150
        row.delivered_at = MagicMock()
        row.delivered_at.isoformat.return_value = "2026-01-01T00:00:00"
        mock_db.execute.return_value.fetchall.return_value = [row]

        r = client.get("/events/deliveries")
        assert r.status_code == 200
        assert r.json()["total"] == 1
        assert r.json()["deliveries"][0]["success"] is True


class TestEventTest:
    def test_test_event_not_found(self, client):
        r = client.post("/events/test?subscription_id=00000000-0000-0000-0000-000000000001")
        assert r.status_code == 404

    @patch("tools.event_bus.publish_event")
    def test_test_event_fires(self, mock_publish, client, mock_db):
        sub = MagicMock()
        sub.id = "00000000-0000-0000-0000-000000000001"
        sub.target_url = "https://example.com/hook"
        sub.secret = None
        sub.event_type = "project.deployed"
        mock_db.execute.return_value.fetchone.return_value = sub

        r = client.post("/events/test?subscription_id=00000000-0000-0000-0000-000000000001")
        assert r.status_code == 200
        assert r.json()["sent"] is True
        mock_publish.assert_called_once()
