"""
Event Bus — Webhook-based event delivery system.

Publishes events to registered webhook subscribers with HMAC signatures.
Delivery is fire-and-forget via background threads.
"""

import hashlib
import hmac
import json
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text

from db.session import engine

EVENT_TYPES = [
    "project.created",
    "project.plan_generated",
    "project.verified",
    "project.deployed",
    "project.deploy_failed",
    "project.stalled",
    "pipeline.advanced",
    "execution.success",
    "execution.failure",
    "cost.threshold_exceeded",
    "persona.feedback_received",
]


def publish_event(event_type: str, payload: dict, tenant_id: str = "default"):
    """Publish an event to all matching subscribers (fire-and-forget)."""
    thread = threading.Thread(
        target=_deliver_event,
        args=(event_type, payload, tenant_id),
        daemon=True,
    )
    thread.start()


def _deliver_event(event_type: str, payload: dict, tenant_id: str):
    """Deliver event to all active subscribers."""
    try:
        with engine.connect() as conn:
            subs = conn.execute(text(
                "SELECT id, target_url, secret FROM webhook_subscriptions "
                "WHERE event_type = :et AND tenant_id = :tid AND active = true AND failure_count < 5"
            ), {"et": event_type, "tid": tenant_id}).fetchall()
    except Exception:
        return

    for sub in subs:
        _send_webhook(sub, event_type, payload)


def _send_webhook(sub, event_type: str, payload: dict):
    """Send a single webhook delivery."""
    import httpx

    body = json.dumps({
        "event": event_type,
        "payload": payload,
        "delivered_at": datetime.now(timezone.utc).isoformat(),
    })
    headers = {"Content-Type": "application/json", "X-Event-Type": event_type}

    if sub.secret:
        sig = hmac.new(sub.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
        headers["X-Signature-256"] = f"sha256={sig}"

    start = time.time()
    try:
        resp = httpx.post(sub.target_url, content=body, headers=headers, timeout=10)
        duration_ms = int((time.time() - start) * 1000)
        success = 200 <= resp.status_code < 300

        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO webhook_deliveries "
                "(subscription_id, event_type, payload, response_status, response_body, success, duration_ms) "
                "VALUES (:sid, :et, :payload, :rs, :rb, :s, :d)"
            ), {
                "sid": str(sub.id), "et": event_type,
                "payload": body, "rs": resp.status_code,
                "rb": resp.text[:500], "s": success, "d": duration_ms,
            })
            if success:
                conn.execute(text(
                    "UPDATE webhook_subscriptions SET last_triggered_at = now(), failure_count = 0 WHERE id = :id"
                ), {"id": str(sub.id)})
            else:
                conn.execute(text(
                    "UPDATE webhook_subscriptions SET failure_count = failure_count + 1 WHERE id = :id"
                ), {"id": str(sub.id)})
    except Exception:
        try:
            with engine.begin() as conn:
                conn.execute(text(
                    "UPDATE webhook_subscriptions SET failure_count = failure_count + 1 WHERE id = :id"
                ), {"id": str(sub.id)})
        except Exception:
            pass
