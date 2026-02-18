"""
Incident Manager

Creates, tracks, and resolves incidents detected by the live monitor.
In-memory store with DB persistence when db_session is available.

Incident lifecycle:
    detected → open → (acknowledged) → resolved

Fires Slack alerts for high/critical incidents immediately.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional


# In-memory store for when DB is not available
_incidents: dict = {}


def create_incident(
    slug: str,
    customer_name: str,
    incident_type: str,
    severity: str,
    details: dict,
    db_session=None,
) -> dict:
    """
    Create a new incident.

    Returns:
        Incident dict with id, slug, type, severity, status, detected_at
    """
    incident_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    incident = {
        "id": incident_id,
        "slug": slug,
        "customer_name": customer_name,
        "incident_type": incident_type,
        "severity": severity,
        "status": "open",
        "detected_at": now,
        "acknowledged_at": None,
        "resolved_at": None,
        "details": details,
        "resolution_note": None,
    }

    _incidents[incident_id] = incident

    # Persist to DB if available
    if db_session:
        try:
            _persist_incident(db_session, incident)
        except Exception:
            pass

    # Fire alert for high/critical severity
    if severity in ("high", "critical"):
        _fire_incident_alert(incident)

    return dict(incident)


def acknowledge_incident(incident_id: str, note: str = None, db_session=None) -> Optional[dict]:
    if incident_id not in _incidents:
        return None
    _incidents[incident_id]["status"] = "acknowledged"
    _incidents[incident_id]["acknowledged_at"] = datetime.now(timezone.utc).isoformat()
    if note:
        _incidents[incident_id]["resolution_note"] = note
    return dict(_incidents[incident_id])


def resolve_incident(incident_id: str, note: str = None, db_session=None) -> Optional[dict]:
    if incident_id not in _incidents:
        return None
    _incidents[incident_id]["status"] = "resolved"
    _incidents[incident_id]["resolved_at"] = datetime.now(timezone.utc).isoformat()
    if note:
        _incidents[incident_id]["resolution_note"] = note
    return dict(_incidents[incident_id])


def get_incident(incident_id: str) -> Optional[dict]:
    incident = _incidents.get(incident_id)
    return dict(incident) if incident else None


def list_incidents(
    status: str = None,
    severity: str = None,
    slug: str = None,
    limit: int = 50,
) -> list:
    incidents = list(_incidents.values())
    if status:
        incidents = [i for i in incidents if i["status"] == status]
    if severity:
        incidents = [i for i in incidents if i["severity"] == severity]
    if slug:
        incidents = [i for i in incidents if i["slug"] == slug]
    incidents.sort(key=lambda i: i["detected_at"], reverse=True)
    return [dict(i) for i in incidents[:limit]]


def get_open_count_by_severity() -> dict:
    open_incidents = [i for i in _incidents.values() if i["status"] in ("open", "acknowledged")]
    counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    for i in open_incidents:
        sev = i.get("severity", "low")
        if sev in counts:
            counts[sev] += 1
    return counts


def auto_detect_and_create(
    slug: str,
    customer_name: str,
    execution_analysis: dict,
    db_session=None,
) -> list:
    """
    Given execution analysis from error_classifier, auto-detect and
    create incidents for patterns that warrant attention.

    Returns list of newly created incidents.
    """
    created = []
    if not execution_analysis.get("needs_attention"):
        return created

    dominant = execution_analysis.get("dominant_error_type")
    error_count = execution_analysis.get("error_count", 0)
    success_rate = execution_analysis.get("success_rate", 1.0)

    # Check if we already have an open incident of this type
    existing = list_incidents(status="open", slug=slug)
    existing_types = {i["incident_type"] for i in existing}

    if dominant and dominant not in existing_types:
        # Determine severity from error pattern
        high_severity_types = {"credential_invalid", "credential_expired", "webhook_dead"}
        severity = "high" if dominant in high_severity_types else (
            "critical" if error_count > 10 or success_rate < 0.2 else "medium"
        )

        from tools.error_classifier import _get_suggestion
        incident = create_incident(
            slug=slug,
            customer_name=customer_name,
            incident_type=dominant,
            severity=severity,
            details={
                "error_count": error_count,
                "success_rate": success_rate,
                "dominant_error_type": dominant,
                "suggestion": _get_suggestion(dominant),
                "execution_sample": execution_analysis.get("incidents", [])[:3],
            },
            db_session=db_session,
        )
        created.append(incident)

    return created


def _persist_incident(db_session, incident: dict):
    """Persist incident to DB. Uses raw SQL for portability."""
    from sqlalchemy import text
    db_session.execute(text("""
        INSERT INTO incidents (id, slug, customer_name, incident_type, severity,
                               status, detected_at, details)
        VALUES (:id, :slug, :customer_name, :incident_type, :severity,
                :status, :detected_at, :details::jsonb)
        ON CONFLICT (id) DO NOTHING
    """), {
        "id": incident["id"],
        "slug": incident["slug"],
        "customer_name": incident["customer_name"],
        "incident_type": incident["incident_type"],
        "severity": incident["severity"],
        "status": incident["status"],
        "detected_at": incident["detected_at"],
        "details": str(incident["details"]).replace("'", '"'),
    })


def _fire_incident_alert(incident: dict):
    """Fire Slack alert for high/critical incidents."""
    import os
    try:
        from tools.alerting import send_slack_alert
        slack_url = os.environ.get("SLACK_WEBHOOK_URL")
        if slack_url:
            send_slack_alert(
                slug=incident["slug"],
                customer_name=incident["customer_name"],
                verdict=f"INCIDENT: {incident['incident_type'].replace('_', ' ').title()}",
                confidence_score=0.0,
                error_count=incident["details"].get("error_count", 0),
                recommendations=[incident["details"].get("suggestion", "")],
                webhook_url=slack_url,
            )
    except Exception:
        pass


if __name__ == "__main__":
    print("=== Incident Manager Self-Check ===\n")
    _incidents.clear()

    print("Test 1: Create incident")
    inc = create_incident(
        "acme-form-to-slack", "Acme Corp", "credential_invalid", "high",
        {"error_count": 5, "suggestion": "Re-auth Slack"}
    )
    assert inc["status"] == "open"
    assert inc["severity"] == "high"
    assert "id" in inc
    print(f"  ID: {inc['id'][:8]}...")
    print("  [OK]")

    print("Test 2: Get incident")
    fetched = get_incident(inc["id"])
    assert fetched["slug"] == "acme-form-to-slack"
    print("  [OK]")

    print("Test 3: Acknowledge incident")
    acked = acknowledge_incident(inc["id"], note="Checking with team")
    assert acked["status"] == "acknowledged"
    print("  [OK]")

    print("Test 4: Resolve incident")
    resolved = resolve_incident(inc["id"], note="Re-authenticated Slack connection")
    assert resolved["status"] == "resolved"
    assert resolved["resolution_note"] == "Re-authenticated Slack connection"
    print("  [OK]")

    print("Test 5: List incidents by status")
    open_incs = list_incidents(status="open")
    assert len(open_incs) == 0
    all_incs = list_incidents()
    assert len(all_incs) == 1
    print("  [OK]")

    print("Test 6: Open count by severity")
    create_incident("test", "Test", "rate_limit", "medium", {})
    create_incident("test2", "Test2", "connection_timeout", "high", {})
    counts = get_open_count_by_severity()
    assert counts["medium"] >= 1
    assert counts["high"] >= 1
    print(f"  Counts: {counts}")
    print("  [OK]")

    print("Test 7: auto_detect_and_create")
    _incidents.clear()
    analysis = {
        "needs_attention": True,
        "dominant_error_type": "rate_limit",
        "error_count": 4,
        "success_rate": 0.6,
        "incidents": [],
    }
    created = auto_detect_and_create("beta-test", "Beta Corp", analysis)
    assert len(created) == 1
    assert created[0]["incident_type"] == "rate_limit"
    print(f"  Auto-created: {created[0]['incident_type']}")
    print("  [OK]")

    print("Test 8: auto_detect skips if incident already exists")
    created2 = auto_detect_and_create("beta-test", "Beta Corp", analysis)
    assert len(created2) == 0
    print("  [OK]")

    print("\n=== All incident manager checks passed ===")
