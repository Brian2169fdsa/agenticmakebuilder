"""
Live Scenario Monitor — Execution Poller

Polls Make.com API every 15 minutes for execution history
across all registered client scenarios.

Runs as a background task (FastAPI lifespan) or standalone cron.
Fires Slack alerts immediately for high/critical incidents.

Configuration:
    MAKE_API_KEY        — required for live polling
    MAKE_TEAM_ID        — required for live polling
    MONITOR_INTERVAL    — poll interval in seconds (default: 900 = 15 min)
    MONITOR_LOOKBACK    — how many recent executions to analyze (default: 20)
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone


MONITOR_INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "900"))
MONITOR_LOOKBACK = int(os.environ.get("MONITOR_LOOKBACK", "20"))

# Last snapshot: slug → {"snapped_at": ..., "success_rate": ..., "error_count": ...}
_snapshots: dict = {}


def _load_registry() -> list:
    """Load active scenarios from registry.json."""
    registry_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scenarios", "registry.json"
    )
    if not os.path.exists(registry_path):
        return []
    with open(registry_path) as f:
        data = json.load(f)
    return [s for s in data.get("scenarios", []) if s.get("active", True)]


def poll_once(db_session=None) -> dict:
    """
    Run one full poll cycle across all registered scenarios.

    Returns:
        dict with scenarios_checked, incidents_created, snapshots, errors
    """
    from tools.logger import log
    from tools.error_classifier import classify_execution_history
    from tools.incident_manager import auto_detect_and_create

    api_key = os.environ.get("MAKE_API_KEY")
    team_id = os.environ.get("MAKE_TEAM_ID")

    if not api_key or not team_id:
        return {
            "success": False,
            "error": "MAKE_API_KEY and MAKE_TEAM_ID required for live monitoring.",
            "dry_run": True,
        }

    from tools.make_api_client import MakeAPIClient, MakeAPIError
    client = MakeAPIClient(api_key=api_key, team_id=int(team_id))

    scenarios = _load_registry()
    results = []
    total_incidents = []
    errors = []

    for scenario in scenarios:
        slug = scenario.get("slug", "unknown")
        customer_name = scenario.get("customer_name", "Customer")
        make_scenario_id = scenario.get("make_scenario_id")

        if not make_scenario_id:
            results.append({
                "slug": slug, "status": "skipped",
                "reason": "No make_scenario_id in registry"
            })
            continue

        try:
            executions = client.get_execution_history(
                int(make_scenario_id), limit=MONITOR_LOOKBACK
            )
            analysis = classify_execution_history(executions)

            # Store snapshot
            _snapshots[slug] = {
                "slug": slug,
                "customer_name": customer_name,
                "snapped_at": datetime.now(timezone.utc).isoformat(),
                "executions_checked": len(executions),
                "success_count": analysis["success_count"],
                "error_count": analysis["error_count"],
                "success_rate": analysis["success_rate"],
                "dominant_error_type": analysis["dominant_error_type"],
                "needs_attention": analysis["needs_attention"],
            }

            # Auto-create incidents if needed
            new_incidents = auto_detect_and_create(
                slug, customer_name, analysis, db_session=db_session
            )
            total_incidents.extend(new_incidents)

            if new_incidents:
                log("monitor.incident_created", slug=slug,
                    count=len(new_incidents), level="warning")

            results.append({
                "slug": slug,
                "status": "ok",
                "success_rate": analysis["success_rate"],
                "new_incidents": len(new_incidents),
            })

        except MakeAPIError as e:
            errors.append({"slug": slug, "error": str(e)})
            log("monitor.poll_error", slug=slug, error=str(e), level="error")

    if db_session:
        try:
            db_session.commit()
        except Exception:
            db_session.rollback()

    log("monitor.poll_complete",
        scenarios_checked=len(results),
        incidents_created=len(total_incidents),
        errors=len(errors))

    return {
        "success": True,
        "polled_at": datetime.now(timezone.utc).isoformat(),
        "scenarios_checked": len(results),
        "incidents_created": len(total_incidents),
        "new_incidents": total_incidents,
        "results": results,
        "errors": errors,
    }


def get_monitor_status() -> dict:
    """
    Return current health status from latest snapshots.
    Used by GET /monitor/status endpoint.
    """
    from tools.incident_manager import get_open_count_by_severity, list_incidents

    snapshots = list(_snapshots.values())
    needs_attention = [s for s in snapshots if s.get("needs_attention")]
    healthy = [s for s in snapshots if not s.get("needs_attention")]

    open_incidents = list_incidents(status="open")
    open_critical = [i for i in open_incidents if i["severity"] in ("high", "critical")]

    return {
        "status": "critical" if open_critical else ("warning" if needs_attention else "healthy"),
        "monitored_scenarios": len(snapshots),
        "healthy": len(healthy),
        "needs_attention": len(needs_attention),
        "open_incidents": len(open_incidents),
        "critical_incidents": len(open_critical),
        "incident_severity_counts": get_open_count_by_severity(),
        "last_poll": max((s["snapped_at"] for s in snapshots), default=None),
        "snapshots": snapshots,
    }


async def run_monitor_loop(db_session_factory=None):
    """
    Async loop for FastAPI background monitoring.
    Polls every MONITOR_INTERVAL seconds.

    Usage in lifespan:
        asyncio.create_task(run_monitor_loop(SessionLocal))
    """
    from tools.logger import log
    log("monitor.started", interval_seconds=MONITOR_INTERVAL)

    while True:
        db = None
        try:
            if db_session_factory:
                db = db_session_factory()
            result = poll_once(db_session=db)
        except Exception as e:
            log("monitor.loop_error", error=str(e), level="error")
        finally:
            if db:
                try:
                    db.close()
                except Exception:
                    pass

        await asyncio.sleep(MONITOR_INTERVAL)


if __name__ == "__main__":
    print("=== Execution Poller Self-Check ===\n")

    print("Test 1: poll_once without API key → dry_run mode")
    os.environ.pop("MAKE_API_KEY", None)
    os.environ.pop("MAKE_TEAM_ID", None)
    result = poll_once()
    assert result["success"] == False
    assert result["dry_run"] == True
    print(f"  Error: {result['error']}")
    print("  [OK]")

    print("Test 2: get_monitor_status with empty snapshots")
    _snapshots.clear()
    status = get_monitor_status()
    assert status["monitored_scenarios"] == 0
    assert status["status"] == "healthy"
    print("  [OK]")

    print("Test 3: get_monitor_status with snapshot data")
    _snapshots["test-slug"] = {
        "slug": "test-slug",
        "customer_name": "Test Corp",
        "snapped_at": datetime.now(timezone.utc).isoformat(),
        "executions_checked": 10,
        "success_count": 7,
        "error_count": 3,
        "success_rate": 0.7,
        "dominant_error_type": "rate_limit",
        "needs_attention": True,
    }
    status = get_monitor_status()
    assert status["monitored_scenarios"] == 1
    assert status["needs_attention"] == 1
    print(f"  Status: {status['status']}")
    print("  [OK]")

    print("\n=== All poller checks passed ===")
    print("Note: Live polling requires MAKE_API_KEY + MAKE_TEAM_ID + make_scenario_id in registry.")
