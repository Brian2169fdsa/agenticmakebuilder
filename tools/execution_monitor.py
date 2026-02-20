"""
Execution Monitor — Make.com scenario health checker.

Monitors deployed scenarios by checking execution logs,
calculating success rates, and triggering alerts for failing scenarios.
"""

from typing import Optional

from tools.makecom_client import get_makecom_client, MakecomError


def check_scenario_health(scenario_id: int) -> dict:
    """Check health of a single Make.com scenario by its execution history.

    Args:
        scenario_id: Make.com scenario ID.

    Returns:
        Dict with health status, success_rate, execution stats.
    """
    try:
        client = get_makecom_client()
        executions = client.get_executions(scenario_id, limit=10)
    except MakecomError:
        return {
            "scenario_id": scenario_id,
            "health": "unknown",
            "error": "Could not fetch executions",
            "success_rate": 0,
            "total": 0,
        }

    if not executions:
        return {
            "scenario_id": scenario_id,
            "health": "inactive",
            "success_rate": 0,
            "total": 0,
            "successful": 0,
            "failed": 0,
            "warnings": 0,
        }

    total = len(executions)
    successful = sum(1 for e in executions if e.get("status") == 1)
    failed = sum(1 for e in executions if e.get("status") == 4)
    warnings = sum(1 for e in executions if e.get("status") == 3)

    success_rate = successful / total if total > 0 else 0

    durations = [e.get("duration", 0) for e in executions if e.get("duration")]
    avg_duration_ms = sum(durations) / len(durations) if durations else 0

    if success_rate >= 0.9:
        health = "healthy"
    elif success_rate >= 0.7:
        health = "degraded"
    else:
        health = "failing"

    return {
        "scenario_id": scenario_id,
        "health": health,
        "success_rate": round(success_rate, 3),
        "total": total,
        "successful": successful,
        "failed": failed,
        "warnings": warnings,
        "avg_duration_ms": round(avg_duration_ms),
    }


def monitor_all_active_deployments(db) -> list:
    """Monitor all active (non-deleted) deployments.

    Args:
        db: SQLAlchemy session.

    Returns:
        List of health dicts for each monitored scenario.
    """
    from sqlalchemy import text

    try:
        rows = db.execute(text(
            "SELECT external_id, project_id FROM deployments "
            "WHERE status != 'deleted' AND external_id IS NOT NULL AND target = 'make.com' "
            "ORDER BY deployed_at DESC"
        )).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        external_id = row[0]
        project_id = str(row[1])
        try:
            scenario_id = int(external_id)
            health = check_scenario_health(scenario_id)
            health["project_id"] = project_id

            # Send alerts for failing scenarios
            if health["health"] == "failing":
                try:
                    from tools.notification_sender import send_notification
                    send_notification(
                        department="operations", type="alert",
                        title=f"Scenario {scenario_id} is failing",
                        message=f"Success rate: {health['success_rate']*100:.0f}%. "
                                f"{health['failed']} failures in last {health['total']} executions.",
                        reference_type="deployment", reference_id=project_id,
                    )
                except Exception:
                    pass
            elif health["health"] == "degraded":
                try:
                    from tools.notification_sender import send_notification
                    send_notification(
                        department="operations", type="warning",
                        title=f"Scenario {scenario_id} is degraded",
                        message=f"Success rate: {health['success_rate']*100:.0f}%.",
                        reference_type="deployment", reference_id=project_id,
                    )
                except Exception:
                    pass

            results.append(health)
        except (ValueError, TypeError):
            continue

    return results
