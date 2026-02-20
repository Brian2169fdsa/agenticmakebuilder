"""
Notification Sender — Supabase department_notifications integration.

Sends structured notifications to the connect-hub dashboard for
stall alerts, cost warnings, and general agent activity.

All functions handle exceptions silently — notifications are non-critical.
"""

from datetime import datetime, timezone
from typing import Optional

from tools.supabase_client import get_supabase_client


def send_notification(
    department: str,
    type: str,
    title: str,
    message: str,
    reference_type: str,
    reference_id: Optional[str] = None,
) -> Optional[dict]:
    """Insert a notification into department_notifications in Supabase.

    Args:
        department: Target department (e.g. "engineering", "operations").
        type: Notification type (e.g. "warning", "alert", "info").
        title: Short notification title.
        message: Full notification body.
        reference_type: What the notification refers to (e.g. "project", "cost").
        reference_id: Optional UUID of the referenced entity.

    Returns:
        Inserted row dict, or None on failure/not configured.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        data = {
            "department": department,
            "type": type,
            "title": title,
            "message": message,
            "reference_type": reference_type,
            "reference_id": reference_id,
            "read": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = client.insert("department_notifications", data)
        return result[0] if result else None
    except Exception:
        return None


def send_stall_alert(project_id: str, days_stalled: float) -> Optional[dict]:
    """Send a stall alert notification for a stuck project.

    Args:
        project_id: UUID string of the stalled project.
        days_stalled: Number of days the project has been stalled.

    Returns:
        Inserted notification dict, or None.
    """
    days_rounded = round(days_stalled, 1)
    return send_notification(
        department="operations",
        type="warning",
        title=f"Project stalled: {days_rounded} days",
        message=(
            f"Project {project_id} has been stalled for {days_rounded} days "
            f"with no pipeline activity. Review required."
        ),
        reference_type="project",
        reference_id=project_id,
    )


def send_cost_alert(project_id: str, margin_pct: float) -> Optional[dict]:
    """Send a cost alert notification when margin is dangerously low.

    Args:
        project_id: UUID string.
        margin_pct: Current margin percentage.

    Returns:
        Inserted notification dict, or None.
    """
    if margin_pct < 0:
        level = "alert"
        title = f"Negative margin: {margin_pct:.1f}%"
        message = (
            f"Project {project_id} has negative margin ({margin_pct:.1f}%). "
            f"API costs exceed revenue. Immediate review required."
        )
    else:
        level = "warning"
        title = f"Low margin warning: {margin_pct:.1f}%"
        message = (
            f"Project {project_id} margin is below 20% ({margin_pct:.1f}%). "
            f"Cost optimization recommended."
        )

    return send_notification(
        department="operations",
        type=level,
        title=title,
        message=message,
        reference_type="cost",
        reference_id=project_id,
    )


def send_build_fail_alert(
    project_id: str, confidence_score: float, fix_count: int = 0
) -> Optional[dict]:
    """Send a build failure alert notification.

    Args:
        project_id: UUID string.
        confidence_score: The failing confidence score.
        fix_count: Number of fix instructions generated.

    Returns:
        Inserted notification dict, or None.
    """
    return send_notification(
        department="customer_delivery",
        type="alert",
        title=f"Build failed: confidence {confidence_score:.1f}%",
        message=(
            f"Project {project_id} build verification failed with "
            f"confidence {confidence_score:.1f}%. "
            f"{fix_count} fix instruction(s) generated. Review required."
        ),
        reference_type="build",
        reference_id=project_id,
    )
