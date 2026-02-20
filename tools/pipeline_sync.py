"""
Pipeline Sync — Supabase Integration Layer

Syncs agenticmakebuilder pipeline state, verification results, financials,
and activity to the connect-hub Supabase instance in real time.

All functions handle exceptions silently — Supabase sync is non-critical.
If SUPABASE_URL is not set, all functions return None gracefully.
"""

from datetime import datetime, timezone
from typing import Optional

from tools.supabase_client import get_supabase_client


def sync_project_state(
    project_id: str,
    stage: str,
    agent_name: str,
    context_bundle: Optional[dict] = None,
) -> Optional[dict]:
    """Upsert pipeline state to project_agent_state in Supabase.

    Args:
        project_id: UUID string.
        stage: Current pipeline stage (intake, build, verify, deploy).
        agent_name: Name of the active agent.
        context_bundle: Optional context data for the current stage.

    Returns:
        Upserted row dict, or None on failure/not configured.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        now = datetime.now(timezone.utc).isoformat()
        data = {
            "project_id": project_id,
            "current_stage": stage,
            "current_agent": agent_name,
            "pipeline_health": "on_track",
            "updated_at": now,
        }
        if context_bundle:
            data["stage_history"] = context_bundle.get("stage_history", [])

        result = client.upsert(
            "project_agent_state", data, on_conflict="project_id"
        )
        return result[0] if result else None
    except Exception:
        return None


def sync_build_verification(
    project_id: str,
    confidence_score: float,
    passed: bool,
    fix_instructions: Optional[list] = None,
) -> Optional[dict]:
    """Insert a verification result to build_verifications in Supabase.

    Args:
        project_id: UUID string.
        confidence_score: 0-100 confidence score.
        passed: Whether verification passed.
        fix_instructions: List of fix instruction dicts if failed.

    Returns:
        Inserted row dict, or None on failure/not configured.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        data = {
            "project_id": project_id,
            "confidence_score": confidence_score,
            "passed": passed,
            "fix_instructions": fix_instructions,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        result = client.insert("build_verifications", data)
        return result[0] if result else None
    except Exception:
        return None


def sync_project_financials(
    project_id: str,
    revenue: float,
    api_cost: float,
) -> Optional[dict]:
    """Upsert project financials (revenue, cost, margin) to Supabase.

    Args:
        project_id: UUID string.
        revenue: Total project revenue.
        api_cost: Cumulative API cost.

    Returns:
        Upserted row dict, or None on failure/not configured.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        margin = revenue - api_cost
        margin_pct = (margin / revenue * 100) if revenue > 0 else 0.0

        data = {
            "project_id": project_id,
            "revenue": round(revenue, 2),
            "api_cost": round(api_cost, 4),
            "margin": round(margin, 4),
            "margin_pct": round(margin_pct, 1),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        result = client.upsert(
            "project_financials", data, on_conflict="project_id"
        )
        return result[0] if result else None
    except Exception:
        return None


def sync_activity(
    project_id: str,
    action_type: str,
    description: str,
    agent_name: str = "agenticmakebuilder",
    visibility: str = "internal",
) -> Optional[dict]:
    """Insert an activity record to project_activity in Supabase.

    Args:
        project_id: UUID string.
        action_type: Type of action (e.g. "build_started", "agent_handoff").
        description: Human-readable description.
        agent_name: Name of the acting agent.
        visibility: "internal" or "client".

    Returns:
        Inserted row dict, or None on failure/not configured.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        data = {
            "project_id": project_id,
            "action_type": action_type,
            "description": description,
            "agent_name": agent_name,
            "visibility": visibility,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = client.insert("project_activity", data)
        return result[0] if result else None
    except Exception:
        return None


def get_project_stage(project_id: str) -> Optional[str]:
    """Query project_agent_state in Supabase and return current stage.

    Args:
        project_id: UUID string.

    Returns:
        Current stage string, or None if not found/not configured.
    """
    client = get_supabase_client()
    if not client:
        return None

    try:
        rows = client.select(
            "project_agent_state",
            filters={"project_id": project_id},
            columns="current_stage",
            limit=1,
        )
        if rows:
            return rows[0].get("current_stage")
        return None
    except Exception:
        return None
