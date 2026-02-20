"""
Job Queue — Background task management.

Provides enqueue/dequeue/status operations for the job_queue table.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import text

from db.session import engine


def enqueue_job(
    job_type: str,
    payload: dict,
    project_id: str = None,
    priority: int = 5,
    tenant_id: str = "default",
) -> str:
    """Insert a job into the queue. Returns job_id."""
    import json
    try:
        with engine.begin() as conn:
            result = conn.execute(text(
                "INSERT INTO job_queue (job_type, payload, project_id, priority, tenant_id) "
                "VALUES (:jt, :payload, :pid, :pri, :tid) RETURNING id"
            ), {
                "jt": job_type,
                "payload": json.dumps(payload),
                "pid": project_id,
                "pri": priority,
                "tid": tenant_id,
            })
            return str(result.fetchone()[0])
    except Exception as e:
        raise RuntimeError(f"Enqueue failed: {e}")


def get_job(job_id: str) -> Optional[dict]:
    """Get full job record by ID."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text("SELECT * FROM job_queue WHERE id = :jid"), {"jid": job_id}).fetchone()
            if not row:
                return None
            return {
                "id": str(row.id),
                "job_type": row.job_type,
                "status": row.status,
                "payload": row.payload,
                "result": row.result,
                "error_message": row.error_message,
                "tenant_id": row.tenant_id,
                "project_id": row.project_id,
                "priority": row.priority,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "started_at": row.started_at.isoformat() if row.started_at else None,
                "completed_at": row.completed_at.isoformat() if row.completed_at else None,
                "retry_count": row.retry_count,
                "max_retries": row.max_retries,
            }
    except Exception:
        return None


def update_job_status(job_id: str, status: str, result: dict = None, error: str = None):
    """Update job status and optionally set result or error."""
    import json
    sets = ["status = :status"]
    params = {"jid": job_id, "status": status}

    if status == "running":
        sets.append("started_at = :now")
        params["now"] = datetime.now(timezone.utc)
    if status in ("completed", "failed", "cancelled"):
        sets.append("completed_at = :now")
        params["now"] = datetime.now(timezone.utc)
    if result is not None:
        sets.append("result = :result")
        params["result"] = json.dumps(result)
    if error is not None:
        sets.append("error_message = :err")
        params["err"] = error

    try:
        with engine.begin() as conn:
            conn.execute(text(f"UPDATE job_queue SET {', '.join(sets)} WHERE id = :jid"), params)
    except Exception:
        pass


def get_pending_jobs(job_type: str = None, limit: int = 10) -> list:
    """Get pending jobs ordered by priority DESC, created_at ASC."""
    query = "SELECT * FROM job_queue WHERE status = 'pending'"
    params = {"lim": limit}
    if job_type:
        query += " AND job_type = :jt"
        params["jt"] = job_type
    query += " ORDER BY priority DESC, created_at ASC LIMIT :lim"

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(query), params).fetchall()
            return [
                {
                    "id": str(r.id), "job_type": r.job_type, "payload": r.payload,
                    "project_id": r.project_id, "priority": r.priority,
                    "retry_count": r.retry_count, "max_retries": r.max_retries,
                }
                for r in rows
            ]
    except Exception:
        return []


def cancel_job(job_id: str) -> bool:
    """Cancel a pending job. Returns True if cancelled."""
    try:
        with engine.begin() as conn:
            result = conn.execute(text(
                "UPDATE job_queue SET status = 'cancelled', completed_at = now() "
                "WHERE id = :jid AND status = 'pending'"
            ), {"jid": job_id})
            return result.rowcount > 0
    except Exception:
        return False


def get_job_stats() -> dict:
    """Queue statistics: counts by status and type."""
    stats = {"by_status": {}, "by_type": {}, "total": 0}
    try:
        with engine.connect() as conn:
            status_rows = conn.execute(text(
                "SELECT status, COUNT(*) as cnt FROM job_queue GROUP BY status"
            )).fetchall()
            stats["by_status"] = {r.status: r.cnt for r in status_rows}
            stats["total"] = sum(r.cnt for r in status_rows)

            type_rows = conn.execute(text(
                "SELECT job_type, COUNT(*) as cnt FROM job_queue GROUP BY job_type"
            )).fetchall()
            stats["by_type"] = {r.job_type: r.cnt for r in type_rows}
    except Exception:
        pass
    return stats


def cleanup_old_jobs(days: int = 7) -> int:
    """Delete old completed/failed/cancelled jobs. Returns count deleted."""
    try:
        with engine.begin() as conn:
            result = conn.execute(text(
                "DELETE FROM job_queue WHERE status IN ('completed', 'cancelled', 'failed') "
                "AND completed_at < now() - make_interval(days => :days)"
            ), {"days": days})
            return result.rowcount
    except Exception:
        return 0
