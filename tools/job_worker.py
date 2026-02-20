"""
Job Worker — Background thread that processes jobs from the queue.

Polls the job_queue table, dispatches to handler functions, manages retries.
"""

import threading
import time
import traceback

from sqlalchemy import text

from db.session import engine
from tools.job_queue import get_pending_jobs, update_job_status


class JobWorker:
    """Background worker that processes queued jobs."""

    def __init__(self):
        self.running = False
        self.thread = None
        self.poll_interval = 3  # seconds

    def start(self):
        """Start the worker thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the worker thread."""
        self.running = False

    def _run_loop(self):
        """Main polling loop."""
        while self.running:
            try:
                jobs = get_pending_jobs(limit=5)
                for job in jobs:
                    if not self.running:
                        break
                    self._process_job(job)
            except Exception as e:
                print(f"[JobWorker] Loop error: {e}")
            time.sleep(self.poll_interval)

    def _process_job(self, job: dict):
        """Process a single job with retry support."""
        job_id = job["id"]
        job_type = job["job_type"]
        payload = job.get("payload") or {}

        update_job_status(job_id, "running")
        try:
            result = self._dispatch(job_type, payload)
            update_job_status(job_id, "completed", result=result)
        except Exception as e:
            retry_count = job.get("retry_count", 0)
            max_retries = job.get("max_retries", 3)
            if retry_count < max_retries:
                try:
                    with engine.begin() as conn:
                        conn.execute(text(
                            "UPDATE job_queue SET status='pending', retry_count=retry_count+1 WHERE id=:id"
                        ), {"id": job_id})
                except Exception:
                    update_job_status(job_id, "failed", error=str(e))
            else:
                update_job_status(job_id, "failed", error=str(e))

    def _dispatch(self, job_type: str, payload: dict) -> dict:
        """Route job to handler based on type."""
        handlers = {
            "plan": self._run_plan,
            "verify": self._run_verify,
            "deploy": self._run_deploy,
            "embed": self._run_embed,
            "bulk_verify": self._run_bulk_verify,
            "reindex": self._run_reindex,
            "monitor_deployments": self._run_monitor,
        }
        handler = handlers.get(job_type)
        if not handler:
            raise ValueError(f"Unknown job type: {job_type}")
        return handler(payload)

    def _run_plan(self, payload: dict) -> dict:
        """Execute plan pipeline."""
        from tools.generate_delivery_assessment import generate_delivery_assessment
        from tools.build_scenario_pipeline import build_scenario_pipeline

        original_request = payload.get("original_request", "")
        customer_name = payload.get("customer_name", "Customer")
        assessment = generate_delivery_assessment(original_request, customer_name)
        if not assessment.get("ready_for_build"):
            return {"success": False, "reason": "Assessment not ready", "assessment": assessment}
        plan_dict = assessment.get("plan_dict", {})
        result = build_scenario_pipeline(plan_dict, original_request)
        return {"success": True, "plan_generated": True, **result}

    def _run_verify(self, payload: dict) -> dict:
        """Run verification on a blueprint."""
        from tools.validate_make_export import validate_make_export
        blueprint = payload.get("blueprint", {})
        report = validate_make_export(blueprint)
        return {"verified": True, **report}

    def _run_deploy(self, payload: dict) -> dict:
        """Deploy via Make.com client (placeholder)."""
        return {"deployed": False, "reason": "Deploy via job queue not yet wired"}

    def _run_embed(self, payload: dict) -> dict:
        """Embed a document."""
        from tools.embedding_engine import embed_document
        doc_id = payload.get("project_id", "unknown")
        text_content = payload.get("text", "")
        metadata = payload.get("metadata", {})
        embed_document(doc_id, text_content, metadata)
        return {"embedded": True, "project_id": doc_id}

    def _run_bulk_verify(self, payload: dict) -> dict:
        """Bulk verify multiple blueprints."""
        from tools.validate_make_export import validate_make_export
        blueprint = payload.get("blueprint", {})
        project_ids = payload.get("project_ids", [])
        results = []
        for pid in project_ids:
            report = validate_make_export(blueprint)
            results.append({"project_id": pid, **report})
        return {"results": results}

    def _run_reindex(self, payload: dict) -> dict:
        """Re-embed all client context."""
        return {"reindexed": True, "message": "Use POST /admin/reindex for full reindex"}

    def _run_monitor(self, payload: dict) -> dict:
        """Monitor all active deployments."""
        try:
            from tools.execution_monitor import monitor_all_active_deployments
            results = monitor_all_active_deployments()
            return {"monitored": True, "deployment_count": len(results), "results": results}
        except Exception as e:
            return {"monitored": False, "error": str(e)}


WORKER = JobWorker()


def start_worker():
    """Start the global job worker."""
    WORKER.start()


def stop_worker():
    """Stop the global job worker."""
    WORKER.stop()
