"""
Agentic Make Builder — FastAPI Application

Endpoints:
  GET  /health       — liveness probe
  POST /intake       — natural language → assessment (no structured fields needed)
  POST /assess       — structured intake → delivery report + plan_dict
  POST /plan         — structured intake → full pipeline (assess + build + 11 artifacts)
  POST /build        — plan_dict + original_request → full pipeline (compiler direct)
  POST /audit        — audit an existing Make.com scenario blueprint

HTTP status codes:
  200 — success
  400 — malformed request / missing required fields
  422 — validation failure (spec/export errors, confidence too low)
  500 — internal pipeline error
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from db.session import get_db, check_db
from tools.module_registry_loader import load_module_registry
from tools.build_scenario_pipeline import build_scenario_pipeline
from tools.generate_delivery_assessment import generate_delivery_assessment
from tools.validate_canonical_spec import validate_canonical_spec
from tools.validate_make_export import validate_make_export
from tools.confidence_scorer import compute_confidence

# --- Global registry loaded once at startup ---
_registry = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _registry
    _registry = load_module_registry()
    check_db()
    yield


app = FastAPI(
    title="Agentic Make Builder",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────

class AssessRequest(BaseModel):
    original_request: str
    customer_name: Optional[str] = "Customer"
    ticket_summary: Optional[str] = None
    business_objective: Optional[str] = None
    trigger_type: Optional[str] = None
    trigger_description: Optional[str] = None
    processing_steps: Optional[list] = None
    routing: Optional[dict] = None
    error_handling_preference: Optional[str] = "ignore"
    expected_outputs: Optional[list] = None
    estimated_frequency: Optional[str] = None
    project_name: Optional[str] = "default"


class BuildRequest(BaseModel):
    plan: dict
    original_request: str
    project_name: Optional[str] = "default"


class IntakeRequest(BaseModel):
    """Natural language intake — no structured fields required."""
    message: str
    customer_name: Optional[str] = "Customer"
    project_name: Optional[str] = "default"


class AuditRequest(BaseModel):
    """Audit an existing Make.com scenario blueprint."""
    blueprint: dict                        # Make.com export JSON
    scenario_name: Optional[str] = "Existing Scenario"
    customer_name: Optional[str] = "Customer"


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/intake")
def intake(request: IntakeRequest, db: Session = Depends(get_db)):
    """
    Natural language front door.
    Accepts a free-text message and attempts to extract intent,
    returning either a delivery assessment or clarification questions.

    Example:
      { "message": "When a Typeform is submitted, post the results to a Slack channel" }
    """
    # Build a minimal intake dict from the natural language message
    # The assessment tool will fill in what it can and flag what's missing
    synthetic_intake = {
        "original_request": request.message,
        "customer_name": request.customer_name,
        "ticket_summary": request.message,
        "business_objective": request.message,
        "trigger_type": None,       # Let the assessor infer
        "trigger_description": None,
        "processing_steps": [],
        "routing": {"needed": False},
        "error_handling_preference": "ignore",
        "expected_outputs": [],
        "estimated_frequency": None,
    }

    try:
        assessment = generate_delivery_assessment(synthetic_intake, _registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

    return {
        "ready_for_build": assessment.get("ready_for_build", False),
        "clarification_questions": assessment.get("clarification_questions", []),
        "delivery_report": assessment.get("delivery_report"),
        "plan_dict": assessment.get("plan_dict"),
        "hint": "Submit clarification answers via POST /assess with structured fields, or POST /plan to build directly."
    }


@app.post("/assess")
def assess(request: AssessRequest):
    """
    Structured intake → delivery report + plan_dict.
    Returns clarification questions if the intake is incomplete.
    Does not run the build pipeline.
    """
    intake = request.model_dump()

    try:
        result = generate_delivery_assessment(intake, _registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

    return result


@app.post("/plan")
def plan(request: AssessRequest, db: Session = Depends(get_db)):
    """
    Full pipeline: assess → build → 11 artifacts.
    Returns 422 if assessment is incomplete or confidence is too low.
    Returns 500 on pipeline exception.
    """
    intake = request.model_dump()
    project_name = intake.pop("project_name", "default") or "default"

    try:
        assessment = generate_delivery_assessment(intake, _registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

    if not assessment.get("ready_for_build"):
        return JSONResponse(
            status_code=422,
            content={
                "ready_for_build": False,
                "clarification_questions": assessment.get("clarification_questions", []),
                "delivery_report": assessment.get("delivery_report"),
            }
        )

    try:
        build_result = build_scenario_pipeline(
            plan=assessment["plan_dict"],
            registry=_registry,
            original_request=intake.get("original_request", ""),
            db_session=db,
            project_name=project_name,
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    if not build_result["success"]:
        return JSONResponse(
            status_code=422,
            content={
                "ready_for_build": True,
                "success": False,
                "failure_reason": build_result.get("failure_reason"),
                "confidence": build_result.get("confidence"),
                "canonical_validation": build_result.get("canonical_validation"),
                "export_validation": build_result.get("export_validation"),
            }
        )

    return {
        "ready_for_build": True,
        "success": True,
        "delivery_report": assessment.get("delivery_report"),
        "build_result": build_result,
    }


@app.post("/build")
def build(request: BuildRequest, db: Session = Depends(get_db)):
    """
    Compiler direct access. Requires a pre-built plan_dict.
    Use /plan for the full pipeline including assessment.
    """
    if not request.plan:
        raise HTTPException(status_code=400, detail="plan is required")
    if not request.original_request:
        raise HTTPException(status_code=400, detail="original_request is required")

    try:
        result = build_scenario_pipeline(
            plan=request.plan,
            registry=_registry,
            original_request=request.original_request,
            db_session=db,
            project_name=request.project_name or "default",
        )
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    if not result["success"]:
        return JSONResponse(
            status_code=422,
            content=result,
        )

    return result


@app.post("/audit")
def audit(request: AuditRequest):
    """
    Audit an existing Make.com scenario blueprint.

    Submit any Make.com export JSON (downloaded from Make.com → scenario → Export)
    and get back a validation report, confidence score, and list of issues.

    Use this for:
    - Daily health monitoring of live client scenarios
    - Pre-delivery quality checks on manually built scenarios
    - Comparing your scenario against ManageAI best practices
    """
    blueprint = request.blueprint
    scenario_name = request.scenario_name or blueprint.get("name", "Existing Scenario")

    if not blueprint:
        raise HTTPException(status_code=400, detail="blueprint is required")

    if "flow" not in blueprint:
        raise HTTPException(
            status_code=400,
            detail="Invalid Make.com blueprint — missing 'flow' array. "
                   "Export your scenario from Make.com and submit the full JSON."
        )

    try:
        export_report = validate_make_export(blueprint, _registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

    # Score the blueprint quality
    agent_notes = []
    confidence = compute_confidence(export_report, agent_notes=agent_notes, retry_count=0)

    # Summarize issues
    errors = export_report.get("errors", [])
    warnings = export_report.get("warnings", [])

    verdict = "HEALTHY"
    if len(errors) > 0:
        verdict = "NEEDS ATTENTION"
    if len(errors) > 5 or confidence["score"] < 0.60:
        verdict = "CRITICAL"

    return {
        "scenario_name": scenario_name,
        "customer_name": request.customer_name,
        "verdict": verdict,
        "confidence": {
            "score": confidence["score"],
            "grade": confidence["grade"],
            "explanation": confidence["explanation"],
        },
        "validation": {
            "valid": export_report.get("valid", False),
            "checks_run": export_report.get("checks_run", 0),
            "checks_passed": export_report.get("checks_passed", 0),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "errors": [
            {"rule_id": e["rule_id"], "message": e["message"], "severity": e.get("severity", "error")}
            for e in errors[:20]
        ],
        "warnings": [
            {"rule_id": w["rule_id"], "message": w["message"]}
            for w in warnings[:20]
        ],
        "recommendations": _audit_recommendations(errors, warnings, confidence),
        "hint": "Schedule this endpoint to run daily against your active client scenarios for automated health monitoring."
    }


def _audit_recommendations(errors, warnings, confidence):
    """Generate plain-English recommendations from audit results."""
    recs = []

    if confidence["score"] >= 0.90:
        recs.append("Scenario is in excellent health. No action required.")
    elif confidence["score"] >= 0.70:
        recs.append("Scenario is healthy but has minor issues worth reviewing.")
    else:
        recs.append("Scenario has significant issues that should be addressed before production use.")

    for e in errors[:5]:
        rule_id = e.get("rule_id", "")
        msg = e.get("message", "")
        if "credential" in msg.lower() or "connection" in msg.lower():
            recs.append(f"⚠️  Check connection/credential configuration: {msg}")
        elif "missing" in msg.lower():
            recs.append(f"⚠️  Missing required field: {msg}")
        else:
            recs.append(f"⚠️  [{rule_id}] {msg}")

    if len(warnings) > 3:
        recs.append(f"Consider reviewing {len(warnings)} warnings — they may indicate suboptimal configuration.")

    return recs
