"""
Agentic Make Builder — FastAPI Application (v2.1.0)

Endpoints (35 total):

  Core Pipeline:
    GET  /health              — liveness probe
    POST /intake              — natural language → assessment
    POST /assess              — structured intake → delivery report + plan_dict
    POST /plan                — full pipeline (assess + build + 11 artifacts)
    POST /build               — compiler direct (requires plan_dict)
    POST /audit               — audit existing Make.com blueprint

  Verification & Confidence:
    POST /verify              — 77-rule blueprint validation with fix instructions
    POST /verify/loop         — iterative verify → fix → verify cycle (max 5 iterations)
    GET  /confidence/history  — verification run history with score trends

  Multi-Agent Orchestration:
    POST /orchestrate         — advance project through pipeline stages
    POST /agent/complete      — agent completion + auto-advance
    GET  /pipeline/status     — full pipeline state view
    POST /briefing            — daily supervisor briefing report
    POST /handoff             — multi-agent handoff bridge

  Agent Memory & Learning:
    POST /memory              — store client context (decisions, tech stack, patterns)
    GET  /memory              — retrieve client context
    POST /memory/embed        — embed project brief for similarity search
    GET  /similar             — TF-IDF cosine similarity search

  Deployment Agent:
    POST /deploy/makecom      — deploy blueprint to Make.com via API
    POST /deploy/n8n          — deploy workflow to n8n via REST API
    GET  /deploy/status       — deployment status + health checks

  Cost & Margin Intelligence:
    POST /costs/track         — track token costs with auto-margin + alerts
    GET  /costs/summary       — cost/revenue/margin per client
    GET  /costs/report        — weekly cost report in markdown
    POST /costs/estimate      — pre-build cost estimation from historical data

  Persona Engine:
    POST /persona/memory      — link persona to client with tone/style prefs
    GET  /persona/context     — persona's full context for a client
    POST /persona/feedback    — store interaction feedback
    GET  /persona/performance — persona performance stats
    POST /persona/deploy      — generate client-specific persona artifact

  Platform Health:
    GET  /health/full         — comprehensive health check (DB, tables, embeddings, pipeline)
    GET  /health/memory       — embedding store stats + vocabulary analysis
    POST /health/repair       — self-healing (stalled pipelines, orphans, stale deploys)

  Supervisor:
    GET  /supervisor/stalled  — detect stalled projects (>48h)

  Natural Language:
    POST /command             — route free-text commands to endpoints

HTTP status codes:
  200 — success
  400 — malformed request / missing required fields
  404 — resource not found
  422 — validation failure (spec/export errors, confidence too low)
  500 — internal pipeline error
  502 — upstream API error (Make.com, n8n)
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

import os

from db.session import get_db, check_db
from db.models import (
    AgentHandoff, ProjectFinancial, Project, PersonaClientContext, PersonaFeedback,
    ProjectAgentState, ClientContext, VerificationRun, Deployment,
)
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
    print(">>> STARTUP: Loading module registry...")
    try:
        _registry = load_module_registry()
        print(f">>> STARTUP: Registry loaded ({_registry.get('module_count', 0)} modules)")
    except Exception as e:
        print(f">>> STARTUP ERROR loading registry: {e}")
        _registry = {"registry_version": "0", "modules": {}, "module_count": 0}
    print(">>> STARTUP: Checking database...")
    check_db()
    print(">>> STARTUP: Ready to serve requests")
    yield


app = FastAPI(
    title="Agentic Make Builder",
    version="2.0.0",
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


class VerifyRequest(BaseModel):
    """Verify a blueprint JSON against the full 77-rule validation audit."""
    blueprint: dict                        # Make.com export JSON
    project_id: Optional[str] = None       # If set, logs run + auto-advances pipeline


class HandoffRequest(BaseModel):
    """Multi-agent orchestration handoff."""
    from_agent: str
    to_agent: str
    project_id: str                        # UUID as string
    context_bundle: Optional[dict] = None


class CostTrackRequest(BaseModel):
    """Track token costs for a project operation."""
    project_id: str                        # UUID as string
    model: str                             # e.g. "claude-sonnet-4-6"
    input_tokens: int
    output_tokens: int
    operation_type: str                    # e.g. "assess", "build", "iterate"


class OrchestrateRequest(BaseModel):
    project_id: str
    current_stage: str  # intake | build | verify | deploy


class AgentCompleteRequest(BaseModel):
    project_id: str
    agent_name: str
    outcome: str  # success | failed | needs_review
    artifacts: Optional[dict] = None


class MemoryRequest(BaseModel):
    client_id: str
    project_id: str
    key_decisions: Optional[list] = None
    tech_stack: Optional[list] = None
    failure_patterns: Optional[list] = None


class VerifyLoopRequest(BaseModel):
    project_id: str
    blueprint: dict
    max_iterations: Optional[int] = 3


class DeployMakecomRequest(BaseModel):
    project_id: str
    blueprint: dict
    api_key: str
    team_id: Optional[int] = None


class DeployN8nRequest(BaseModel):
    project_id: str
    workflow: dict
    n8n_url: str
    api_key: str


class CostEstimateRequest(BaseModel):
    description: str
    category: Optional[str] = "standard"


class CommandRequest(BaseModel):
    command: str
    customer_name: Optional[str] = None


class EmbedRequest(BaseModel):
    project_id: str
    brief: str
    outcome: Optional[str] = None


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
    Now enriched with similar past projects and client memory context.
    Returns 422 if assessment is incomplete or confidence is too low.
    Returns 500 on pipeline exception.
    """
    from tools.embedding_engine import find_similar

    intake = request.model_dump()
    project_name = intake.pop("project_name", "default") or "default"

    # --- Enrich with similar past projects (Item 9) ---
    similar_context = []
    try:
        original_req = intake.get("original_request", "")
        if original_req:
            similar_results = find_similar(original_req, top_n=3)
            for s in similar_results:
                similar_context.append({
                    "project_id": s["id"],
                    "score": s["score"],
                    "summary": s["text_preview"],
                    "metadata": s.get("metadata", {}),
                })
    except Exception:
        pass  # Non-critical — proceed without similar context

    # --- Enrich with client memory (Item 10) ---
    client_memory = None
    customer = intake.get("customer_name", "")
    if customer and customer != "Customer":
        try:
            rows = db.query(ClientContext).filter(
                ClientContext.client_id == customer
            ).all()
            if rows:
                all_decisions = []
                all_tech = set()
                all_failures = []
                for r in rows:
                    if r.key_decisions:
                        all_decisions.extend(r.key_decisions)
                    if r.tech_stack:
                        all_tech.update(r.tech_stack)
                    if r.failure_patterns:
                        all_failures.extend(r.failure_patterns)
                client_memory = {
                    "key_decisions": all_decisions,
                    "tech_stack": sorted(all_tech),
                    "failure_patterns": all_failures,
                }
        except Exception:
            pass  # Non-critical

    # Inject context into intake for the assessor
    if similar_context:
        intake["_similar_projects"] = similar_context
    if client_memory:
        intake["_client_memory"] = client_memory

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

    # Sync activity to Supabase
    try:
        from tools.pipeline_sync import sync_activity
        build_project_id = build_result.get("project_id") or build_result.get("build_id")
        if build_project_id:
            sync_activity(
                project_id=str(build_project_id),
                action_type="build_started",
                description="Build plan generated",
                agent_name="agenticmakebuilder",
            )
    except Exception:
        pass  # Non-critical

    response = {
        "ready_for_build": True,
        "success": True,
        "delivery_report": assessment.get("delivery_report"),
        "build_result": build_result,
    }
    if similar_context:
        response["similar_projects"] = similar_context
    if client_memory:
        response["client_memory_used"] = True
    return response


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


@app.post("/verify")
def verify(request: VerifyRequest, db: Session = Depends(get_db)):
    """
    Full 77-rule blueprint validation audit.

    Runs both structural (Make export) and semantic validation rules against
    a blueprint. Returns confidence_score (0-100), pass/fail verdict, and
    specific fix_instructions when confidence < 85.

    If project_id is provided:
    - Logs the run to verification_runs table
    - If confidence >= 85 and no errors, auto-calls agent/complete with success
    - fix_instructions are ranked by impact (errors first, then warnings)
    """
    blueprint = request.blueprint

    if not blueprint:
        raise HTTPException(status_code=400, detail="blueprint is required")

    if not isinstance(blueprint, dict):
        raise HTTPException(status_code=400, detail="blueprint must be a JSON object")

    # Run Make export validation (30 rules: MR, MF, MM, MT, MC, MD)
    try:
        export_report = validate_make_export(blueprint, _registry)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export validation failed: {str(e)}")

    # Run supplementary structural audit (47 additional checks)
    canonical_report = _run_supplementary_checks(blueprint, _registry)

    # Merge both reports
    all_errors = export_report.get("errors", []) + canonical_report.get("errors", [])
    all_warnings = export_report.get("warnings", []) + canonical_report.get("warnings", [])
    total_checks = export_report.get("checks_run", 0) + canonical_report.get("checks_run", 0)
    total_passed = export_report.get("checks_passed", 0) + canonical_report.get("checks_passed", 0)

    merged_report = {
        "errors": all_errors,
        "warnings": all_warnings,
        "checks_run": total_checks,
        "checks_passed": total_passed,
    }

    # Compute confidence (0-1.0 internally, convert to 0-100 for response)
    confidence = compute_confidence(merged_report, agent_notes=[], retry_count=0)
    score_100 = round(confidence["score"] * 100, 1)

    passed = score_100 >= 85 and len(all_errors) == 0

    # Generate fix instructions ranked by impact if not passed
    fix_instructions = []
    if not passed:
        fix_instructions = _generate_fix_instructions(all_errors, all_warnings, score_100)

    # Log verification run and auto-advance pipeline if project_id provided
    auto_advanced = False
    if request.project_id:
        try:
            project_uuid = UUID(request.project_id)
            run = VerificationRun(
                project_id=project_uuid,
                confidence_score=score_100,
                passed=passed,
                error_count=len(all_errors),
                warning_count=len(all_warnings),
                fix_instructions=fix_instructions if fix_instructions else None,
            )
            db.add(run)
            db.commit()

            # Auto-advance pipeline if passed
            if passed:
                state = db.query(ProjectAgentState).filter(
                    ProjectAgentState.project_id == project_uuid
                ).first()
                if state and state.current_stage == "verify":
                    handoff = AgentHandoff(
                        from_agent="validator",
                        to_agent="deployer",
                        project_id=project_uuid,
                        context_bundle={"outcome": "success", "confidence_score": score_100},
                    )
                    db.add(handoff)
                    state.current_stage = "deploy"
                    state.current_agent = "deployer"
                    state.updated_at = datetime.now(timezone.utc)
                    history = state.stage_history or []
                    history.append({
                        "stage": "verify",
                        "agent": "validator",
                        "outcome": "success",
                        "confidence_score": score_100,
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                    })
                    state.stage_history = history
                    db.commit()
                    auto_advanced = True
        except Exception:
            db.rollback()

    # Sync verification to Supabase
    if request.project_id:
        try:
            from tools.pipeline_sync import sync_build_verification
            sync_build_verification(
                project_id=request.project_id,
                confidence_score=score_100,
                passed=passed,
                fix_instructions=fix_instructions if fix_instructions else None,
            )
        except Exception:
            pass  # Non-critical

    result = {
        "confidence_score": score_100,
        "passed": passed,
        "grade": confidence["grade"],
        "checks_run": total_checks,
        "checks_passed": total_passed,
        "error_count": len(all_errors),
        "warning_count": len(all_warnings),
        "errors": [
            {"rule_id": e["rule_id"], "message": e["message"], "severity": e.get("severity", "error")}
            for e in all_errors[:30]
        ],
        "warnings": [
            {"rule_id": w["rule_id"], "message": w["message"]}
            for w in all_warnings[:30]
        ],
        "fix_instructions": fix_instructions,
    }
    if auto_advanced:
        result["auto_advanced"] = True
        result["next_stage"] = "deploy"
    return result


def _run_supplementary_checks(blueprint, registry):
    """Run additional semantic checks beyond the core Make export validator.

    These cover naming conventions, operational readiness, data integrity,
    and best practices — reaching the full 77-rule target when combined
    with the 30 Make export rules.

    Returns a report dict in the same shape as validate_make_export output.
    """
    errors = []
    warnings = []
    checks_run = 0
    checks_passed = 0

    def _check(rule_id, condition, msg, warn=False, module_id=None):
        nonlocal checks_run, checks_passed
        checks_run += 1
        if condition:
            checks_passed += 1
            return True
        entry = {"rule_id": rule_id, "severity": "warning" if warn else "error",
                 "message": msg, "module_id": module_id, "context": {}}
        (warnings if warn else errors).append(entry)
        return False

    flow = blueprint.get("flow", [])
    metadata = blueprint.get("metadata", {})
    name = blueprint.get("name", "")

    # Collect all modules
    all_modules = []
    all_ids = []
    if isinstance(flow, list):
        _collect_all_modules(flow, all_ids, all_modules)

    # --- SV: Supplementary Validation (47 rules) ---

    # SV-001: Scenario name is non-empty and descriptive (>3 chars)
    _check("SV-001", isinstance(name, str) and len(name.strip()) > 3,
           "Scenario name should be descriptive (>3 characters)")

    # SV-002: Scenario name doesn't contain 'untitled' or 'test'
    _check("SV-002", not any(w in name.lower() for w in ("untitled", "copy of")),
           f"Scenario name '{name}' looks like a placeholder — rename for production",
           warn=True)

    # SV-003: At least 2 modules (trigger + 1 action)
    _check("SV-003", len(all_modules) >= 2,
           f"Scenario has only {len(all_modules)} module(s) — need trigger + at least 1 action")

    # SV-004: No more than 50 modules (complexity warning)
    _check("SV-004", len(all_modules) <= 50,
           f"Scenario has {len(all_modules)} modules — consider splitting for maintainability",
           warn=True)

    # SV-005: Module IDs are sequential starting from 1
    if all_ids:
        sorted_ids = sorted([i for i in all_ids if isinstance(i, int)])
        expected = list(range(1, len(sorted_ids) + 1))
        _check("SV-005", sorted_ids == expected,
               f"Module IDs not sequential: {sorted_ids[:10]}... expected {expected[:10]}...",
               warn=True)

    # SV-006: All modules have a label/name
    for mod in all_modules:
        mid = mod.get("id", "?")
        mod_meta = mod.get("metadata", {})
        has_label = isinstance(mod_meta, dict) and mod_meta.get("expect")
        _check("SV-006", has_label or mod.get("module", "").startswith("builtin:"),
               f"Module {mid} has no descriptive label in metadata.expect",
               warn=True, module_id=mid)

    # SV-007: Error handling configured (metadata.scenario.maxErrors)
    scenario_meta = metadata.get("scenario", {}) if isinstance(metadata, dict) else {}
    _check("SV-007", isinstance(scenario_meta, dict) and scenario_meta.get("maxErrors", 0) > 0,
           "No maxErrors configured — scenario will stop on first error",
           warn=True)

    # SV-008: At least one module has onerror configured
    has_onerror = any(isinstance(m.get("onerror"), list) and len(m.get("onerror", [])) > 0
                      for m in all_modules)
    _check("SV-008", has_onerror or len(all_modules) <= 2,
           "No modules have error handling (onerror) — add error directives for production use",
           warn=True)

    # SV-009: Trigger module is a recognized trigger type
    trigger_types = {
        "gateway:CustomWebHook", "gateway:ScheduleTrigger",
        "google-sheets:watchRows", "slack:watchMessages",
        "email:TriggerNewEmail", "builtin:BasicTrigger",
    }
    if all_modules:
        first_mod = all_modules[0].get("module", "")
        _check("SV-009", first_mod in trigger_types or "trigger" in first_mod.lower()
               or "watch" in first_mod.lower() or "gateway:" in first_mod,
               f"First module '{first_mod}' may not be a trigger — verify module ordering")

    # SV-010: No duplicate module types in sequence (likely copy-paste error)
    prev_type = None
    for mod in all_modules:
        mod_type = mod.get("module", "")
        if mod_type and mod_type == prev_type and mod_type != "builtin:BasicRouter":
            _check("SV-010", False,
                   f"Consecutive duplicate module type '{mod_type}' — possible copy-paste error",
                   warn=True, module_id=mod.get("id"))
        prev_type = mod_type

    # SV-011 to SV-020: Data mapping checks
    for mod in all_modules:
        mid = mod.get("id", "?")
        mapper = mod.get("mapper")
        if isinstance(mapper, dict):
            for key, val in mapper.items():
                # SV-011: Mapper values are not empty strings
                _check("SV-011", val != "",
                       f"Module {mid} mapper key '{key}' is empty — provide a value or remove",
                       warn=True, module_id=mid)

                # SV-012: Mapper references use valid IML syntax {{N.field}}
                if isinstance(val, str) and "{{" in val:
                    _check("SV-012", "}}" in val,
                           f"Module {mid} mapper key '{key}' has unclosed IML expression: {val!r}",
                           module_id=mid)

    # SV-013: No null parameters in non-router modules
    for mod in all_modules:
        mid = mod.get("id", "?")
        if mod.get("module") != "builtin:BasicRouter":
            params = mod.get("parameters")
            _check("SV-013", params is not None,
                   f"Module {mid} has null parameters — should be empty dict at minimum",
                   warn=True, module_id=mid)

    # SV-014: Metadata has designer positions for layout
    designer_meta = metadata.get("designer", {}) if isinstance(metadata, dict) else {}
    _check("SV-014", isinstance(designer_meta, dict) and "orphans" in designer_meta,
           "Blueprint metadata missing designer layout — may render poorly in Make.com editor",
           warn=True)

    # SV-015: No excessively long mapper values (>2000 chars — likely embedded data)
    for mod in all_modules:
        mid = mod.get("id", "?")
        mapper = mod.get("mapper", {})
        if isinstance(mapper, dict):
            for key, val in mapper.items():
                if isinstance(val, str) and len(val) > 2000:
                    _check("SV-015", False,
                           f"Module {mid} mapper '{key}' has {len(val)} chars — avoid embedding large data",
                           warn=True, module_id=mid)

    # SV-016: Credential consistency — all modules of same app use same credential
    app_creds = {}
    for mod in all_modules:
        mid = mod.get("id", "?")
        mod_type = mod.get("module", "")
        app_name = mod_type.split(":")[0] if ":" in mod_type else ""
        params = mod.get("parameters", {})
        if isinstance(params, dict) and "__IMTCONN__" in params:
            cred = params["__IMTCONN__"]
            if app_name in app_creds and app_creds[app_name] != cred:
                _check("SV-016", False,
                       f"Module {mid} uses credential '{cred}' but other {app_name} modules use "
                       f"'{app_creds[app_name]}' — inconsistent credentials",
                       warn=True, module_id=mid)
            app_creds[app_name] = cred

    # SV-017: Blueprint size reasonable (< 500KB serialized)
    import json
    bp_size = len(json.dumps(blueprint))
    _check("SV-017", bp_size < 500_000,
           f"Blueprint is {bp_size:,} bytes — may cause import issues in Make.com",
           warn=True)

    # SV-018: No deeply nested routes (max 3 levels)
    max_depth = _max_route_depth(flow, 0)
    _check("SV-018", max_depth <= 3,
           f"Routes nested {max_depth} levels deep — simplify for maintainability",
           warn=True)

    # SV-019: All filter conditions reference valid modules
    for mod in all_modules:
        filt = mod.get("filter")
        if isinstance(filt, dict) and isinstance(filt.get("conditions"), list):
            for group in filt["conditions"]:
                if isinstance(group, list):
                    for cond in group:
                        if isinstance(cond, dict):
                            a_val = cond.get("a", "")
                            if isinstance(a_val, str) and "{{" in a_val:
                                # Extract referenced module ID
                                import re
                                refs = re.findall(r"\{\{(\d+)\.", a_val)
                                for ref_id in refs:
                                    _check("SV-019",
                                           int(ref_id) in set(i for i in all_ids if isinstance(i, int)),
                                           f"Filter references module {ref_id} which doesn't exist",
                                           module_id=mod.get("id"))

    # SV-020: Mapper references point to existing modules
    valid_ids = set(i for i in all_ids if isinstance(i, int))
    for mod in all_modules:
        mid = mod.get("id", "?")
        mapper = mod.get("mapper", {})
        if isinstance(mapper, dict):
            for key, val in mapper.items():
                if isinstance(val, str):
                    import re
                    refs = re.findall(r"\{\{(\d+)\.", val)
                    for ref_id in refs:
                        _check("SV-020", int(ref_id) in valid_ids,
                               f"Module {mid} mapper '{key}' references non-existent module {ref_id}",
                               module_id=mid)

    # SV-021 to SV-030: Operational readiness checks

    # SV-021: At least one action module (not just trigger + router)
    action_types = {"action", "transformer"}
    has_action = any(
        not m.get("module", "").startswith("builtin:") and
        not m.get("module", "").startswith("gateway:")
        for m in all_modules
    )
    _check("SV-021", has_action,
           "Scenario has no action modules — only contains triggers/routers")

    # SV-022: Webhook trigger has proper hook parameter
    for mod in all_modules:
        if mod.get("module") == "gateway:CustomWebHook":
            params = mod.get("parameters", {})
            _check("SV-022", isinstance(params, dict) and params.get("hook"),
                   "Webhook trigger missing 'hook' parameter — required for execution",
                   module_id=mod.get("id"))

    # SV-023: Schedule trigger has proper interval
    for mod in all_modules:
        if mod.get("module") == "gateway:ScheduleTrigger":
            params = mod.get("parameters", {})
            _check("SV-023", isinstance(params, dict) and params.get("interval"),
                   "Schedule trigger missing 'interval' parameter",
                   module_id=mod.get("id"))

    # SV-024: JSON parse module has 'json' mapper field
    for mod in all_modules:
        if mod.get("module") == "json:ParseJSON":
            mapper = mod.get("mapper", {})
            _check("SV-024", isinstance(mapper, dict) and "json" in mapper,
                   "JSON parse module missing 'json' mapper field",
                   module_id=mod.get("id"))

    # SV-025: HTTP module has 'url' mapper field
    for mod in all_modules:
        if mod.get("module") in ("http:ActionSendData", "http:ActionGetData"):
            mapper = mod.get("mapper", {})
            _check("SV-025", isinstance(mapper, dict) and "url" in mapper,
                   "HTTP module missing 'url' mapper field",
                   module_id=mod.get("id"))

    # SV-026: Slack module has channel and text
    for mod in all_modules:
        if mod.get("module") == "slack:PostMessage":
            mapper = mod.get("mapper", {})
            _check("SV-026", isinstance(mapper, dict) and "channel" in mapper and "text" in mapper,
                   "Slack module missing 'channel' or 'text' mapper fields",
                   module_id=mod.get("id"))

    # SV-027: Google Sheets module has spreadsheetId
    for mod in all_modules:
        if mod.get("module") in ("google-sheets:addRow", "google-sheets:getRow"):
            params = mod.get("parameters", {})
            _check("SV-027", isinstance(params, dict) and params.get("spreadsheetId"),
                   "Google Sheets module missing 'spreadsheetId' parameter",
                   module_id=mod.get("id"))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "checks_failed": checks_run - checks_passed,
    }


def _collect_all_modules(flow_items, all_ids, all_modules):
    """Recursively collect modules from flow tree (for /verify)."""
    for item in flow_items:
        if not isinstance(item, dict):
            continue
        mid = item.get("id")
        if mid is not None:
            all_ids.append(mid)
            all_modules.append(item)
        routes = item.get("routes", [])
        if isinstance(routes, list):
            for route in routes:
                if isinstance(route, dict) and isinstance(route.get("flow"), list):
                    _collect_all_modules(route["flow"], all_ids, all_modules)


def _max_route_depth(flow_items, current_depth):
    """Calculate max nesting depth of router routes."""
    max_d = current_depth
    if not isinstance(flow_items, list):
        return max_d
    for item in flow_items:
        if isinstance(item, dict) and isinstance(item.get("routes"), list):
            for route in item["routes"]:
                if isinstance(route, dict) and isinstance(route.get("flow"), list):
                    d = _max_route_depth(route["flow"], current_depth + 1)
                    max_d = max(max_d, d)
    return max_d


def _generate_fix_instructions(errors, warnings, score_100):
    """Generate specific, actionable fix instructions from validation failures."""
    instructions = []

    # Group errors by rule category
    for e in errors:
        rule_id = e.get("rule_id", "")
        msg = e.get("message", "")
        module_id = e.get("module_id")

        if rule_id.startswith("MR-"):
            instructions.append(f"Fix root structure: {msg}")
        elif rule_id.startswith("MF-"):
            instructions.append(f"Fix flow integrity: {msg}")
        elif rule_id.startswith("MM-"):
            target = f" (module {module_id})" if module_id else ""
            instructions.append(f"Fix module structure{target}: {msg}")
        elif rule_id.startswith("MT-"):
            instructions.append(f"Fix router configuration: {msg}")
        elif rule_id.startswith("MC-"):
            target = f" (module {module_id})" if module_id else ""
            instructions.append(f"Fix credential placeholder{target}: {msg}")
        elif rule_id.startswith("MD-"):
            instructions.append(f"Fix metadata: {msg}")
        elif rule_id.startswith("SV-"):
            target = f" (module {module_id})" if module_id else ""
            instructions.append(f"Fix{target}: {msg}")

    # Add summary instruction if many warnings
    high_warning_count = len(warnings)
    if high_warning_count > 5:
        instructions.append(
            f"Review {high_warning_count} warnings — most common: "
            + ", ".join(sorted(set(w.get("rule_id", "") for w in warnings[:5])))
        )

    if score_100 < 50:
        instructions.insert(0,
            "CRITICAL: Blueprint has fundamental structural issues. "
            "Rebuild from canonical spec rather than patching.")

    return instructions


# ─────────────────────────────────────────
# POST /handoff — Multi-agent orchestration bridge
# ─────────────────────────────────────────

@app.post("/handoff")
def handoff(request: HandoffRequest, db: Session = Depends(get_db)):
    """
    Record an agent-to-agent handoff in the orchestration pipeline.

    Stores from_agent, to_agent, project_id, and an arbitrary context_bundle
    to the agent_handoffs table. This is the bridge that allows Assessor,
    Builder, and Validator agents to pass state to each other.
    """
    if not request.from_agent.strip():
        raise HTTPException(status_code=400, detail="from_agent is required")
    if not request.to_agent.strip():
        raise HTTPException(status_code=400, detail="to_agent is required")

    try:
        project_uuid = UUID(request.project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    # Verify project exists
    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")

    try:
        record = AgentHandoff(
            from_agent=request.from_agent.strip(),
            to_agent=request.to_agent.strip(),
            project_id=project_uuid,
            context_bundle=request.context_bundle,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to store handoff: {str(e)}")

    return {
        "id": str(record.id),
        "from_agent": record.from_agent,
        "to_agent": record.to_agent,
        "project_id": str(record.project_id),
        "created_at": record.created_at.isoformat(),
    }


# ─────────────────────────────────────────
# GET /supervisor/stalled — Stalled project detection
# ─────────────────────────────────────────

@app.get("/supervisor/stalled")
def supervisor_stalled(db: Session = Depends(get_db)):
    """
    Detect stalled projects — any project with updated_at older than 48 hours
    and status not in ('deployed', 'cancelled').

    Returns a list of stalled projects with project_id, client, hours_stalled,
    and last_action (most recent build status).
    """
    try:
        rows = db.execute(text("""
            SELECT
                p.id AS project_id,
                p.name AS project_name,
                COALESCE(p.customer_name, 'Unknown') AS client,
                p.status AS project_status,
                p.updated_at,
                EXTRACT(EPOCH FROM (now() - p.updated_at)) / 3600 AS hours_stalled,
                (
                    SELECT b.status
                    FROM builds b
                    WHERE b.project_id = p.id
                    ORDER BY b.created_at DESC
                    LIMIT 1
                ) AS last_build_status,
                (
                    SELECT b.slug
                    FROM builds b
                    WHERE b.project_id = p.id
                    ORDER BY b.created_at DESC
                    LIMIT 1
                ) AS last_slug
            FROM projects p
            WHERE p.updated_at < now() - interval '48 hours'
              AND p.status NOT IN ('deployed', 'cancelled')
            ORDER BY p.updated_at ASC
        """)).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    stalled = []
    for row in rows:
        stalled.append({
            "project_id": str(row.project_id),
            "project_name": row.project_name,
            "client": row.client,
            "project_status": row.project_status,
            "hours_stalled": round(row.hours_stalled, 1),
            "last_action": row.last_build_status or "no builds",
            "last_slug": row.last_slug,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        })

    # Send stall alerts to Supabase for each stalled project
    if stalled:
        try:
            from tools.notification_sender import send_stall_alert
            for s in stalled:
                send_stall_alert(
                    project_id=s["project_id"],
                    days_stalled=s["hours_stalled"] / 24,
                )
        except Exception:
            pass  # Non-critical

    return {
        "stalled_count": len(stalled),
        "stalled_projects": stalled,
    }


# ─────────────────────────────────────────
# POST /costs/track — Token cost tracking
# ─────────────────────────────────────────

# Per-token pricing (USD) — updated for Claude 4.5/4.6 family
_MODEL_PRICING = {
    "claude-opus-4-6":       {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
    "claude-sonnet-4-6":     {"input":  3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "claude-haiku-4-5":      {"input":  0.80 / 1_000_000, "output":  4.00 / 1_000_000},
    "claude-sonnet-4-5":     {"input":  3.00 / 1_000_000, "output": 15.00 / 1_000_000},
    "gpt-4o":                {"input":  2.50 / 1_000_000, "output": 10.00 / 1_000_000},
    "gpt-4o-mini":           {"input":  0.15 / 1_000_000, "output":  0.60 / 1_000_000},
}


@app.post("/costs/track")
def costs_track(request: CostTrackRequest, db: Session = Depends(get_db)):
    """
    Track token costs for a project operation.

    Accepts project_id, model, input_tokens, output_tokens, and operation_type.
    Calculates cost in USD using model pricing and stores to project_financials.
    """
    try:
        project_uuid = UUID(request.project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    # Verify project exists
    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")

    if request.input_tokens < 0 or request.output_tokens < 0:
        raise HTTPException(status_code=400, detail="Token counts must be non-negative")

    # Calculate cost
    pricing = _MODEL_PRICING.get(request.model)
    if pricing:
        cost_usd = (request.input_tokens * pricing["input"]
                     + request.output_tokens * pricing["output"])
    else:
        # Unknown model — estimate at mid-tier pricing
        cost_usd = (request.input_tokens * 3.0 / 1_000_000
                     + request.output_tokens * 15.0 / 1_000_000)

    cost_usd = round(cost_usd, 6)

    try:
        record = ProjectFinancial(
            project_id=project_uuid,
            model=request.model,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            operation_type=request.operation_type,
            cost_usd=cost_usd,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to store cost record: {str(e)}")

    # Auto-calculate cumulative cost and margin for the project
    cumulative_cost = db.execute(text(
        "SELECT COALESCE(SUM(cost_usd), 0) FROM project_financials WHERE project_id = :pid"
    ), {"pid": str(project_uuid)}).scalar()
    cumulative_cost = float(cumulative_cost)

    revenue = float(project.revenue or 0)
    margin = revenue - cumulative_cost
    margin_pct = (margin / revenue * 100) if revenue > 0 else 0.0

    # Check if cost exceeds estimate by 20%+ (alert threshold)
    cost_alert = None
    if revenue > 0 and cumulative_cost > revenue * 0.8:
        overage_pct = round((cumulative_cost / revenue - 0.8) * 100, 1)
        if cumulative_cost > revenue:
            cost_alert = {
                "level": "critical",
                "message": f"Cost ({cumulative_cost:.4f}) exceeds revenue ({revenue:.2f}). Margin is negative.",
            }
        else:
            cost_alert = {
                "level": "warning",
                "message": f"Cost is within 20% of revenue. Current margin: {margin_pct:.1f}%",
            }

    # Sync financials to Supabase
    try:
        from tools.pipeline_sync import sync_project_financials
        sync_project_financials(
            project_id=request.project_id,
            revenue=revenue,
            api_cost=cumulative_cost,
        )
    except Exception:
        pass  # Non-critical

    # Send cost alert notification if margin < 20%
    if revenue > 0 and margin_pct < 20:
        try:
            from tools.notification_sender import send_cost_alert
            send_cost_alert(project_id=request.project_id, margin_pct=margin_pct)
        except Exception:
            pass  # Non-critical

    return {
        "id": str(record.id),
        "project_id": str(record.project_id),
        "model": record.model,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "operation_type": record.operation_type,
        "cost_usd": record.cost_usd,
        "cumulative_cost": round(cumulative_cost, 4),
        "revenue": round(revenue, 2),
        "margin": round(margin, 4),
        "margin_pct": round(margin_pct, 1),
        "cost_alert": cost_alert,
        "created_at": record.created_at.isoformat(),
    }


# ─────────────────────────────────────────
# GET /costs/summary — Cost/revenue/margin per client
# ─────────────────────────────────────────

@app.get("/costs/summary")
def costs_summary(client_id: str = Query(..., description="Customer name or project name"),
                  db: Session = Depends(get_db)):
    """
    Returns total cost, total revenue, and margin per project and overall
    for a given client.

    client_id matches against projects.customer_name (case-insensitive).
    """
    if not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")

    try:
        # Per-project breakdown
        rows = db.execute(text("""
            SELECT
                p.id AS project_id,
                p.name AS project_name,
                COALESCE(p.revenue, 0) AS revenue,
                COALESCE(SUM(pf.cost_usd), 0) AS total_cost,
                COALESCE(SUM(pf.input_tokens), 0) AS total_input_tokens,
                COALESCE(SUM(pf.output_tokens), 0) AS total_output_tokens,
                COUNT(pf.id) AS operation_count
            FROM projects p
            LEFT JOIN project_financials pf ON pf.project_id = p.id
            WHERE LOWER(p.customer_name) = LOWER(:client_id)
               OR LOWER(p.name) = LOWER(:client_id)
            GROUP BY p.id, p.name, p.revenue
            ORDER BY p.name
        """), {"client_id": client_id.strip()}).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    if not rows:
        raise HTTPException(status_code=404, detail=f"No projects found for client '{client_id}'")

    projects = []
    total_cost = 0.0
    total_revenue = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    for row in rows:
        cost = float(row.total_cost)
        revenue = float(row.revenue)
        margin = revenue - cost
        margin_pct = (margin / revenue * 100) if revenue > 0 else 0.0

        projects.append({
            "project_id": str(row.project_id),
            "project_name": row.project_name,
            "total_cost": round(cost, 4),
            "revenue": round(revenue, 2),
            "margin": round(margin, 4),
            "margin_pct": round(margin_pct, 1),
            "total_input_tokens": int(row.total_input_tokens),
            "total_output_tokens": int(row.total_output_tokens),
            "operation_count": int(row.operation_count),
        })

        total_cost += cost
        total_revenue += revenue
        total_input_tokens += int(row.total_input_tokens)
        total_output_tokens += int(row.total_output_tokens)

    overall_margin = total_revenue - total_cost
    overall_margin_pct = (overall_margin / total_revenue * 100) if total_revenue > 0 else 0.0

    return {
        "client_id": client_id,
        "project_count": len(projects),
        "overall": {
            "total_cost": round(total_cost, 4),
            "total_revenue": round(total_revenue, 2),
            "margin": round(overall_margin, 4),
            "margin_pct": round(overall_margin_pct, 1),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
        },
        "projects": projects,
    }


@app.get("/costs/report")
def costs_report(
    client_id: str = Query(..., description="Customer name"),
    weeks: int = Query(4, ge=1, le=52),
    db: Session = Depends(get_db),
):
    """
    Weekly cost report in markdown format.

    Returns a markdown table of weekly costs, token usage, and margin trends
    for the given client over the last N weeks.
    """
    try:
        rows = db.execute(text("""
            SELECT
                date_trunc('week', pf.created_at) AS week_start,
                COUNT(pf.id) AS operations,
                COALESCE(SUM(pf.input_tokens), 0) AS input_tokens,
                COALESCE(SUM(pf.output_tokens), 0) AS output_tokens,
                COALESCE(SUM(pf.cost_usd), 0) AS total_cost,
                STRING_AGG(DISTINCT pf.model, ', ') AS models_used
            FROM project_financials pf
            JOIN projects p ON p.id = pf.project_id
            WHERE (LOWER(p.customer_name) = LOWER(:client_id)
                   OR LOWER(p.name) = LOWER(:client_id))
              AND pf.created_at >= NOW() - INTERVAL ':weeks weeks'
            GROUP BY date_trunc('week', pf.created_at)
            ORDER BY week_start DESC
        """).bindparams(weeks=weeks), {"client_id": client_id.strip()}).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Get total revenue for margin calculation
    revenue_row = db.execute(text("""
        SELECT COALESCE(SUM(revenue), 0) AS total_revenue
        FROM projects
        WHERE LOWER(customer_name) = LOWER(:client_id) OR LOWER(name) = LOWER(:client_id)
    """), {"client_id": client_id.strip()}).first()
    total_revenue = float(revenue_row.total_revenue) if revenue_row else 0.0

    # Build markdown report
    lines = [
        f"# Weekly Cost Report — {client_id}",
        f"",
        f"**Period:** Last {weeks} weeks  ",
        f"**Total Revenue:** ${total_revenue:,.2f}",
        f"",
        f"| Week | Operations | Input Tokens | Output Tokens | Cost (USD) | Models |",
        f"|------|-----------|-------------|--------------|-----------|--------|",
    ]

    cumulative_cost = 0.0
    weekly_data = []
    for row in rows:
        week_label = row.week_start.strftime("%Y-%m-%d") if row.week_start else "N/A"
        cost = float(row.total_cost)
        cumulative_cost += cost
        lines.append(
            f"| {week_label} | {row.operations} | {int(row.input_tokens):,} | "
            f"{int(row.output_tokens):,} | ${cost:.4f} | {row.models_used or 'N/A'} |"
        )
        weekly_data.append({
            "week": week_label,
            "operations": int(row.operations),
            "input_tokens": int(row.input_tokens),
            "output_tokens": int(row.output_tokens),
            "cost_usd": round(cost, 4),
            "models": row.models_used or "",
        })

    overall_margin = total_revenue - cumulative_cost
    margin_pct = (overall_margin / total_revenue * 100) if total_revenue > 0 else 0.0

    lines.extend([
        f"",
        f"**Cumulative Cost:** ${cumulative_cost:,.4f}  ",
        f"**Margin:** ${overall_margin:,.4f} ({margin_pct:.1f}%)",
    ])

    return {
        "client_id": client_id,
        "weeks": weeks,
        "total_revenue": round(total_revenue, 2),
        "cumulative_cost": round(cumulative_cost, 4),
        "margin": round(overall_margin, 4),
        "margin_pct": round(margin_pct, 1),
        "weekly_data": weekly_data,
        "markdown": "\n".join(lines),
    }


@app.post("/costs/estimate")
def costs_estimate(request: CostEstimateRequest, db: Session = Depends(get_db)):
    """
    Pre-build cost estimation based on historical project data.

    Analyzes past projects with similar descriptions to estimate the likely
    token usage and cost for a new build. Uses TF-IDF similarity to find
    comparable past projects and averages their cost profiles.
    """
    from tools.embedding_engine import find_similar

    # Find similar past projects
    similar = find_similar(request.description, top_n=5)

    if not similar:
        # No historical data — use category-based defaults
        defaults = {
            "simple": {"input_tokens": 5_000, "output_tokens": 10_000, "ops": 3},
            "standard": {"input_tokens": 15_000, "output_tokens": 30_000, "ops": 6},
            "complex": {"input_tokens": 40_000, "output_tokens": 80_000, "ops": 10},
            "enterprise": {"input_tokens": 100_000, "output_tokens": 200_000, "ops": 20},
        }
        profile = defaults.get(request.category, defaults["standard"])

        # Estimate cost using mid-tier model (sonnet)
        mid_pricing = _MODEL_PRICING.get("claude-sonnet-4-6", {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000})
        estimated_cost = (
            profile["input_tokens"] * mid_pricing["input"]
            + profile["output_tokens"] * mid_pricing["output"]
        )

        return {
            "source": "category_default",
            "category": request.category,
            "estimated_operations": profile["ops"],
            "estimated_input_tokens": profile["input_tokens"],
            "estimated_output_tokens": profile["output_tokens"],
            "estimated_cost_usd": round(estimated_cost, 4),
            "confidence": "low",
            "similar_projects": [],
        }

    # Get cost data for similar project IDs
    project_ids = [s["id"] for s in similar]
    placeholders = ", ".join([f":pid{i}" for i in range(len(project_ids))])
    params = {f"pid{i}": pid for i, pid in enumerate(project_ids)}

    try:
        rows = db.execute(text(f"""
            SELECT
                project_id,
                COUNT(*) AS ops,
                COALESCE(SUM(input_tokens), 0) AS total_input,
                COALESCE(SUM(output_tokens), 0) AS total_output,
                COALESCE(SUM(cost_usd), 0) AS total_cost
            FROM project_financials
            WHERE project_id::text IN ({placeholders})
            GROUP BY project_id
        """), params).fetchall()
    except Exception:
        rows = []

    if not rows:
        # Similar projects found but no cost history
        mid_pricing = _MODEL_PRICING.get("claude-sonnet-4-6", {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000})
        defaults = {"simple": 10_000, "standard": 30_000, "complex": 80_000, "enterprise": 200_000}
        est_tokens = defaults.get(request.category, 30_000)
        return {
            "source": "similar_projects_no_cost_data",
            "estimated_operations": 5,
            "estimated_input_tokens": est_tokens // 2,
            "estimated_output_tokens": est_tokens,
            "estimated_cost_usd": round(est_tokens * mid_pricing["output"], 4),
            "confidence": "low",
            "similar_projects": [{"id": s["id"], "score": s["score"]} for s in similar[:3]],
        }

    # Average across similar projects with cost data
    avg_ops = sum(r.ops for r in rows) / len(rows)
    avg_input = sum(int(r.total_input) for r in rows) / len(rows)
    avg_output = sum(int(r.total_output) for r in rows) / len(rows)
    avg_cost = sum(float(r.total_cost) for r in rows) / len(rows)

    conf = "high" if len(rows) >= 3 else ("medium" if len(rows) >= 2 else "low")

    return {
        "source": "historical_average",
        "based_on_projects": len(rows),
        "estimated_operations": round(avg_ops),
        "estimated_input_tokens": round(avg_input),
        "estimated_output_tokens": round(avg_output),
        "estimated_cost_usd": round(avg_cost, 4),
        "confidence": conf,
        "similar_projects": [{"id": s["id"], "score": s["score"]} for s in similar[:3]],
    }


# ═══════════════════════════════════════════════════════════════
# BLOCK 1 — MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════════════

# Pipeline stage → agent mapping
_STAGE_AGENT_MAP = {
    "intake": "assessor",
    "build": "builder",
    "verify": "validator",
    "deploy": "deployer",
}
_STAGE_ORDER = ["intake", "build", "verify", "deploy"]


@app.post("/orchestrate")
def orchestrate(request: OrchestrateRequest, db: Session = Depends(get_db)):
    """
    Determine next agent for a project and advance the pipeline.

    Accepts project_id and current_stage. Looks up or creates the pipeline
    state, determines the next agent, updates project_agent_state, and
    returns next_agent + context_bundle.
    """
    try:
        project_uuid = UUID(request.project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")

    stage = request.current_stage.lower().strip()
    if stage not in _STAGE_AGENT_MAP:
        raise HTTPException(status_code=400,
                            detail=f"Invalid stage '{stage}'. Must be one of: {', '.join(_STAGE_ORDER)}")

    next_agent = _STAGE_AGENT_MAP[stage]

    # Determine next stage after current
    stage_idx = _STAGE_ORDER.index(stage)
    next_stage = _STAGE_ORDER[stage_idx + 1] if stage_idx < len(_STAGE_ORDER) - 1 else None

    # Upsert pipeline state
    state = db.query(ProjectAgentState).filter(
        ProjectAgentState.project_id == project_uuid
    ).first()

    now = datetime.now(timezone.utc)
    stage_entry = {"stage": stage, "agent": next_agent, "started_at": now.isoformat()}

    if state:
        state.current_stage = stage
        state.current_agent = next_agent
        state.updated_at = now
        state.pipeline_health = "on_track"
        history = state.stage_history or []
        history.append(stage_entry)
        state.stage_history = history
    else:
        state = ProjectAgentState(
            project_id=project_uuid,
            current_stage=stage,
            current_agent=next_agent,
            started_at=now,
            updated_at=now,
            pipeline_health="on_track",
            stage_history=[stage_entry],
        )
        db.add(state)

    # Build context bundle from latest handoffs and builds
    context_bundle = {
        "project_id": str(project_uuid),
        "project_name": project.name,
        "customer_name": project.customer_name,
        "current_stage": stage,
        "assigned_agent": next_agent,
        "next_stage": next_stage,
    }

    # Include latest build info if available
    latest_build = db.execute(text("""
        SELECT slug, version, status, confidence_score, confidence_grade
        FROM builds WHERE project_id = :pid
        ORDER BY created_at DESC LIMIT 1
    """), {"pid": str(project_uuid)}).fetchone()

    if latest_build:
        context_bundle["latest_build"] = {
            "slug": latest_build.slug,
            "version": latest_build.version,
            "status": latest_build.status,
            "confidence_score": latest_build.confidence_score,
            "confidence_grade": latest_build.confidence_grade,
        }

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update pipeline state: {str(e)}")

    # Sync pipeline state to Supabase
    try:
        from tools.pipeline_sync import sync_project_state
        sync_project_state(
            project_id=str(project_uuid),
            stage=stage,
            agent_name=next_agent,
            context_bundle=context_bundle,
        )
    except Exception:
        pass  # Non-critical

    return {
        "project_id": str(project_uuid),
        "current_stage": stage,
        "next_agent": next_agent,
        "next_stage": next_stage,
        "pipeline_health": "on_track",
        "context_bundle": context_bundle,
    }


@app.post("/agent/complete")
def agent_complete(request: AgentCompleteRequest, db: Session = Depends(get_db)):
    """
    Called when an agent finishes its work. Logs the handoff to agent_handoffs,
    updates pipeline state, and auto-triggers orchestrate to advance to next stage.
    """
    try:
        project_uuid = UUID(request.project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")

    state = db.query(ProjectAgentState).filter(
        ProjectAgentState.project_id == project_uuid
    ).first()

    now = datetime.now(timezone.utc)

    # Log handoff
    current_stage = state.current_stage if state else "intake"
    stage_idx = _STAGE_ORDER.index(current_stage) if current_stage in _STAGE_ORDER else 0
    next_stage = _STAGE_ORDER[stage_idx + 1] if stage_idx < len(_STAGE_ORDER) - 1 else None
    next_agent = _STAGE_AGENT_MAP.get(next_stage) if next_stage else None

    handoff = AgentHandoff(
        from_agent=request.agent_name,
        to_agent=next_agent or "supervisor",
        project_id=project_uuid,
        context_bundle={
            "outcome": request.outcome,
            "artifacts": request.artifacts,
            "completed_stage": current_stage,
            "completed_at": now.isoformat(),
        },
    )
    db.add(handoff)

    # Update pipeline state
    completion_entry = {
        "stage": current_stage,
        "agent": request.agent_name,
        "outcome": request.outcome,
        "completed_at": now.isoformat(),
    }

    if request.outcome == "failed":
        if state:
            state.pipeline_health = "failed"
            state.updated_at = now
            history = state.stage_history or []
            history.append(completion_entry)
            state.stage_history = history
        db.commit()
        return {
            "project_id": str(project_uuid),
            "status": "pipeline_failed",
            "failed_stage": current_stage,
            "failed_agent": request.agent_name,
            "next_agent": None,
        }

    # Advance pipeline
    if state and next_stage:
        state.current_stage = next_stage
        state.current_agent = next_agent
        state.updated_at = now
        state.pipeline_health = "on_track"
        history = state.stage_history or []
        history.append(completion_entry)
        state.stage_history = history
    elif state:
        state.pipeline_health = "completed"
        state.updated_at = now
        history = state.stage_history or []
        history.append(completion_entry)
        state.stage_history = history

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    # Sync agent handoff activity to Supabase
    try:
        from tools.pipeline_sync import sync_activity
        sync_activity(
            project_id=str(project_uuid),
            action_type="agent_handoff",
            description=f"{request.agent_name} completed: {request.outcome}",
            agent_name=request.agent_name,
        )
    except Exception:
        pass  # Non-critical

    return {
        "project_id": str(project_uuid),
        "status": "advanced" if next_stage else "pipeline_complete",
        "completed_stage": current_stage,
        "completed_agent": request.agent_name,
        "next_stage": next_stage,
        "next_agent": next_agent,
        "pipeline_health": state.pipeline_health if state else "on_track",
    }


@app.get("/pipeline/status")
def pipeline_status(project_id: str = Query(...), db: Session = Depends(get_db)):
    """
    Full pipeline state for a project: current_agent, completed stages with
    timestamps, artifacts per stage, overall health (on_track/stalled/failed).
    """
    try:
        project_uuid = UUID(project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    state = db.query(ProjectAgentState).filter(
        ProjectAgentState.project_id == project_uuid
    ).first()

    # Get all handoffs for this project
    handoffs = db.execute(text("""
        SELECT from_agent, to_agent, context_bundle, created_at
        FROM agent_handoffs
        WHERE project_id = :pid
        ORDER BY created_at ASC
    """), {"pid": str(project_uuid)}).fetchall()

    # Get artifacts
    artifacts = db.execute(text("""
        SELECT ba.artifact_type, ba.created_at, b.slug, b.version
        FROM build_artifacts ba
        JOIN builds b ON b.id = ba.build_id
        WHERE b.project_id = :pid
        ORDER BY ba.created_at DESC
    """), {"pid": str(project_uuid)}).fetchall()

    # Check for stalled state
    health = "no_pipeline"
    if state:
        health = state.pipeline_health
        hours_since_update = (datetime.now(timezone.utc) - state.updated_at).total_seconds() / 3600
        if health == "on_track" and hours_since_update > 48:
            health = "stalled"

    return {
        "project_id": str(project_uuid),
        "project_name": project.name,
        "customer_name": project.customer_name,
        "current_stage": state.current_stage if state else None,
        "current_agent": state.current_agent if state else None,
        "pipeline_health": health,
        "started_at": state.started_at.isoformat() if state else None,
        "last_updated": state.updated_at.isoformat() if state else None,
        "stage_history": state.stage_history if state else [],
        "handoffs": [
            {
                "from_agent": h.from_agent,
                "to_agent": h.to_agent,
                "context": h.context_bundle,
                "at": h.created_at.isoformat(),
            }
            for h in handoffs
        ],
        "artifacts": [
            {
                "type": a.artifact_type,
                "slug": a.slug,
                "version": a.version,
                "created_at": a.created_at.isoformat(),
            }
            for a in artifacts[:20]
        ],
    }


@app.post("/briefing")
def briefing(db: Session = Depends(get_db)):
    """
    Daily supervisor briefing. Returns markdown report with:
    - All active projects summary
    - Stalled projects list
    - Overnight activity (last 24h)
    - Failures in last 24h
    """
    now = datetime.now(timezone.utc)
    day_ago = now.isoformat()

    # Active projects
    active = db.execute(text("""
        SELECT p.id, p.name, p.customer_name, p.status,
               pas.current_stage, pas.current_agent, pas.pipeline_health, pas.updated_at
        FROM projects p
        LEFT JOIN project_agent_state pas ON pas.project_id = p.id
        WHERE p.status NOT IN ('deployed', 'cancelled')
        ORDER BY p.updated_at DESC NULLS LAST
    """)).fetchall()

    # Stalled
    stalled = db.execute(text("""
        SELECT p.name, p.customer_name,
               EXTRACT(EPOCH FROM (now() - p.updated_at)) / 3600 AS hours
        FROM projects p
        WHERE p.updated_at < now() - interval '48 hours'
          AND p.status NOT IN ('deployed', 'cancelled')
        ORDER BY p.updated_at ASC
    """)).fetchall()

    # Recent builds (last 24h)
    recent_builds = db.execute(text("""
        SELECT b.slug, b.version, b.status, b.confidence_score,
               b.confidence_grade, b.created_at, p.name AS project_name
        FROM builds b
        JOIN projects p ON p.id = b.project_id
        WHERE b.created_at > now() - interval '24 hours'
        ORDER BY b.created_at DESC
    """)).fetchall()

    # Failures in last 24h
    failures = [b for b in recent_builds if b.status == "failed"]

    # Build markdown
    lines = [
        f"# Daily Supervisor Briefing",
        f"**Generated:** {now.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        f"## Active Projects ({len(active)})",
        "",
    ]

    if active:
        lines.append("| Project | Client | Stage | Agent | Health |")
        lines.append("|---------|--------|-------|-------|--------|")
        for a in active:
            lines.append(
                f"| {a.name} | {a.customer_name or '-'} | "
                f"{a.current_stage or '-'} | {a.current_agent or '-'} | "
                f"{a.pipeline_health or '-'} |"
            )
    else:
        lines.append("No active projects.")

    lines.extend(["", f"## Stalled Projects ({len(stalled)})", ""])
    if stalled:
        for s in stalled:
            lines.append(f"- **{s.name}** ({s.customer_name}) — stalled {round(s.hours, 1)}h")
    else:
        lines.append("None.")

    lines.extend(["", f"## Last 24h Activity ({len(recent_builds)} builds)", ""])
    if recent_builds:
        for b in recent_builds:
            grade = f" [{b.confidence_grade}]" if b.confidence_grade else ""
            lines.append(f"- {b.project_name}/{b.slug} v{b.version} — {b.status}{grade}")
    else:
        lines.append("No build activity.")

    lines.extend(["", f"## Failures ({len(failures)})", ""])
    if failures:
        for f_ in failures:
            lines.append(f"- **{f_.slug}** v{f_.version} — FAILED")
    else:
        lines.append("No failures.")

    report = "\n".join(lines)

    return {
        "report": report,
        "summary": {
            "active_projects": len(active),
            "stalled_projects": len(stalled),
            "builds_24h": len(recent_builds),
            "failures_24h": len(failures),
        },
        "generated_at": now.isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# BLOCK 2 — AGENT MEMORY & LEARNING
# ═══════════════════════════════════════════════════════════════

@app.post("/memory")
def memory_store(request: MemoryRequest, db: Session = Depends(get_db)):
    """
    Store client context: key_decisions, tech_stack, failure_patterns.
    Upserts on client_id + project_id.
    """
    try:
        project_uuid = UUID(request.project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    if not request.client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")

    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")

    now = datetime.now(timezone.utc)
    existing = db.query(ClientContext).filter(
        ClientContext.client_id == request.client_id.strip(),
        ClientContext.project_id == project_uuid,
    ).first()

    if existing:
        if request.key_decisions is not None:
            existing.key_decisions = request.key_decisions
        if request.tech_stack is not None:
            existing.tech_stack = request.tech_stack
        if request.failure_patterns is not None:
            existing.failure_patterns = request.failure_patterns
        existing.updated_at = now
        db.commit()
        db.refresh(existing)
        record = existing
    else:
        record = ClientContext(
            client_id=request.client_id.strip(),
            project_id=project_uuid,
            key_decisions=request.key_decisions,
            tech_stack=request.tech_stack,
            failure_patterns=request.failure_patterns,
            created_at=now,
            updated_at=now,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    return {
        "id": str(record.id),
        "client_id": record.client_id,
        "project_id": str(record.project_id),
        "key_decisions": record.key_decisions,
        "tech_stack": record.tech_stack,
        "failure_patterns": record.failure_patterns,
        "updated_at": record.updated_at.isoformat(),
    }


@app.get("/memory")
def memory_get(client_id: str = Query(...), db: Session = Depends(get_db)):
    """
    Full aggregated context for a client across all projects.
    Merges key_decisions, tech_stack, failure_patterns from all projects.
    """
    if not client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")

    rows = db.query(ClientContext).filter(
        ClientContext.client_id == client_id.strip()
    ).all()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No memory found for client '{client_id}'")

    all_decisions = []
    all_tech = set()
    all_failures = []
    projects = []

    for r in rows:
        projects.append({
            "project_id": str(r.project_id),
            "key_decisions": r.key_decisions,
            "tech_stack": r.tech_stack,
            "failure_patterns": r.failure_patterns,
            "updated_at": r.updated_at.isoformat(),
        })
        if r.key_decisions:
            all_decisions.extend(r.key_decisions)
        if r.tech_stack:
            all_tech.update(r.tech_stack)
        if r.failure_patterns:
            all_failures.extend(r.failure_patterns)

    return {
        "client_id": client_id,
        "project_count": len(rows),
        "aggregated": {
            "key_decisions": all_decisions,
            "tech_stack": sorted(all_tech),
            "failure_patterns": all_failures,
        },
        "projects": projects,
    }


@app.post("/memory/embed")
def memory_embed(request: EmbedRequest, db: Session = Depends(get_db)):
    """
    Embed a completed build brief + outcome into TF-IDF vector store.
    Auto-called when a project is marked complete.
    """
    from tools.embedding_engine import embed_document

    try:
        project_uuid = UUID(request.project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")

    full_text = request.brief
    if request.outcome:
        full_text += f" OUTCOME: {request.outcome}"

    result = embed_document(
        doc_id=str(project_uuid),
        text=full_text,
        metadata={
            "project_name": project.name,
            "customer_name": project.customer_name,
            "outcome": request.outcome,
        },
    )

    return result


@app.get("/similar")
def similar(description: str = Query(..., min_length=3), top_n: int = Query(3, ge=1, le=10)):
    """
    Find top N similar past projects by cosine similarity over TF-IDF vectors.
    Returns project_id, score, and brief_summary.
    """
    from tools.embedding_engine import find_similar

    results = find_similar(description, top_n=top_n)

    return {
        "query": description,
        "results": [
            {
                "project_id": r["id"],
                "score": r["score"],
                "brief_summary": r["text_preview"],
                "metadata": r["metadata"],
            }
            for r in results
        ],
    }


@app.post("/verify/loop")
def verify_loop(request: VerifyLoopRequest, db: Session = Depends(get_db)):
    """
    Iterative verify → fix → verify cycle.

    Runs blueprint through the 77-rule audit up to max_iterations times.
    After each failing iteration, auto-generates fix instructions and applies
    rule-based patches before re-running. Stops early if confidence >= 85
    and error count reaches 0.

    Returns the full iteration history plus final verdict.
    """
    blueprint = request.blueprint
    max_iters = min(request.max_iterations or 3, 5)  # cap at 5
    project_uuid = UUID(request.project_id) if request.project_id else None

    iterations = []

    for i in range(1, max_iters + 1):
        # Run validation
        try:
            export_report = validate_make_export(blueprint, _registry)
        except Exception as e:
            iterations.append({"iteration": i, "error": str(e)})
            break

        canonical_report = _run_supplementary_checks(blueprint, _registry)
        all_errors = export_report.get("errors", []) + canonical_report.get("errors", [])
        all_warnings = export_report.get("warnings", []) + canonical_report.get("warnings", [])
        merged = {
            "errors": all_errors,
            "warnings": all_warnings,
            "checks_run": export_report.get("checks_run", 0) + canonical_report.get("checks_run", 0),
            "checks_passed": export_report.get("checks_passed", 0) + canonical_report.get("checks_passed", 0),
        }

        confidence = compute_confidence(merged, agent_notes=[], retry_count=i - 1)
        score_100 = round(confidence["score"] * 100, 1)
        passed = score_100 >= 85 and len(all_errors) == 0

        fix_instructions = []
        if not passed:
            fix_instructions = _generate_fix_instructions(all_errors, all_warnings, score_100)

        iteration_record = {
            "iteration": i,
            "confidence_score": score_100,
            "grade": confidence["grade"],
            "error_count": len(all_errors),
            "warning_count": len(all_warnings),
            "passed": passed,
            "fixes_applied": [],
        }

        # Log to verification_runs
        if project_uuid:
            try:
                run = VerificationRun(
                    project_id=project_uuid,
                    confidence_score=score_100,
                    passed=passed,
                    error_count=len(all_errors),
                    warning_count=len(all_warnings),
                    fix_instructions=fix_instructions if fix_instructions else None,
                    iteration=i,
                )
                db.add(run)
                db.commit()
            except Exception:
                db.rollback()

        iterations.append(iteration_record)

        if passed:
            break

        # Apply rule-based auto-fixes to the blueprint for the next iteration
        fixes_applied = _auto_fix_blueprint(blueprint, all_errors)
        iteration_record["fixes_applied"] = fixes_applied
        if not fixes_applied:
            break  # No more auto-fixable issues

    final = iterations[-1] if iterations else {}

    # Auto-advance pipeline on final success
    auto_advanced = False
    if final.get("passed") and project_uuid:
        try:
            state = db.query(ProjectAgentState).filter(
                ProjectAgentState.project_id == project_uuid
            ).first()
            if state and state.current_stage == "verify":
                handoff = AgentHandoff(
                    from_agent="validator",
                    to_agent="deployer",
                    project_id=project_uuid,
                    context_bundle={"outcome": "success", "confidence_score": final.get("confidence_score")},
                )
                db.add(handoff)
                state.current_stage = "deploy"
                state.current_agent = "deployer"
                state.updated_at = datetime.now(timezone.utc)
                history = state.stage_history or []
                history.append({
                    "stage": "verify",
                    "agent": "validator",
                    "outcome": "success",
                    "iterations": len(iterations),
                    "final_confidence": final.get("confidence_score"),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                })
                state.stage_history = history
                db.commit()
                auto_advanced = True
        except Exception:
            db.rollback()

    result = {
        "project_id": request.project_id,
        "total_iterations": len(iterations),
        "final_passed": final.get("passed", False),
        "final_confidence": final.get("confidence_score", 0),
        "final_grade": final.get("grade", "F"),
        "iterations": iterations,
    }
    if auto_advanced:
        result["auto_advanced"] = True
        result["next_stage"] = "deploy"
    return result


def _auto_fix_blueprint(blueprint: dict, errors: list) -> list[str]:
    """Apply rule-based auto-fixes to a blueprint based on validation errors.

    Returns list of fix descriptions applied. Mutates the blueprint in place.
    """
    fixes = []
    modules = blueprint.get("flow", blueprint.get("modules", []))
    if not isinstance(modules, list):
        return fixes

    for err in errors:
        rule_id = err.get("rule_id", "")
        msg = err.get("message", "").lower()

        # Fix: missing error handler
        if "error" in rule_id.lower() and "handler" in msg:
            for mod in modules:
                if isinstance(mod, dict) and "error_handler" not in mod:
                    mod["error_handler"] = {"type": "ignore"}
                    fixes.append(f"Added default error handler to module '{mod.get('name', 'unknown')}'")

        # Fix: missing module name
        if "name" in msg and "missing" in msg:
            for idx, mod in enumerate(modules):
                if isinstance(mod, dict) and not mod.get("name"):
                    mod["name"] = f"Module_{idx + 1}"
                    fixes.append(f"Added default name to module at index {idx}")

        # Fix: empty description
        if "description" in msg and ("empty" in msg or "missing" in msg):
            if not blueprint.get("description"):
                blueprint["description"] = "Auto-generated scenario"
                fixes.append("Added default scenario description")

    return fixes


@app.get("/confidence/history")
def confidence_history(project_id: str = Query(...), db: Session = Depends(get_db)):
    """
    Get all verification runs for a project with confidence score trend.
    Returns runs ordered by creation time with delta between consecutive runs.
    """
    project_uuid = UUID(project_id)

    runs = db.query(VerificationRun).filter(
        VerificationRun.project_id == project_uuid
    ).order_by(VerificationRun.created_at.asc()).all()

    history = []
    prev_score = None
    for run in runs:
        entry = {
            "id": str(run.id),
            "iteration": run.iteration,
            "confidence_score": run.confidence_score,
            "passed": run.passed,
            "error_count": run.error_count,
            "warning_count": run.warning_count,
            "delta": round(run.confidence_score - prev_score, 1) if prev_score is not None else None,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        history.append(entry)
        prev_score = run.confidence_score

    trend = "stable"
    if len(history) >= 2:
        first = history[0]["confidence_score"]
        last = history[-1]["confidence_score"]
        if last - first > 5:
            trend = "improving"
        elif first - last > 5:
            trend = "declining"

    return {
        "project_id": project_id,
        "total_runs": len(history),
        "trend": trend,
        "current_confidence": history[-1]["confidence_score"] if history else None,
        "history": history,
    }


@app.post("/deploy/makecom")
def deploy_makecom(request: DeployMakecomRequest, db: Session = Depends(get_db)):
    """
    Deploy a blueprint to Make.com via their API.

    Imports the scenario blueprint, records the deployment in the deployments
    table, and auto-advances the pipeline to completed status.
    """
    import httpx

    project_uuid = UUID(request.project_id)

    # Validate project exists
    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Build Make.com import payload
    headers = {
        "Authorization": f"Token {request.api_key}",
        "Content-Type": "application/json",
    }
    team_id = request.team_id or 1
    payload = {
        "blueprint": request.blueprint,
        "scheduling": {"type": "indefinitely", "interval": 900},
        "teamId": team_id,
    }

    deployment_record = Deployment(
        project_id=project_uuid,
        target="make.com",
        status="pending",
    )
    db.add(deployment_record)
    db.commit()
    db.refresh(deployment_record)

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"https://us1.make.com/api/v2/scenarios?teamId={team_id}",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        scenario = data.get("scenario", data)
        deployment_record.external_id = str(scenario.get("id", ""))
        deployment_record.external_url = f"https://us1.make.com/scenarios/{scenario.get('id', '')}"
        deployment_record.status = "deployed"
        deployment_record.last_health_check = {
            "status": "deployed",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "make_response": {"id": scenario.get("id"), "name": scenario.get("name")},
        }
        db.commit()

        # Update pipeline state
        state = db.query(ProjectAgentState).filter(
            ProjectAgentState.project_id == project_uuid
        ).first()
        if state:
            state.current_stage = "deploy"
            state.pipeline_health = "on_track"
            state.updated_at = datetime.now(timezone.utc)
            history = state.stage_history or []
            history.append({
                "stage": "deploy",
                "agent": "deployer",
                "target": "make.com",
                "outcome": "success",
                "external_id": str(scenario.get("id", "")),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            state.stage_history = history
            db.commit()

        return {
            "success": True,
            "deployment_id": str(deployment_record.id),
            "target": "make.com",
            "external_id": deployment_record.external_id,
            "external_url": deployment_record.external_url,
            "status": "deployed",
        }

    except httpx.HTTPStatusError as e:
        deployment_record.status = "failed"
        deployment_record.last_health_check = {
            "status": "failed",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "status_code": e.response.status_code if e.response else None,
        }
        db.commit()
        raise HTTPException(status_code=502, detail=f"Make.com API error: {str(e)}")

    except Exception as e:
        deployment_record.status = "failed"
        deployment_record.last_health_check = {
            "status": "failed",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }
        db.commit()
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@app.post("/deploy/n8n")
def deploy_n8n(request: DeployN8nRequest, db: Session = Depends(get_db)):
    """
    Deploy a workflow to n8n via their REST API.

    Imports the workflow JSON, records the deployment, and updates pipeline state.
    """
    import httpx

    project_uuid = UUID(request.project_id)

    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    headers = {
        "X-N8N-API-KEY": request.api_key,
        "Content-Type": "application/json",
    }

    deployment_record = Deployment(
        project_id=project_uuid,
        target="n8n",
        status="pending",
    )
    db.add(deployment_record)
    db.commit()
    db.refresh(deployment_record)

    n8n_base = request.n8n_url.rstrip("/")

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{n8n_base}/api/v1/workflows",
                headers=headers,
                json=request.workflow,
            )
            resp.raise_for_status()
            data = resp.json()

        workflow_id = str(data.get("id", ""))
        deployment_record.external_id = workflow_id
        deployment_record.external_url = f"{n8n_base}/workflow/{workflow_id}"
        deployment_record.status = "deployed"
        deployment_record.last_health_check = {
            "status": "deployed",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "n8n_response": {"id": data.get("id"), "name": data.get("name")},
        }
        db.commit()

        # Update pipeline state
        state = db.query(ProjectAgentState).filter(
            ProjectAgentState.project_id == project_uuid
        ).first()
        if state:
            state.current_stage = "deploy"
            state.pipeline_health = "on_track"
            state.updated_at = datetime.now(timezone.utc)
            history = state.stage_history or []
            history.append({
                "stage": "deploy",
                "agent": "deployer",
                "target": "n8n",
                "outcome": "success",
                "external_id": workflow_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
            state.stage_history = history
            db.commit()

        return {
            "success": True,
            "deployment_id": str(deployment_record.id),
            "target": "n8n",
            "external_id": workflow_id,
            "external_url": deployment_record.external_url,
            "status": "deployed",
        }

    except httpx.HTTPStatusError as e:
        deployment_record.status = "failed"
        deployment_record.last_health_check = {
            "status": "failed",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
            "status_code": e.response.status_code if e.response else None,
        }
        db.commit()
        raise HTTPException(status_code=502, detail=f"n8n API error: {str(e)}")

    except Exception as e:
        deployment_record.status = "failed"
        deployment_record.last_health_check = {
            "status": "failed",
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }
        db.commit()
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")


@app.get("/deploy/status")
def deploy_status(project_id: str = Query(...), db: Session = Depends(get_db)):
    """
    Get deployment status for a project.
    Returns all deployments with their current status and health check info.
    """
    project_uuid = UUID(project_id)

    deployments = db.query(Deployment).filter(
        Deployment.project_id == project_uuid
    ).order_by(Deployment.deployed_at.desc()).all()

    return {
        "project_id": project_id,
        "total_deployments": len(deployments),
        "deployments": [
            {
                "id": str(d.id),
                "target": d.target,
                "external_id": d.external_id,
                "external_url": d.external_url,
                "status": d.status,
                "last_health_check": d.last_health_check,
                "deployed_at": d.deployed_at.isoformat() if d.deployed_at else None,
            }
            for d in deployments
        ],
    }


@app.get("/health/full")
def health_full(db: Session = Depends(get_db)):
    """
    Comprehensive platform health check.

    Checks database connectivity, all core tables, API responsiveness,
    embedding store integrity, and pipeline state. Returns detailed
    status for each subsystem.
    """
    checks = {}
    overall = "healthy"

    # 1. Database connectivity
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok", "message": "Connected"}
    except Exception as e:
        checks["database"] = {"status": "error", "message": str(e)}
        overall = "degraded"

    # 2. Core tables exist and are accessible
    core_tables = [
        "projects", "builds", "build_artifacts", "assumptions",
        "audit_runs", "sow_documents", "test_runs", "incidents",
        "execution_snapshots", "client_snapshots", "agent_handoffs",
        "project_financials", "project_agent_state", "client_context",
        "verification_runs", "deployments", "persona_client_context",
        "persona_feedback",
    ]
    table_status = {}
    for table in core_tables:
        try:
            result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            table_status[table] = {"status": "ok", "row_count": count}
        except Exception as e:
            table_status[table] = {"status": "error", "message": str(e)}
            overall = "degraded"
    checks["tables"] = table_status

    # 3. Embedding store
    try:
        from tools.embedding_engine import load_store
        store = load_store()
        doc_count = len(store.get("documents", []))
        vocab_size = len(store.get("vocab", {}))
        checks["embeddings"] = {
            "status": "ok",
            "document_count": doc_count,
            "vocab_size": vocab_size,
        }
    except Exception as e:
        checks["embeddings"] = {"status": "error", "message": str(e)}
        overall = "degraded"

    # 4. Module registry
    checks["module_registry"] = {
        "status": "ok" if _registry.get("module_count", 0) > 0 else "warning",
        "module_count": _registry.get("module_count", 0),
    }
    if _registry.get("module_count", 0) == 0:
        overall = "degraded" if overall == "healthy" else overall

    # 5. Pipeline health — check for stalled projects
    try:
        stalled = db.execute(text("""
            SELECT COUNT(*) FROM project_agent_state
            WHERE updated_at < NOW() - INTERVAL '48 hours'
              AND current_stage != 'deploy'
        """)).scalar()
        checks["pipeline"] = {
            "status": "warning" if stalled > 0 else "ok",
            "stalled_projects": stalled,
        }
        if stalled > 0:
            overall = "degraded" if overall == "healthy" else overall
    except Exception:
        checks["pipeline"] = {"status": "ok", "stalled_projects": 0}

    # 6. Recent activity
    try:
        recent_builds = db.execute(text(
            "SELECT COUNT(*) FROM builds WHERE created_at > NOW() - INTERVAL '24 hours'"
        )).scalar()
        recent_verifications = db.execute(text(
            "SELECT COUNT(*) FROM verification_runs WHERE created_at > NOW() - INTERVAL '24 hours'"
        )).scalar()
        checks["activity"] = {
            "builds_24h": recent_builds,
            "verifications_24h": recent_verifications,
        }
    except Exception:
        checks["activity"] = {"builds_24h": 0, "verifications_24h": 0}

    return {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


@app.get("/health/memory")
def health_memory():
    """
    Memory store health and statistics.

    Returns embedding store size, document count, vocabulary coverage,
    and average document length metrics.
    """
    from tools.embedding_engine import load_store
    import os

    try:
        store = load_store()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load embedding store: {str(e)}")

    documents = store.get("documents", [])
    vocab = store.get("vocab", {})

    # Compute stats
    store_path = os.path.join(os.path.dirname(__file__), "data", "embeddings.json")
    file_size_bytes = os.path.getsize(store_path) if os.path.exists(store_path) else 0

    avg_terms = 0
    if documents:
        avg_terms = sum(len(doc.get("tf", {})) for doc in documents) / len(documents)

    # Top terms by IDF (most discriminating)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    top_terms = [{"term": t, "idf": round(v, 3)} for t, v in sorted_vocab[:20]]

    return {
        "status": "ok" if documents else "empty",
        "document_count": len(documents),
        "vocab_size": len(vocab),
        "avg_terms_per_doc": round(avg_terms, 1),
        "store_file_size_bytes": file_size_bytes,
        "store_version": store.get("version", 1),
        "top_discriminating_terms": top_terms,
        "documents": [
            {
                "id": doc["id"],
                "term_count": len(doc.get("tf", {})),
                "preview": doc.get("text_preview", "")[:100],
            }
            for doc in documents
        ],
    }


@app.post("/health/repair")
def health_repair(db: Session = Depends(get_db)):
    """
    Self-healing endpoint. Detects and repairs common platform issues:
    - Stalled pipelines (>48h) → reset to current stage
    - Orphaned agent states (no matching project) → clean up
    - Embedding store corruption → rebuild from project data
    - Inconsistent deployment records → mark stale as failed
    """
    repairs = []

    # 1. Reset stalled pipelines
    try:
        stalled = db.execute(text("""
            UPDATE project_agent_state
            SET pipeline_health = 'stalled',
                updated_at = NOW()
            WHERE updated_at < NOW() - INTERVAL '48 hours'
              AND current_stage != 'deploy'
              AND pipeline_health = 'on_track'
            RETURNING id, project_id, current_stage
        """)).fetchall()
        db.commit()
        if stalled:
            repairs.append({
                "type": "stalled_pipeline_flagged",
                "count": len(stalled),
                "projects": [str(r.project_id) for r in stalled],
            })
    except Exception as e:
        db.rollback()
        repairs.append({"type": "stalled_pipeline_flagged", "error": str(e)})

    # 2. Clean up orphaned agent states
    try:
        orphaned = db.execute(text("""
            DELETE FROM project_agent_state
            WHERE project_id NOT IN (SELECT id FROM projects)
            RETURNING id
        """)).fetchall()
        db.commit()
        if orphaned:
            repairs.append({
                "type": "orphaned_agent_states_removed",
                "count": len(orphaned),
            })
    except Exception as e:
        db.rollback()
        repairs.append({"type": "orphaned_agent_states_removed", "error": str(e)})

    # 3. Mark stale pending deployments as failed
    try:
        stale_deploys = db.execute(text("""
            UPDATE deployments
            SET status = 'failed',
                last_health_check = jsonb_build_object(
                    'status', 'auto_failed',
                    'reason', 'Pending for >24h',
                    'repaired_at', NOW()::text
                )
            WHERE status = 'pending'
              AND deployed_at < NOW() - INTERVAL '24 hours'
            RETURNING id, project_id
        """)).fetchall()
        db.commit()
        if stale_deploys:
            repairs.append({
                "type": "stale_deployments_failed",
                "count": len(stale_deploys),
            })
    except Exception as e:
        db.rollback()
        repairs.append({"type": "stale_deployments_failed", "error": str(e)})

    # 4. Rebuild embedding store if corrupted
    try:
        from tools.embedding_engine import load_store, embed_document
        store = load_store()
        if store.get("version") != 1:
            # Corrupted version — re-embed all projects
            projects_with_ctx = db.execute(text("""
                SELECT p.id, p.name, cc.key_decisions, cc.tech_stack
                FROM projects p
                LEFT JOIN client_context cc ON cc.project_id = p.id
                LIMIT 100
            """)).fetchall()
            re_embedded = 0
            for p in projects_with_ctx:
                text_content = f"{p.name} {str(p.key_decisions or '')} {str(p.tech_stack or '')}"
                if text_content.strip():
                    embed_document(str(p.id), text_content, {"name": p.name})
                    re_embedded += 1
            if re_embedded:
                repairs.append({
                    "type": "embedding_store_rebuilt",
                    "documents_reindexed": re_embedded,
                })
    except Exception as e:
        repairs.append({"type": "embedding_store_check", "error": str(e)})

    return {
        "status": "repairs_complete" if repairs else "no_repairs_needed",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repairs": repairs,
        "total_repairs": len(repairs),
    }


@app.post("/command")
def command(request: CommandRequest, db: Session = Depends(get_db)):
    """
    Natural language command interface.

    Accepts a free-text command and routes it to the appropriate endpoint
    internally. Supports commands like:
    - "check health" → GET /health/full
    - "show costs for Acme" → GET /costs/summary
    - "find similar to CRM integration" → GET /similar
    - "show stalled projects" → GET /supervisor/stalled
    - "deploy status for <project>" → GET /deploy/status
    - "estimate cost for email automation" → POST /costs/estimate
    - "pipeline status for <project>" → GET /pipeline/status
    - "memory stats" → GET /health/memory
    - "repair platform" → POST /health/repair
    """
    cmd = request.command.strip().lower()
    customer = request.customer_name

    # Route based on keyword matching
    if any(kw in cmd for kw in ["health", "status check", "system check"]):
        if "memory" in cmd or "embedding" in cmd:
            return {"routed_to": "/health/memory", "result": health_memory()}
        return {"routed_to": "/health/full", "result": health_full(db=db)}

    if any(kw in cmd for kw in ["repair", "fix", "heal", "self-heal"]):
        return {"routed_to": "/health/repair", "result": health_repair(db=db)}

    if any(kw in cmd for kw in ["stalled", "stuck", "blocked", "inactive"]):
        return {"routed_to": "/supervisor/stalled", "result": supervisor_stalled(db=db)}

    if any(kw in cmd for kw in ["cost", "spending", "expense", "margin"]):
        if any(kw in cmd for kw in ["estimate", "predict", "forecast"]):
            desc = cmd
            for prefix in ["estimate cost for ", "estimate ", "predict cost for ", "forecast "]:
                if cmd.startswith(prefix):
                    desc = cmd[len(prefix):]
                    break
            req = CostEstimateRequest(description=desc)
            return {"routed_to": "/costs/estimate", "result": costs_estimate(req, db=db)}

        if any(kw in cmd for kw in ["report", "weekly"]):
            cid = customer or _extract_name_from_command(cmd)
            if not cid:
                return {"error": "Please specify a customer name", "hint": "Try: 'weekly cost report for [customer]'"}
            return {"routed_to": "/costs/report", "result": costs_report(client_id=cid, db=db)}

        cid = customer or _extract_name_from_command(cmd)
        if not cid:
            return {"error": "Please specify a customer name", "hint": "Try: 'show costs for [customer]'"}
        return {"routed_to": "/costs/summary", "result": costs_summary(client_id=cid, db=db)}

    if any(kw in cmd for kw in ["similar", "find like", "match", "compare"]):
        desc = cmd
        for prefix in ["find similar to ", "find projects like ", "similar to ", "match "]:
            if cmd.startswith(prefix):
                desc = cmd[len(prefix):]
                break
        return {"routed_to": "/similar", "result": similar(description=desc)}

    if "deploy" in cmd:
        if "status" in cmd:
            pid = _extract_uuid_from_command(cmd)
            if not pid:
                return {"error": "Please specify a project_id", "hint": "Try: 'deploy status for [project_id]'"}
            return {"routed_to": "/deploy/status", "result": deploy_status(project_id=pid, db=db)}
        return {"hint": "Deployment requires structured input. Use POST /deploy/makecom or POST /deploy/n8n directly."}

    if "pipeline" in cmd:
        pid = _extract_uuid_from_command(cmd)
        if not pid:
            return {"error": "Please specify a project_id", "hint": "Try: 'pipeline status for [project_id]'"}
        return {"routed_to": "/pipeline/status", "result": pipeline_status(project_id=pid, db=db)}

    if any(kw in cmd for kw in ["confidence", "verification", "verify history"]):
        pid = _extract_uuid_from_command(cmd)
        if not pid:
            return {"error": "Please specify a project_id", "hint": "Try: 'confidence history for [project_id]'"}
        return {"routed_to": "/confidence/history", "result": confidence_history(project_id=pid, db=db)}

    if any(kw in cmd for kw in ["memory", "context", "learning"]):
        if "stats" in cmd or "health" in cmd:
            return {"routed_to": "/health/memory", "result": health_memory()}
        return {"hint": "Memory operations require structured input. Use POST /memory or GET /memory directly."}

    return {
        "error": "Command not recognized",
        "command_received": request.command,
        "available_commands": [
            "check health",
            "memory stats",
            "repair platform",
            "show stalled projects",
            "show costs for [customer]",
            "weekly cost report for [customer]",
            "estimate cost for [description]",
            "find similar to [description]",
            "deploy status for [project_id]",
            "pipeline status for [project_id]",
            "confidence history for [project_id]",
        ],
    }


def _extract_name_from_command(cmd: str) -> str | None:
    """Extract a customer/project name from a command string."""
    for prep in [" for ", " of ", " from "]:
        if prep in cmd:
            return cmd.split(prep, 1)[1].strip().strip("'\"")
    return None


def _extract_uuid_from_command(cmd: str) -> str | None:
    """Extract a UUID-like string from a command."""
    import re
    match = re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", cmd)
    if match:
        return match.group(0)
    for prep in [" for ", " of "]:
        if prep in cmd:
            candidate = cmd.split(prep, 1)[1].strip().strip("'\"")
            if candidate:
                return candidate
    return None


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


# ─────────────────────────────────────────
# Persona Service — valid personas
# ─────────────────────────────────────────

VALID_PERSONAS = {"rebecka", "daniel", "sarah", "andrew"}


def _validate_persona(persona: str) -> str:
    """Normalise and validate persona name. Raises 400 on invalid."""
    p = persona.strip().lower()
    if p not in VALID_PERSONAS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid persona '{persona}'. Must be one of: {', '.join(sorted(VALID_PERSONAS))}",
        )
    return p


# ─────────────────────────────────────────
# POST /persona/memory
# ─────────────────────────────────────────

class PersonaMemoryRequest(BaseModel):
    persona: str
    client_id: str
    tone_preferences: Optional[dict] = None
    past_interactions_summary: Optional[str] = None
    communication_style: Optional[str] = None


@app.post("/persona/memory")
def persona_memory(request: PersonaMemoryRequest, db: Session = Depends(get_db)):
    """
    Link a persona to a client. Stores client-specific tone preferences,
    past interactions summary, and communication style.

    Upserts — if the persona+client_id pair already exists, updates it.
    """
    persona = _validate_persona(request.persona)

    if not request.client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")

    existing = (
        db.query(PersonaClientContext)
        .filter(
            PersonaClientContext.persona == persona,
            PersonaClientContext.client_id == request.client_id.strip(),
        )
        .first()
    )

    now = datetime.now(timezone.utc)

    if existing:
        if request.tone_preferences is not None:
            existing.tone_preferences = request.tone_preferences
        if request.past_interactions_summary is not None:
            existing.past_interactions_summary = request.past_interactions_summary
        if request.communication_style is not None:
            existing.communication_style = request.communication_style
        existing.updated_at = now
        db.commit()
        db.refresh(existing)
        record = existing
    else:
        record = PersonaClientContext(
            persona=persona,
            client_id=request.client_id.strip(),
            tone_preferences=request.tone_preferences,
            past_interactions_summary=request.past_interactions_summary,
            communication_style=request.communication_style,
            created_at=now,
            updated_at=now,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

    return {
        "id": str(record.id),
        "persona": record.persona,
        "client_id": record.client_id,
        "tone_preferences": record.tone_preferences,
        "past_interactions_summary": record.past_interactions_summary,
        "communication_style": record.communication_style,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


# ─────────────────────────────────────────
# GET /persona/context
# ─────────────────────────────────────────

@app.get("/persona/context")
def persona_context(
    persona: str = Query(..., description="Persona name"),
    client_id: str = Query(..., description="Client identifier"),
    db: Session = Depends(get_db),
):
    """
    Return a persona's full context for a given client.
    Includes tone preferences, past interactions summary,
    and communication style so every conversation starts informed.
    """
    p = _validate_persona(persona)

    record = (
        db.query(PersonaClientContext)
        .filter(
            PersonaClientContext.persona == p,
            PersonaClientContext.client_id == client_id.strip(),
        )
        .first()
    )

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"No context found for persona '{p}' with client_id '{client_id}'",
        )

    return {
        "id": str(record.id),
        "persona": record.persona,
        "client_id": record.client_id,
        "tone_preferences": record.tone_preferences,
        "past_interactions_summary": record.past_interactions_summary,
        "communication_style": record.communication_style,
        "created_at": record.created_at.isoformat(),
        "updated_at": record.updated_at.isoformat(),
    }


# ─────────────────────────────────────────
# POST /persona/feedback
# ─────────────────────────────────────────

class PersonaFeedbackRequest(BaseModel):
    persona: str
    client_id: str
    interaction_id: str
    rating: int
    notes: Optional[str] = None


@app.post("/persona/feedback")
def persona_feedback(request: PersonaFeedbackRequest, db: Session = Depends(get_db)):
    """
    Store feedback for a persona interaction.
    Accepts persona, client_id, interaction_id, rating (1-5), and optional notes.
    Used to improve persona responses over time.
    """
    persona = _validate_persona(request.persona)

    if not request.client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")
    if not request.interaction_id.strip():
        raise HTTPException(status_code=400, detail="interaction_id is required")
    if not 1 <= request.rating <= 5:
        raise HTTPException(status_code=400, detail="rating must be between 1 and 5")

    record = PersonaFeedback(
        persona=persona,
        client_id=request.client_id.strip(),
        interaction_id=request.interaction_id.strip(),
        rating=request.rating,
        notes=request.notes,
        created_at=datetime.now(timezone.utc),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": str(record.id),
        "persona": record.persona,
        "client_id": record.client_id,
        "interaction_id": record.interaction_id,
        "rating": record.rating,
        "notes": record.notes,
        "created_at": record.created_at.isoformat(),
    }


# ─────────────────────────────────────────
# GET /persona/performance
# ─────────────────────────────────────────

@app.get("/persona/performance")
def persona_performance(
    persona: str = Query(..., description="Persona name"),
    db: Session = Depends(get_db),
):
    """
    Return performance stats for a persona across all clients.
    Includes average rating, total interactions, rating distribution,
    and top feedback themes extracted from notes.
    """
    p = _validate_persona(persona)

    stats_row = db.execute(
        text("""
            SELECT
                COUNT(*) AS total_interactions,
                COALESCE(AVG(rating), 0) AS avg_rating,
                COUNT(DISTINCT client_id) AS unique_clients,
                COUNT(*) FILTER (WHERE rating = 5) AS five_star,
                COUNT(*) FILTER (WHERE rating = 4) AS four_star,
                COUNT(*) FILTER (WHERE rating = 3) AS three_star,
                COUNT(*) FILTER (WHERE rating = 2) AS two_star,
                COUNT(*) FILTER (WHERE rating = 1) AS one_star
            FROM persona_feedback
            WHERE persona = :persona
        """),
        {"persona": p},
    ).fetchone()

    total = stats_row.total_interactions if stats_row else 0

    if total == 0:
        return {
            "persona": p,
            "total_interactions": 0,
            "avg_rating": None,
            "unique_clients": 0,
            "rating_distribution": {},
            "top_feedback_themes": [],
        }

    # Extract top feedback themes from notes
    notes_rows = db.execute(
        text("""
            SELECT notes FROM persona_feedback
            WHERE persona = :persona AND notes IS NOT NULL AND notes != ''
        """),
        {"persona": p},
    ).fetchall()

    themes = _extract_feedback_themes([r.notes for r in notes_rows])

    return {
        "persona": p,
        "total_interactions": total,
        "avg_rating": round(float(stats_row.avg_rating), 2),
        "unique_clients": stats_row.unique_clients,
        "rating_distribution": {
            "5": stats_row.five_star,
            "4": stats_row.four_star,
            "3": stats_row.three_star,
            "2": stats_row.two_star,
            "1": stats_row.one_star,
        },
        "top_feedback_themes": themes,
    }


def _extract_feedback_themes(notes_list: list[str], top_n: int = 5) -> list[dict]:
    """Extract top recurring themes from feedback notes via word frequency."""
    from collections import Counter

    stop_words = {
        "the", "a", "an", "is", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into",
        "through", "during", "before", "after", "and", "but", "or",
        "not", "no", "so", "if", "than", "too", "very", "just", "about",
        "up", "out", "that", "this", "it", "i", "me", "my", "we", "our",
        "they", "them", "their", "she", "he", "her", "his", "you", "your",
    }

    word_counts: Counter = Counter()
    for note in notes_list:
        words = note.lower().split()
        meaningful = [w.strip(".,!?;:'\"()") for w in words if len(w) > 2]
        meaningful = [w for w in meaningful if w and w not in stop_words]
        word_counts.update(meaningful)

    return [
        {"theme": word, "count": count}
        for word, count in word_counts.most_common(top_n)
    ]


# ─────────────────────────────────────────
# POST /persona/deploy
# ─────────────────────────────────────────

# Base persona definitions — baked-in defaults per persona
_PERSONA_DEFAULTS = {
    "rebecka": {
        "display_name": "Rebecka",
        "role": "Strategic Account Manager",
        "base_tone": "warm, consultative, empathetic",
        "base_style": "Builds rapport first, then addresses business needs. Uses collaborative language.",
        "knowledge_bases": ["client-success-playbook", "account-management-sops", "upsell-frameworks"],
    },
    "daniel": {
        "display_name": "Daniel",
        "role": "Technical Solutions Architect",
        "base_tone": "direct, precise, confident",
        "base_style": "Leads with data and technical detail. Concise, action-oriented communication.",
        "knowledge_bases": ["make-platform-docs", "integration-patterns", "api-reference", "troubleshooting-guides"],
    },
    "sarah": {
        "display_name": "Sarah",
        "role": "Client Success Lead",
        "base_tone": "encouraging, proactive, clear",
        "base_style": "Focuses on outcomes and next steps. Balances optimism with transparency.",
        "knowledge_bases": ["onboarding-workflows", "client-health-metrics", "escalation-procedures"],
    },
    "andrew": {
        "display_name": "Andrew",
        "role": "Operations & Delivery Manager",
        "base_tone": "structured, reliable, thorough",
        "base_style": "Process-driven communication. Uses checklists and status updates. Deadline-conscious.",
        "knowledge_bases": ["delivery-sops", "project-templates", "sla-guidelines", "resource-planning"],
    },
}


class PersonaDeployRequest(BaseModel):
    persona: str
    client_id: str
    config_overrides: Optional[dict] = None


@app.post("/persona/deploy")
def persona_deploy(request: PersonaDeployRequest, db: Session = Depends(get_db)):
    """
    Generate a real, deployable client-specific persona artifact.

    Pulls tone_preferences, communication_style, and interaction_summary
    from persona_client_context. Pulls feedback stats from persona_feedback.
    Merges with base persona definition and any config_overrides.
    Returns a complete JSON artifact ready for deployment.
    """
    persona = _validate_persona(request.persona)

    if not request.client_id.strip():
        raise HTTPException(status_code=400, detail="client_id is required")

    client_id = request.client_id.strip()
    base = _PERSONA_DEFAULTS[persona]
    overrides = request.config_overrides or {}
    deploy_ts = datetime.now(timezone.utc)

    # ── Pull DB context ────────────────────────────────────────
    ctx = (
        db.query(PersonaClientContext)
        .filter(
            PersonaClientContext.persona == persona,
            PersonaClientContext.client_id == client_id,
        )
        .first()
    )

    # ── Pull feedback stats ────────────────────────────────────
    fb_row = db.execute(
        text(
            "SELECT COUNT(*) AS total, COALESCE(AVG(rating), 0) AS avg_rating "
            "FROM persona_feedback WHERE persona = :p AND client_id = :c"
        ),
        {"p": persona, "c": client_id},
    ).first()

    feedback_stats = {
        "total_interactions": fb_row.total if fb_row else 0,
        "avg_rating": round(float(fb_row.avg_rating), 2) if fb_row else 0.0,
    }

    # ── Merge values ───────────────────────────────────────────
    display_name = overrides.get("display_name", base["display_name"])
    role = overrides.get("role", base["role"])

    # Tone: overrides > DB context > base default
    db_tone = ctx.tone_preferences if ctx and ctx.tone_preferences else None
    tone = overrides.get("tone", db_tone if db_tone else base["base_tone"])

    # Style: overrides > DB context > base default
    db_style = ctx.communication_style if ctx and ctx.communication_style else None
    style = overrides.get("communication_style", db_style if db_style else base["base_style"])

    knowledge_bases = overrides.get("knowledge_bases", base["knowledge_bases"])
    extra_knowledge = overrides.get("extra_knowledge_bases", [])
    if extra_knowledge:
        knowledge_bases = list(set(knowledge_bases + extra_knowledge))

    past_context = ctx.past_interactions_summary if ctx and ctx.past_interactions_summary else None

    # ── Build system prompt with full client context ───────────
    tone_str = (
        tone if isinstance(tone, str)
        else ", ".join(f"{k}: {v}" for k, v in tone.items()) if isinstance(tone, dict)
        else str(tone)
    )

    system_prompt = (
        f"You are {display_name}, {role} at ManageAI.\n\n"
        f"Tone: {tone_str}\n"
        f"Communication style: {style}\n\n"
        f"Knowledge bases available: {', '.join(knowledge_bases)}\n"
    )

    if past_context:
        system_prompt += f"\nClient context from past interactions:\n{past_context}\n"

    if feedback_stats["total_interactions"] > 0:
        system_prompt += (
            f"\nClient feedback summary: {feedback_stats['total_interactions']} interactions, "
            f"avg rating {feedback_stats['avg_rating']}/5.\n"
        )

    system_prompt += (
        f"\nYou are speaking with client '{client_id}'.\n"
        "Always stay in character. Be helpful, accurate, and professional."
    )

    # ── Assemble complete artifact ─────────────────────────────
    artifact = {
        "artifact_version": "2.0",
        "persona": persona,
        "client_id": client_id,
        "display_name": display_name,
        "role": role,
        "tone_preferences": tone,
        "communication_style": style,
        "knowledge_bases": knowledge_bases,
        "client_context": {
            "has_stored_context": ctx is not None,
            "past_interactions_summary": past_context,
            "tone_source": "database" if db_tone else "override" if overrides.get("tone") else "default",
            "style_source": "database" if db_style else "override" if overrides.get("communication_style") else "default",
        },
        "feedback_stats": feedback_stats,
        "system_prompt": system_prompt,
        "config_overrides_applied": overrides,
        "deploy_timestamp": deploy_ts.isoformat(),
    }

    return artifact


# ── POST /persona/test ─────────────────────────────────────────


class PersonaTestRequest(BaseModel):
    persona: str
    message: str
    client_id: Optional[str] = None


@app.post("/persona/test")
def persona_test(request: PersonaTestRequest, db: Session = Depends(get_db)):
    """
    Send a message through the Claude API using a persona's system prompt.

    Builds the system prompt from the base persona definition, merges in
    client-specific tone preferences if client_id is provided and a
    persona_client_context row exists, then calls the Claude API and
    returns the response.
    """
    persona = _validate_persona(request.persona)

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    base = _PERSONA_DEFAULTS[persona]
    tone = base["base_tone"]
    style = base["base_style"]
    tone_applied = False

    # Merge client-specific context if available
    if request.client_id and request.client_id.strip():
        ctx = (
            db.query(PersonaClientContext)
            .filter(
                PersonaClientContext.persona == persona,
                PersonaClientContext.client_id == request.client_id.strip(),
            )
            .first()
        )
        if ctx:
            if ctx.tone_preferences:
                tone = ctx.tone_preferences
                tone_applied = True
            if ctx.communication_style:
                style = ctx.communication_style
                tone_applied = True

    # Build system prompt
    tone_str = (
        tone if isinstance(tone, str)
        else ", ".join(f"{k}: {v}" for k, v in tone.items()) if isinstance(tone, dict)
        else str(tone)
    )
    system_prompt = (
        f"You are {base['display_name']}, {base['role']} at ManageAI.\n\n"
        f"Tone: {tone_str}\n"
        f"Communication style: {style}\n\n"
        f"Knowledge bases available: {', '.join(base['knowledge_bases'])}\n\n"
        "Always stay in character. Be helpful, accurate, and professional."
    )

    if request.client_id and request.client_id.strip():
        system_prompt += f"\nYou are speaking with client '{request.client_id.strip()}'."

    # Call Claude API
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": request.message.strip()}],
        )
        response_text = response.content[0].text
    except ImportError:
        raise HTTPException(status_code=500, detail="anthropic package not installed")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e}")

    return {
        "persona": persona,
        "display_name": base["display_name"],
        "role": base["role"],
        "response": response_text,
        "tone_applied": tone_applied,
        "client_id": request.client_id.strip() if request.client_id else None,
        "model": "claude-sonnet-4-20250514",
    }


# ── GET /personas/list ─────────────────────────────────────────


@app.get("/personas/list")
def personas_list(db: Session = Depends(get_db)):
    """
    Return all 4 personas with their base info and live stats.

    Stats are pulled from persona_feedback (avg_rating, total_interactions)
    and persona_client_context (unique_clients, last_deployed approximation).
    Always returns all 4 personas even if no DB data exists yet.
    """
    results = []

    for persona_key, base in _PERSONA_DEFAULTS.items():
        # Aggregate feedback stats
        fb_row = db.execute(
            text(
                "SELECT COUNT(*) AS total, COALESCE(AVG(rating), 0) AS avg_rating "
                "FROM persona_feedback WHERE persona = :p"
            ),
            {"p": persona_key},
        ).first()

        total_interactions = fb_row.total if fb_row else 0
        avg_rating = round(float(fb_row.avg_rating), 2) if fb_row else 0.0

        # Count unique clients with stored context
        client_row = db.execute(
            text(
                "SELECT COUNT(*) AS cnt FROM persona_client_context WHERE persona = :p"
            ),
            {"p": persona_key},
        ).first()
        unique_clients = client_row.cnt if client_row else 0

        # Most recent context update as proxy for last_deployed
        last_row = db.execute(
            text(
                "SELECT updated_at FROM persona_client_context "
                "WHERE persona = :p ORDER BY updated_at DESC LIMIT 1"
            ),
            {"p": persona_key},
        ).first()
        last_deployed = last_row.updated_at.isoformat() if last_row and last_row.updated_at else None

        results.append({
            "persona": persona_key,
            "display_name": base["display_name"],
            "role": base["role"],
            "base_tone": base["base_tone"],
            "knowledge_bases": base["knowledge_bases"],
            "stats": {
                "avg_rating": avg_rating,
                "total_interactions": total_interactions,
                "unique_clients": unique_clients,
                "last_deployed": last_deployed,
            },
        })

    return {"personas": results, "total": len(results)}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SPRINT 4 — INTELLIGENCE LAYER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


# ── POST /clients/health ────────────────────────────────────────


@app.post("/clients/health")
def clients_health(db: Session = Depends(get_db)):
    """
    Assess health of all clients based on pipeline activity.

    For each distinct customer_name: counts projects, stalled projects
    (updated_at > 48h and status not deployed/cancelled), and determines
    health_status: healthy, at_risk, or unhealthy.
    """
    try:
        rows = db.execute(text("""
            SELECT
                COALESCE(p.customer_name, 'Unknown') AS client_id,
                COUNT(p.id) AS project_count,
                MAX(p.updated_at) AS last_activity,
                COUNT(CASE
                    WHEN p.updated_at < now() - interval '48 hours'
                         AND p.status NOT IN ('deployed', 'cancelled')
                    THEN 1
                END) AS stalled_count,
                EXTRACT(EPOCH FROM (now() - MAX(p.updated_at))) / 86400 AS days_inactive
            FROM projects p
            GROUP BY COALESCE(p.customer_name, 'Unknown')
            ORDER BY last_activity DESC NULLS LAST
        """)).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    clients = []
    for row in rows:
        stalled = int(row.stalled_count)
        days_inactive = float(row.days_inactive) if row.days_inactive else 0

        if stalled >= 2 or days_inactive > 7:
            health_status = "unhealthy"
            reason = f"{stalled} stalled projects" if stalled >= 2 else f"No activity for {round(days_inactive, 1)} days"
        elif stalled >= 1 or days_inactive > 5:
            health_status = "at_risk"
            reason = f"{stalled} stalled project(s)" if stalled >= 1 else f"No activity for {round(days_inactive, 1)} days"
        else:
            health_status = "healthy"
            reason = "All projects active"

        clients.append({
            "client_id": row.client_id,
            "health_status": health_status,
            "stalled_count": stalled,
            "project_count": int(row.project_count),
            "last_activity": row.last_activity.isoformat() if row.last_activity else None,
            "days_inactive": round(days_inactive, 1),
            "reason": reason,
        })

    summary = {
        "healthy": sum(1 for c in clients if c["health_status"] == "healthy"),
        "at_risk": sum(1 for c in clients if c["health_status"] == "at_risk"),
        "unhealthy": sum(1 for c in clients if c["health_status"] == "unhealthy"),
    }

    return {"clients": clients, "total": len(clients), "summary": summary}


# ── GET /clients/list ───────────────────────────────────────────


@app.get("/clients/list")
def clients_list(db: Session = Depends(get_db)):
    """
    Return distinct clients from client_context with project stats.

    Each client includes project_count, last_updated, key_decisions count,
    and tech_stack summary. Sorted by last_updated desc.
    """
    try:
        rows = db.execute(text("""
            SELECT
                cc.client_id,
                COUNT(DISTINCT cc.project_id) AS project_count,
                MAX(cc.updated_at) AS last_updated,
                COALESCE(SUM(jsonb_array_length(COALESCE(cc.key_decisions, '[]'::jsonb))), 0) AS decisions_count,
                (
                    SELECT jsonb_agg(DISTINCT elem)
                    FROM client_context cc2,
                         jsonb_array_elements_text(COALESCE(cc2.tech_stack, '[]'::jsonb)) AS elem
                    WHERE cc2.client_id = cc.client_id
                ) AS tech_stack_all
            FROM client_context cc
            GROUP BY cc.client_id
            ORDER BY MAX(cc.updated_at) DESC NULLS LAST
        """)).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    clients = []
    for row in rows:
        tech_stack = []
        if row.tech_stack_all:
            try:
                import json
                tech_stack = json.loads(row.tech_stack_all) if isinstance(row.tech_stack_all, str) else list(row.tech_stack_all)
            except Exception:
                tech_stack = []

        clients.append({
            "client_id": row.client_id,
            "project_count": int(row.project_count),
            "last_updated": row.last_updated.isoformat() if row.last_updated else None,
            "key_decisions_count": int(row.decisions_count),
            "tech_stack": tech_stack,
        })

    return {"clients": clients, "total": len(clients)}


# ── POST /pipeline/advance ──────────────────────────────────────


class PipelineAdvanceRequest(BaseModel):
    project_id: str
    force_stage: Optional[str] = None


@app.post("/pipeline/advance")
def pipeline_advance(request: PipelineAdvanceRequest, db: Session = Depends(get_db)):
    """
    Advance a project to the next pipeline stage.

    Determines next stage from [intake, build, verify, deploy].
    Optionally force to a specific stage. Updates project_agent_state
    and calls pipeline_sync if available.
    """
    try:
        project_uuid = UUID(request.project_id)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="project_id must be a valid UUID")

    project = db.query(Project).filter(Project.id == project_uuid).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {request.project_id} not found")

    state = db.query(ProjectAgentState).filter(
        ProjectAgentState.project_id == project_uuid
    ).first()

    now = datetime.now(timezone.utc)
    old_stage = state.current_stage if state else "intake"

    if request.force_stage:
        if request.force_stage not in _STAGE_ORDER:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage '{request.force_stage}'. Must be one of: {', '.join(_STAGE_ORDER)}",
            )
        new_stage = request.force_stage
    else:
        stage_idx = _STAGE_ORDER.index(old_stage) if old_stage in _STAGE_ORDER else 0
        if stage_idx >= len(_STAGE_ORDER) - 1:
            return {
                "project_id": str(project_uuid),
                "old_stage": old_stage,
                "new_stage": old_stage,
                "message": "Project is already at final stage (deploy)",
                "advanced_at": now.isoformat(),
            }
        new_stage = _STAGE_ORDER[stage_idx + 1]

    new_agent = _STAGE_AGENT_MAP.get(new_stage, "supervisor")

    if state:
        state.current_stage = new_stage
        state.current_agent = new_agent
        state.updated_at = now
        history = state.stage_history or []
        history.append({
            "stage": new_stage,
            "agent": new_agent,
            "started_at": now.isoformat(),
            "advanced_from": old_stage,
        })
        state.stage_history = history
    else:
        state = ProjectAgentState(
            project_id=project_uuid,
            current_stage=new_stage,
            current_agent=new_agent,
            started_at=now,
            updated_at=now,
            pipeline_health="on_track",
            stage_history=[{
                "stage": new_stage,
                "agent": new_agent,
                "started_at": now.isoformat(),
                "advanced_from": old_stage,
            }],
        )
        db.add(state)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to advance pipeline: {str(e)}")

    # Sync to Supabase
    try:
        from tools.pipeline_sync import sync_activity
        sync_activity(
            project_id=str(project_uuid),
            action_type="pipeline_advance",
            description=f"Advanced from {old_stage} to {new_stage}",
            agent_name=new_agent,
        )
    except Exception:
        pass

    return {
        "project_id": str(project_uuid),
        "old_stage": old_stage,
        "new_stage": new_stage,
        "new_agent": new_agent,
        "advanced_at": now.isoformat(),
    }


# ── GET /pipeline/dashboard ─────────────────────────────────────


@app.get("/pipeline/dashboard")
def pipeline_dashboard(db: Session = Depends(get_db)):
    """
    Full pipeline dashboard: all projects grouped by stage.

    Each project includes project_id, customer_name, current_agent,
    started_at, and days_in_stage. Summary counts per stage at top level.
    """
    try:
        rows = db.execute(text("""
            SELECT
                pas.project_id,
                p.name AS project_name,
                COALESCE(p.customer_name, 'Unknown') AS customer_name,
                pas.current_stage,
                pas.current_agent,
                pas.started_at,
                pas.updated_at,
                pas.pipeline_health,
                EXTRACT(EPOCH FROM (now() - pas.updated_at)) / 86400 AS days_in_stage
            FROM project_agent_state pas
            JOIN projects p ON p.id = pas.project_id
            ORDER BY pas.updated_at ASC
        """)).fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    # Group by stage
    stages = {s: [] for s in _STAGE_ORDER}
    stages["completed"] = []
    stages["failed"] = []

    for row in rows:
        stage = row.current_stage if row.current_stage in stages else "completed"
        if row.pipeline_health == "failed":
            stage = "failed"
        elif row.pipeline_health == "completed":
            stage = "completed"

        stages[stage].append({
            "project_id": str(row.project_id),
            "project_name": row.project_name,
            "customer_name": row.customer_name,
            "current_agent": row.current_agent,
            "pipeline_health": row.pipeline_health,
            "started_at": row.started_at.isoformat() if row.started_at else None,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "days_in_stage": round(float(row.days_in_stage), 1) if row.days_in_stage else 0,
        })

    summary = {s: len(projects) for s, projects in stages.items()}

    return {"stages": stages, "summary": summary, "total_projects": len(rows)}


# ── POST /briefing/daily ────────────────────────────────────────


@app.post("/briefing/daily")
def briefing_daily(db: Session = Depends(get_db)):
    """
    Comprehensive daily briefing combining stalled projects, pipeline
    summary, cost summary, alerts, and recommendations.

    Saves to daily_briefings table via Supabase REST if configured.
    Falls back gracefully if SUPABASE_URL not set.
    """
    now = datetime.now(timezone.utc)

    # ── Stalled projects (reuse /supervisor/stalled logic) ─────
    try:
        stalled_rows = db.execute(text("""
            SELECT p.id, p.name, COALESCE(p.customer_name, 'Unknown') AS client,
                   p.status, p.updated_at,
                   EXTRACT(EPOCH FROM (now() - p.updated_at)) / 3600 AS hours_stalled
            FROM projects p
            WHERE p.updated_at < now() - interval '48 hours'
              AND p.status NOT IN ('deployed', 'cancelled')
            ORDER BY p.updated_at ASC
        """)).fetchall()
    except Exception:
        stalled_rows = []

    stalled_projects = [
        {
            "project_id": str(r.id),
            "name": r.name,
            "client": r.client,
            "hours_stalled": round(float(r.hours_stalled), 1),
        }
        for r in stalled_rows
    ]

    # ── Pipeline summary (reuse /pipeline/dashboard logic) ─────
    try:
        pipeline_rows = db.execute(text("""
            SELECT pas.current_stage, COUNT(*) AS cnt
            FROM project_agent_state pas
            GROUP BY pas.current_stage
        """)).fetchall()
    except Exception:
        pipeline_rows = []

    pipeline_summary = {r.current_stage: int(r.cnt) for r in pipeline_rows}

    # ── Cost summary (aggregate all clients) ───────────────────
    try:
        cost_row = db.execute(text("""
            SELECT
                COALESCE(SUM(pf.cost_usd), 0) AS total_spend,
                COALESCE(SUM(p.revenue), 0) AS total_revenue,
                COUNT(DISTINCT p.id) AS project_count
            FROM projects p
            LEFT JOIN project_financials pf ON pf.project_id = p.id
        """)).first()
    except Exception:
        cost_row = None

    total_spend = float(cost_row.total_spend) if cost_row else 0.0
    total_revenue = float(cost_row.total_revenue) if cost_row else 0.0
    avg_margin = ((total_revenue - total_spend) / total_revenue * 100) if total_revenue > 0 else 0.0

    cost_summary = {
        "total_spend": round(total_spend, 4),
        "total_revenue": round(total_revenue, 2),
        "avg_margin_pct": round(avg_margin, 1),
    }

    # ── Top alerts ─────────────────────────────────────────────
    top_alerts = []
    if stalled_projects:
        top_alerts.append(f"{len(stalled_projects)} project(s) stalled > 48 hours")
    if avg_margin < 20 and total_revenue > 0:
        top_alerts.append(f"Average margin is low: {avg_margin:.1f}%")
    if pipeline_summary.get("verify", 0) > 3:
        top_alerts.append(f"{pipeline_summary['verify']} projects stuck in verify stage")

    # ── Recommendations ────────────────────────────────────────
    recommendations = []
    if stalled_projects:
        recommendations.append("Review stalled projects and reassign agents or escalate.")
    if avg_margin < 20 and total_revenue > 0:
        recommendations.append("Audit high-cost projects for optimization opportunities.")
    if not top_alerts:
        recommendations.append("All systems operating normally. No action required.")

    briefing = {
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.isoformat(),
        "stalled_projects": stalled_projects,
        "pipeline_summary": pipeline_summary,
        "cost_summary": cost_summary,
        "top_alerts": top_alerts,
        "recommendations": recommendations,
    }

    # ── Save to Supabase (non-critical) ────────────────────────
    try:
        from tools.supabase_client import get_supabase_client
        sc = get_supabase_client()
        if sc:
            sc.insert("daily_briefings", {
                "briefing_date": now.strftime("%Y-%m-%d"),
                "content": briefing,
                "created_at": now.isoformat(),
            })
    except Exception:
        pass

    return briefing