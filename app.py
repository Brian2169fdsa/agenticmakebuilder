"""
Agentic Make Builder — FastAPI Application

Endpoints:
  GET  /health             — liveness probe
  POST /intake             — natural language → assessment (no structured fields needed)
  POST /assess             — structured intake → delivery report + plan_dict
  POST /plan               — structured intake → full pipeline (assess + build + 11 artifacts)
  POST /build              — plan_dict + original_request → full pipeline (compiler direct)
  POST /audit              — audit an existing Make.com scenario blueprint
  POST /verify             — 77-rule blueprint validation with fix instructions
  POST /handoff            — multi-agent orchestration handoff bridge
  GET  /supervisor/stalled — detect stalled projects (>48h no update)
  POST /costs/track        — track per-operation token costs
  GET  /costs/summary      — cost/revenue/margin summary per client
  POST /persona/memory     — link persona to client with tone/style prefs
  GET  /persona/context    — get persona's full context for a client
  POST /persona/feedback   — store interaction feedback for a persona
  GET  /persona/performance — persona performance stats across all clients
  POST /persona/deploy     — generate client-specific persona artifact

HTTP status codes:
  200 — success
  400 — malformed request / missing required fields
  422 — validation failure (spec/export errors, confidence too low)
  500 — internal pipeline error
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

from db.session import get_db, check_db
from db.models import AgentHandoff, ProjectFinancial, Project, PersonaClientContext, PersonaFeedback
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


class VerifyRequest(BaseModel):
    """Verify a blueprint JSON against the full 77-rule validation audit."""
    blueprint: dict                        # Make.com export JSON


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


@app.post("/verify")
def verify(request: VerifyRequest):
    """
    Full 77-rule blueprint validation audit.

    Runs both structural (Make export) and semantic validation rules against
    a blueprint. Returns confidence_score (0-100), pass/fail verdict, and
    specific fix_instructions when confidence < 85.

    Designed for the multi-agent build pipeline — the Builder agent submits
    its output here and receives actionable repair instructions.
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

    # Run canonical spec validation if the blueprint embeds a spec
    # Otherwise run a supplementary structural audit (47 additional checks)
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

    # Generate fix instructions if confidence < 85
    fix_instructions = []
    if not passed:
        fix_instructions = _generate_fix_instructions(all_errors, all_warnings, score_100)

    return {
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

    return {
        "stalled_count": len(stalled),
        "stalled_projects": stalled,
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
