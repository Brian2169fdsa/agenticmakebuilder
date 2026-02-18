"""
Build Scenario Pipeline

Single deterministic entrypoint that orchestrates the full scenario build lifecycle:
  1. normalize_to_canonical_spec — plan → canonical spec
  2. validate_canonical_spec — validate the canonical spec
  3. compute_confidence (spec) — score the canonical spec
  4. generate_make_export — canonical spec → Make.com blueprint
  5. validate_make_export — validate the blueprint
  6. self_heal_make_export — repair structural errors (max 2 retries)
  7. save_versioned_spec — write artifacts to /output/<slug>/vN/
  8. write make_export.json — save blueprint alongside other artifacts
  9. generate delivery summary

Input:
    plan (dict) — structured plan from agent reasoning
    registry (dict) — loaded module registry
    original_request (str) — user's natural language request
    base_output_dir (str, optional) — override output directory

Output:
    dict with:
        - success: bool
        - slug: str
        - version: int
        - output_path: str
        - confidence: dict (score, grade, explanation)
        - canonical_validation: dict (checks_run, checks_passed, errors, warnings)
        - export_validation: dict (checks_run, checks_passed, errors, warnings)
        - heal_result: dict | None (retries_used, repair_log)
        - delivery_summary: str (human-readable markdown)
        - failure_reason: str | None

Pipeline stops on:
    - Canonical spec validation failure (errors > 0) with confidence < 0.70
    - Non-healable export errors after max retries
    - Any unexpected exception (caught and reported)

Deterministic. No network calls. No AI reasoning. Deep-copy safety.
"""

import copy
import json
import os
from datetime import datetime, timezone

from tools.normalize_to_canonical_spec import normalize_to_canonical_spec
from tools.validate_canonical_spec import validate_canonical_spec
from tools.confidence_scorer import compute_confidence
from tools.generate_make_export import generate_make_export
from tools.validate_make_export import validate_make_export
from tools.self_heal_make_export import self_heal_make_export
from tools.spec_version_manager import save_versioned_spec
from tools.timeline_estimator import estimate_timeline
from tools.cost_estimator import estimate_cost
from tools.delivery_packager import package_delivery
from tools.blueprint_generator import generate_blueprint
from db.repo import create_build, store_artifact, finalize_build


# Minimum confidence to proceed past canonical spec validation
MIN_CONFIDENCE = 0.70


def build_scenario_pipeline(plan, registry, original_request,
                            base_output_dir=None, created_at=None,
                            db_session=None, project_name="default"):
    """Orchestrate the full scenario build lifecycle.

    Args:
        plan: Structured plan dict from agent reasoning.
        registry: Loaded module registry from module_registry_loader.
        original_request: User's original natural language request.
        base_output_dir: Optional override for output directory (filesystem mode).
        created_at: Optional ISO 8601 timestamp (for deterministic testing).
        db_session: Optional SQLAlchemy Session. When provided, all artifacts
            are persisted to PostgreSQL instead of the filesystem.
        project_name: Project name for DB grouping (default: "default").

    Returns:
        dict with success, slug, version, output_path, confidence,
        canonical_validation, export_validation, heal_result,
        delivery_summary, failure_reason.
    """
    plan = copy.deepcopy(plan)
    result = _empty_result()
    build_id = None
    heal_attempts = 0

    try:
        # === Phase 1: Normalize ===
        spec = normalize_to_canonical_spec(
            plan, registry, original_request, created_at=created_at
        )
        slug = spec.get("scenario", {}).get("slug", "untitled")
        result["slug"] = slug

        # Create DB build record early (tracks all builds including failures)
        if db_session:
            build_obj = create_build(
                db_session, project_name, slug, original_request,
                created_at=created_at,
            )
            build_id = build_obj.id
            version = build_obj.version

        # === Phase 1: Validate canonical spec ===
        spec_report = validate_canonical_spec(spec, registry)
        result["canonical_validation"] = _summarize_validation(spec_report)

        # === Phase 1: Confidence scoring ===
        agent_notes = spec.get("metadata", {}).get("agent_notes", [])
        confidence = compute_confidence(spec_report, agent_notes=agent_notes, retry_count=0)
        result["confidence"] = {
            "score": confidence["score"],
            "grade": confidence["grade"],
            "explanation": confidence["explanation"]
        }

        # Stop if canonical spec has errors and confidence is below threshold
        if not spec_report["valid"] and confidence["score"] < MIN_CONFIDENCE:
            result["failure_reason"] = (
                f"Canonical spec validation failed with {len(spec_report['errors'])} error(s) "
                f"and confidence {confidence['score']} < {MIN_CONFIDENCE}. "
                f"Errors: {[e['message'] for e in spec_report['errors'][:5]]}"
            )
            if db_session and build_id:
                store_artifact(db_session, build_id, "canonical_spec", content_json=spec)
                store_artifact(db_session, build_id, "validation_report", content_json=spec_report)
                store_artifact(db_session, build_id, "confidence", content_json=confidence)
                finalize_build(
                    db_session, build_id, "failed",
                    confidence_score=confidence["score"],
                    confidence_grade=confidence["grade"],
                    canonical_valid=False,
                    failure_reason=result["failure_reason"],
                )
            result["delivery_summary"] = _build_failure_summary(result)
            return result

        # === Phase 2: Generate Make.com blueprint ===
        blueprint = generate_make_export(spec)

        # === Phase 3: Validate Make.com export ===
        export_report = validate_make_export(blueprint, registry)
        result["export_validation"] = _summarize_validation(export_report)

        # === Phase 4: Self-heal if needed ===
        if not export_report["valid"]:
            heal = self_heal_make_export(blueprint, registry, max_retries=2)
            result["heal_result"] = {
                "retries_used": heal["retries_used"],
                "repairs": len(heal["repair_log"]),
                "repair_log": heal["repair_log"]
            }
            blueprint = heal["blueprint"]
            export_report = heal["final_validation"]
            result["export_validation"] = _summarize_validation(export_report)
            heal_attempts = heal["retries_used"]

            if not heal["success"]:
                remaining_errors = [e["message"] for e in export_report.get("errors", [])[:5]]
                result["failure_reason"] = (
                    f"Make export validation failed after {heal['retries_used']} repair attempt(s). "
                    f"Remaining errors: {remaining_errors}"
                )
                if db_session and build_id:
                    store_artifact(db_session, build_id, "canonical_spec", content_json=spec)
                    store_artifact(db_session, build_id, "validation_report", content_json=spec_report)
                    store_artifact(db_session, build_id, "confidence", content_json=confidence)
                    store_artifact(db_session, build_id, "make_export", content_json=blueprint)
                    store_artifact(db_session, build_id, "export_validation_report", content_json=export_report)
                    finalize_build(
                        db_session, build_id, "failed",
                        confidence_score=confidence["score"],
                        confidence_grade=confidence["grade"],
                        canonical_valid=spec_report.get("valid", False),
                        export_valid=False,
                        heal_attempts=heal_attempts,
                        failure_reason=result["failure_reason"],
                    )
                result["delivery_summary"] = _build_failure_summary(result)
                return result

        # === Phase 5: Persist Artifacts ===
        if db_session:
            # --- DB persistence path ---
            build_log_md = _generate_build_log(
                slug, version, spec, spec_report, confidence
            )
            store_artifact(db_session, build_id, "canonical_spec", content_json=spec)
            store_artifact(db_session, build_id, "validation_report", content_json=spec_report)
            store_artifact(db_session, build_id, "confidence", content_json=confidence)
            store_artifact(db_session, build_id, "build_log", content_text=build_log_md)
            store_artifact(db_session, build_id, "make_export", content_json=blueprint)
            store_artifact(db_session, build_id, "export_validation_report", content_json=export_report)

            timeline = estimate_timeline(spec)
            store_artifact(db_session, build_id, "timeline", content_json=timeline)

            cost = estimate_cost(spec)
            store_artifact(db_session, build_id, "cost_estimate", content_json=cost)

            delivery = package_delivery(spec, result, timeline, cost)
            store_artifact(db_session, build_id, "customer_delivery_summary",
                           content_text=delivery["summary_md"])
            store_artifact(db_session, build_id, "delivery_pack",
                           content_json=delivery["pack_json"])

            finalize_build(
                db_session, build_id, "success",
                confidence_score=confidence["score"],
                confidence_grade=confidence["grade"],
                canonical_valid=spec_report.get("valid", False),
                export_valid=export_report.get("valid", False),
                heal_attempts=heal_attempts,
            )

            result["version"] = version
            result["output_path"] = f"builds/{slug}/v{version}"
            result["success"] = True
            result["delivery_summary"] = _build_success_summary(
                result, spec, blueprint, confidence, timeline, cost
            )
        else:
            # --- Filesystem persistence path (backward compat / tests) ---
            version_result = save_versioned_spec(
                slug=slug,
                canonical_spec=spec,
                validation_report=spec_report,
                confidence=confidence,
                base_output_dir=base_output_dir
            )
            result["version"] = version_result["version"]
            result["output_path"] = version_result["output_path"]

            export_path = os.path.join(version_result["output_path"], "make_export.json")
            with open(export_path, "w") as f:
                json.dump(blueprint, f, indent=2, default=str)

            export_report_path = os.path.join(version_result["output_path"], "export_validation_report.json")
            with open(export_report_path, "w") as f:
                json.dump(export_report, f, indent=2, default=str)

            timeline = estimate_timeline(spec)
            timeline_path = os.path.join(version_result["output_path"], "timeline.json")
            with open(timeline_path, "w") as f:
                json.dump(timeline, f, indent=2, default=str)

            cost = estimate_cost(spec)
            cost_path = os.path.join(version_result["output_path"], "cost_estimate.json")
            with open(cost_path, "w") as f:
                json.dump(cost, f, indent=2, default=str)

            delivery = package_delivery(spec, result, timeline, cost)

            summary_path = os.path.join(version_result["output_path"], "customer_delivery_summary.md")
            with open(summary_path, "w") as f:
                f.write(delivery["summary_md"])

            pack_path = os.path.join(version_result["output_path"], "delivery_pack.json")
            with open(pack_path, "w") as f:
                json.dump(delivery["pack_json"], f, indent=2, default=str)

            generate_blueprint(
                output_path=version_result["output_path"],
                customer_name=spec.get("scenario_name", "Customer").split(":")[0].strip(),
            )

            result["success"] = True
            result["delivery_summary"] = _build_success_summary(
                result, spec, blueprint, confidence, timeline, cost
            )

    except Exception as e:
        if db_session and build_id:
            try:
                finalize_build(
                    db_session, build_id, "failed",
                    failure_reason=f"Pipeline exception: {type(e).__name__}: {str(e)}",
                )
            except Exception:
                pass  # Don't mask the original exception
        result["failure_reason"] = f"Pipeline exception: {type(e).__name__}: {str(e)}"
        result["delivery_summary"] = _build_failure_summary(result)

    return result


def _empty_result():
    """Create an empty pipeline result dict."""
    return {
        "success": False,
        "slug": None,
        "version": None,
        "output_path": None,
        "confidence": None,
        "canonical_validation": None,
        "export_validation": None,
        "heal_result": None,
        "delivery_summary": None,
        "failure_reason": None
    }


def _summarize_validation(report):
    """Extract a summary from a validation report."""
    return {
        "valid": report.get("valid", False),
        "checks_run": report.get("checks_run", 0),
        "checks_passed": report.get("checks_passed", 0),
        "errors": len(report.get("errors", [])),
        "warnings": len(report.get("warnings", [])),
        "error_details": [
            {"rule_id": e["rule_id"], "message": e["message"]}
            for e in report.get("errors", [])
        ]
    }


def _build_success_summary(result, spec, blueprint, confidence, timeline=None, cost=None):
    """Generate a human-readable delivery summary for a successful build."""
    scenario = spec.get("scenario", {})
    modules = spec.get("modules", [])
    trigger = spec.get("trigger", {})
    heal = result.get("heal_result")

    lines = [
        f"# Build Complete: {scenario.get('name', result['slug'])}",
        "",
        f"**Slug:** {result['slug']}",
        f"**Version:** v{result['version']}",
        f"**Output:** `{result['output_path']}`",
        "",
        "## Pipeline Results",
        "",
        f"| Phase | Status |",
        f"|-------|--------|",
        f"| Canonical Spec | {_status_badge(result['canonical_validation'])} |",
        f"| Make Export | {_status_badge(result['export_validation'])} |",
    ]

    if heal:
        lines.append(f"| Self-Heal | {heal['repairs']} repair(s) in {heal['retries_used']} pass(es) |")

    lines.extend([
        f"| Confidence | {confidence['score']} (Grade {confidence['grade']}) |",
        "",
        "## Scenario Structure",
        "",
        f"- **Trigger:** {trigger.get('label', 'N/A')} (`{trigger.get('module', 'N/A')}`)",
        f"- **Modules:** {len(modules)}",
        f"- **Connections:** {len(spec.get('connections', []))}",
        "",
    ])

    if modules:
        lines.append("### Modules")
        lines.append("")
        for mod in modules:
            lines.append(f"- [{mod['id']}] {mod['label']} (`{mod['module']}`)")
        lines.append("")

    if timeline:
        lines.extend([
            "## Timeline Estimate",
            "",
            f"- **Total:** {timeline.get('total_hours', '?')} hours",
            f"- **Complexity:** {timeline.get('complexity_grade', '?').title()}",
            "",
        ])

    if cost:
        lines.extend([
            "## Cost Estimate",
            "",
            f"- **Ops/execution:** {cost.get('ops_per_execution', '?')}",
            f"- **Ops/month:** {cost.get('ops_per_month', '?'):,} (at {cost.get('executions_per_month', '?'):,} exec/mo)",
            f"- **Plan:** Make.com {cost.get('recommended_plan', {}).get('name', '?')} (${cost.get('monthly_operational_cost', 0):.2f}/mo)",
            "",
        ])

    lines.extend([
        "## Artifacts Written",
        "",
        "- `canonical_spec.json`",
        "- `make_export.json`",
        "- `validation_report.json`",
        "- `export_validation_report.json`",
        "- `confidence.json`",
        "- `build_log.md`",
        "- `timeline.json`",
        "- `cost_estimate.json`",
        "- `customer_delivery_summary.md`",
        "- `delivery_pack.json`",
        "- `build_blueprint.md`",
        "",
        f"## Confidence",
        "",
        f"**{confidence['explanation']}**",
    ])

    agent_notes = spec.get("metadata", {}).get("agent_notes", [])
    if agent_notes:
        lines.extend(["", "## Agent Notes", ""])
        for note in agent_notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def _build_failure_summary(result):
    """Generate a human-readable failure summary."""
    lines = [
        f"# Build Failed: {result.get('slug', 'unknown')}",
        "",
        f"**Reason:** {result.get('failure_reason', 'Unknown error')}",
        "",
    ]

    if result.get("canonical_validation"):
        cv = result["canonical_validation"]
        lines.extend([
            "## Canonical Spec Validation",
            "",
            f"- Checks: {cv['checks_passed']}/{cv['checks_run']}",
            f"- Errors: {cv['errors']}",
            f"- Warnings: {cv['warnings']}",
        ])
        if cv.get("error_details"):
            lines.append("")
            for ed in cv["error_details"][:10]:
                lines.append(f"  - **{ed['rule_id']}**: {ed['message']}")
        lines.append("")

    if result.get("export_validation"):
        ev = result["export_validation"]
        lines.extend([
            "## Make Export Validation",
            "",
            f"- Checks: {ev['checks_passed']}/{ev['checks_run']}",
            f"- Errors: {ev['errors']}",
            f"- Warnings: {ev['warnings']}",
        ])
        if ev.get("error_details"):
            lines.append("")
            for ed in ev["error_details"][:10]:
                lines.append(f"  - **{ed['rule_id']}**: {ed['message']}")
        lines.append("")

    if result.get("confidence"):
        lines.extend([
            "## Confidence",
            "",
            f"- Score: {result['confidence']['score']}",
            f"- Grade: {result['confidence']['grade']}",
            f"- {result['confidence']['explanation']}",
        ])

    return "\n".join(lines)


def _status_badge(validation_summary):
    """Format a validation summary as a status string."""
    if validation_summary is None:
        return "N/A"
    if validation_summary["valid"]:
        return f"PASS ({validation_summary['checks_passed']}/{validation_summary['checks_run']})"
    return f"FAIL ({validation_summary['checks_passed']}/{validation_summary['checks_run']}, {validation_summary['errors']} error(s))"


def _generate_build_log(slug, version, canonical_spec, validation_report,
                        confidence):
    """Generate a human-readable build log in Markdown (pure function)."""
    scenario = canonical_spec.get("scenario", {})
    metadata = canonical_spec.get("metadata", {})
    modules = canonical_spec.get("modules", [])
    trigger = canonical_spec.get("trigger", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# Build Log: {scenario.get('name', slug)}",
        "",
        f"**Slug:** {slug}",
        f"**Version:** {version}",
        f"**Built:** {now}",
        "",
        "## Original Request",
        "",
        f"> {metadata.get('original_request', 'N/A')}",
        "",
        "## Scenario Structure",
        "",
        f"- **Trigger:** {trigger.get('label', 'N/A')} ({trigger.get('module', 'N/A')})",
        f"- **Modules:** {len(modules)}",
        f"- **Connections:** {len(canonical_spec.get('connections', []))}",
        "",
    ]

    if modules:
        lines.append("### Modules")
        lines.append("")
        for mod in modules:
            lines.append(
                f"- [{mod.get('id')}] {mod.get('label')} ({mod.get('module')})"
            )
        lines.append("")

    lines.extend([
        "## Validation",
        "",
        f"- **Checks Run:** {validation_report.get('checks_run', 0)}",
        f"- **Checks Passed:** {validation_report.get('checks_passed', 0)}",
        f"- **Errors:** {len(validation_report.get('errors', []))}",
        f"- **Warnings:** {len(validation_report.get('warnings', []))}",
        "",
        "## Confidence",
        "",
        f"- **Score:** {confidence.get('score', 0.0)}",
        f"- **Grade:** {confidence.get('grade', 'N/A')}",
        f"- **{confidence.get('explanation', '')}**",
        "",
    ])

    notes = metadata.get("agent_notes", [])
    if notes:
        lines.append("## Agent Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


# --- Self-check ---
if __name__ == "__main__":
    import shutil
    import tempfile
    from tools.module_registry_loader import load_module_registry

    print("=== Build Scenario Pipeline Self-Check ===\n")

    registry = load_module_registry()
    fixed_ts = "2026-02-16T12:00:00Z"

    # --- Test 1: Successful linear build ---
    print("Test 1: Successful linear build (webhook → parse → slack)")
    test_dir = tempfile.mkdtemp(prefix="amb_pipeline_test_")

    try:
        plan1 = {
            "scenario_name": "Form to Slack",
            "scenario_description": "Receives form data via webhook, posts to Slack.",
            "trigger": {
                "type": "webhook",
                "module": "gateway:CustomWebHook",
                "label": "Receive Form",
                "parameters": {"hook": "__WEBHOOK_ID__"}
            },
            "steps": [
                {
                    "module": "json:ParseJSON",
                    "label": "Parse Payload",
                    "mapper": {"json": "{{1.body}}"}
                },
                {
                    "module": "slack:PostMessage",
                    "label": "Notify Slack",
                    "mapper": {"channel": "#general", "text": "New: {{2.name}}"}
                }
            ],
            "error_handling": {"default_strategy": "ignore"},
            "agent_notes": ["Assumed webhook payload has name field"]
        }

        r1 = build_scenario_pipeline(
            plan1, registry, "Post form data to Slack",
            base_output_dir=test_dir, created_at=fixed_ts
        )

        assert r1["success"] is True, f"Should succeed, got: {r1['failure_reason']}"
        assert r1["slug"] == "form-to-slack"
        assert r1["version"] == 1
        assert r1["output_path"] is not None
        assert r1["confidence"]["grade"] in ("A", "B")
        assert r1["canonical_validation"]["valid"] is True
        assert r1["export_validation"]["valid"] is True
        assert r1["failure_reason"] is None

        vdir = r1["output_path"]
        expected_files = [
            "canonical_spec.json", "make_export.json",
            "validation_report.json", "export_validation_report.json",
            "confidence.json", "build_log.md",
            "timeline.json", "cost_estimate.json",
            "customer_delivery_summary.md", "delivery_pack.json",
            "build_blueprint.md"
        ]
        for fname in expected_files:
            assert os.path.isfile(os.path.join(vdir, fname)), f"Missing: {fname}"

        with open(os.path.join(vdir, "make_export.json"), "r") as f:
            bp = json.load(f)
        assert "flow" in bp
        assert bp["name"] == "Form to Slack"

        with open(os.path.join(vdir, "timeline.json"), "r") as f:
            tl = json.load(f)
        assert tl["total_hours"] > 0
        assert "complexity_grade" in tl

        with open(os.path.join(vdir, "cost_estimate.json"), "r") as f:
            ce = json.load(f)
        assert ce["ops_per_execution"] > 0

        with open(os.path.join(vdir, "delivery_pack.json"), "r") as f:
            dp = json.load(f)
        assert dp["delivery_version"] == "1.0.0"
        assert dp["scenario"]["name"] == "Form to Slack"

        assert os.path.getsize(os.path.join(vdir, "customer_delivery_summary.md")) > 100
        assert os.path.getsize(os.path.join(vdir, "build_blueprint.md")) > 100

        print(f"  Success: {r1['success']}")
        print(f"  Slug: {r1['slug']}, Version: v{r1['version']}")
        print(f"  Confidence: {r1['confidence']['score']} ({r1['confidence']['grade']})")
        print(f"  Spec: {r1['canonical_validation']['checks_passed']}/{r1['canonical_validation']['checks_run']}")
        print(f"  Export: {r1['export_validation']['checks_passed']}/{r1['export_validation']['checks_run']}")
        print(f"  Artifacts: {len(expected_files)} files in {vdir}")
        print("  [OK] Linear build passes with all delivery artifacts including blueprint")

        print("\nTest 2: Successful router build")
        plan2 = {
            "scenario_name": "Priority Router",
            "scenario_description": "Routes by priority to Slack or Sheets.",
            "trigger": {
                "type": "webhook",
                "module": "gateway:CustomWebHook",
                "label": "Webhook",
                "parameters": {"hook": "__WEBHOOK_ID__"}
            },
            "steps": [
                {
                    "module": "builtin:BasicRouter",
                    "label": "Route by Priority"
                },
                {
                    "module": "slack:PostMessage",
                    "label": "Alert Slack",
                    "mapper": {"channel": "#urgent", "text": "ALERT: {{1.title}}"}
                },
                {
                    "module": "google-sheets:addRow",
                    "label": "Log to Sheets",
                    "parameters": {"spreadsheetId": "__SID__", "sheetId": "__SHID__"},
                    "mapper": {"values": ["{{1.data}}"]}
                }
            ],
            "connections": [
                {"from": "trigger", "to": 0},
                {
                    "from": 0, "to": 1,
                    "filter": {
                        "name": "High Priority",
                        "conditions": [[{"a": "{{1.priority}}", "b": "high", "o": "text:equal"}]]
                    }
                },
                {"from": 0, "to": 2}
            ],
            "error_handling": {"default_strategy": "ignore", "max_errors": 5}
        }

        r2 = build_scenario_pipeline(
            plan2, registry, "Route by priority",
            base_output_dir=test_dir, created_at=fixed_ts
        )

        assert r2["success"] is True, f"Router build should succeed, got: {r2['failure_reason']}"
        assert r2["slug"] == "priority-router"
        assert r2["version"] == 1
        assert r2["export_validation"]["valid"] is True

        print(f"  Success: {r2['success']}")
        print(f"  Slug: {r2['slug']}, Version: v{r2['version']}")
        print(f"  Confidence: {r2['confidence']['score']} ({r2['confidence']['grade']})")
        print("  [OK] Router build passes")

        print("\nTest 3: Version increment (same slug, build again)")
        r3 = build_scenario_pipeline(
            plan1, registry, "Post form data to Slack v2",
            base_output_dir=test_dir, created_at=fixed_ts
        )

        assert r3["success"] is True
        assert r3["version"] == 2, f"Expected v2, got v{r3['version']}"
        print(f"  Version: v{r3['version']}")
        print("  [OK] Version auto-incremented")

        print("\nTest 4: Canonical validation failure")
        bad_plan = {
            "scenario_name": "Bad Plan",
            "scenario_description": "Uses a fake module.",
            "trigger": {
                "type": "webhook",
                "module": "gateway:CustomWebHook",
                "label": "WH",
                "parameters": {"hook": "__WEBHOOK_ID__"}
            },
            "steps": [
                {
                    "module": "nonexistent:FakeModule",
                    "label": "Fake"
                }
            ],
            "error_handling": {"default_strategy": "ignore"}
        }

        r4 = build_scenario_pipeline(
            bad_plan, registry, "This should fail",
            base_output_dir=test_dir, created_at=fixed_ts
        )

        assert r4["success"] is False
        assert r4["failure_reason"] is not None
        assert "Canonical spec validation failed" in r4["failure_reason"]
        assert r4["version"] is None
        assert r4["output_path"] is None
        print(f"  Success: {r4['success']}")
        print(f"  Reason: {r4['failure_reason'][:80]}...")
        print("  [OK] Pipeline stopped on canonical validation failure")

        print("\nTest 5: Input not mutated")
        plan_copy = copy.deepcopy(plan1)
        _ = build_scenario_pipeline(
            plan1, registry, "Mutation test",
            base_output_dir=test_dir, created_at=fixed_ts
        )
        assert plan1 == plan_copy, "Input plan was mutated!"
        print("  [OK] Original plan unchanged")

        print("\nTest 6: Determinism")
        det_dir1 = tempfile.mkdtemp(prefix="amb_det1_")
        det_dir2 = tempfile.mkdtemp(prefix="amb_det2_")

        r6a = build_scenario_pipeline(
            plan1, registry, "Determinism test",
            base_output_dir=det_dir1, created_at=fixed_ts
        )
        r6b = build_scenario_pipeline(
            plan1, registry, "Determinism test",
            base_output_dir=det_dir2, created_at=fixed_ts
        )

        assert r6a["success"] == r6b["success"]
        assert r6a["slug"] == r6b["slug"]
        assert r6a["confidence"] == r6b["confidence"]
        assert r6a["canonical_validation"] == r6b["canonical_validation"]
        assert r6a["export_validation"] == r6b["export_validation"]

        with open(os.path.join(r6a["output_path"], "make_export.json")) as f:
            bp_a = json.load(f)
        with open(os.path.join(r6b["output_path"], "make_export.json")) as f:
            bp_b = json.load(f)
        assert bp_a == bp_b

        shutil.rmtree(det_dir1)
        shutil.rmtree(det_dir2)
        print("  [OK] Deterministic (same plan → same output)")

        print("\nTest 7: Delivery summary")
        summary = r1["delivery_summary"]
        assert "# Build Complete" in summary
        assert "form-to-slack" in summary
        assert "PASS" in summary
        assert "canonical_spec.json" in summary
        assert "make_export.json" in summary
        assert "timeline.json" in summary
        assert "cost_estimate.json" in summary
        assert "customer_delivery_summary.md" in summary
        assert "delivery_pack.json" in summary
        assert "build_blueprint.md" in summary
        assert "Timeline Estimate" in summary
        assert "Cost Estimate" in summary
        print("  [OK] Success summary contains all expected sections")

        summary_fail = r4["delivery_summary"]
        assert "# Build Failed" in summary_fail
        assert r4["failure_reason"][:30] in summary_fail
        print("  [OK] Failure summary contains reason")

        print("\nTest 8: Global index")
        with open(os.path.join(test_dir, "index.json")) as f:
            global_idx = json.load(f)
        assert global_idx["total_scenarios"] >= 2
        assert "form-to-slack" in global_idx["scenarios"]
        assert "priority-router" in global_idx["scenarios"]
        print(f"  Scenarios: {global_idx['total_scenarios']}, Versions: {global_idx['total_versions']}")
        print("  [OK] Global index tracks all builds")

    finally:
        shutil.rmtree(test_dir)

    print(f"\n=== All build_scenario_pipeline checks passed ===")