"""
Delivery Packager

Generates customer-facing delivery artifacts:
  - customer_delivery_summary.md — human-readable delivery document
  - delivery_pack.json — machine-readable delivery package

Input:
    spec (dict) — canonical spec
    pipeline_result (dict) — result from build_scenario_pipeline
    timeline (dict) — result from timeline_estimator
    cost (dict) — result from cost_estimator

Output:
    dict with:
        - summary_md: str — markdown delivery summary
        - pack_json: dict — structured delivery package

Deterministic. No network calls. No AI reasoning.
"""

from datetime import datetime, timezone


def package_delivery(spec, pipeline_result, timeline, cost):
    """Generate customer delivery artifacts.

    Args:
        spec: Canonical spec dict.
        pipeline_result: Result from build_scenario_pipeline.
        timeline: Result from timeline_estimator.
        cost: Result from cost_estimator.

    Returns:
        dict with summary_md (str) and pack_json (dict).
    """
    summary_md = _build_summary_md(spec, pipeline_result, timeline, cost)
    pack_json = _build_pack_json(spec, pipeline_result, timeline, cost)

    return {
        "summary_md": summary_md,
        "pack_json": pack_json
    }


def _build_summary_md(spec, pipeline_result, timeline, cost):
    """Generate customer-facing markdown delivery summary."""
    scenario = spec.get("scenario", {})
    trigger = spec.get("trigger", {})
    modules = spec.get("modules", [])
    connections = spec.get("connections", [])
    metadata = spec.get("metadata", {})
    confidence = pipeline_result.get("confidence", {})

    lines = [
        f"# Scenario Delivery: {scenario.get('name', 'Untitled')}",
        "",
        f"> {scenario.get('description', '')}",
        "",
        "---",
        "",
        "## Overview",
        "",
        f"| Item | Detail |",
        f"|------|--------|",
        f"| Scenario | {scenario.get('name', 'Untitled')} |",
        f"| Slug | `{scenario.get('slug', 'unknown')}` |",
        f"| Version | v{pipeline_result.get('version', '?')} |",
        f"| Trigger | {trigger.get('label', 'N/A')} (`{trigger.get('module', 'N/A')}`) |",
        f"| Modules | {len(modules)} |",
        f"| Connections | {len(connections)} |",
        f"| Confidence | {confidence.get('score', '?')} (Grade {confidence.get('grade', '?')}) |",
        "",
        "---",
        "",
        "## Workflow Structure",
        "",
    ]

    # Trigger
    lines.append(f"**Trigger:** {trigger.get('label', 'N/A')}")
    trigger_type = trigger.get("type", "unknown")
    lines.append(f"- Type: {trigger_type}")
    if trigger_type == "webhook":
        lines.append("- Activated by incoming HTTP request")
    lines.append("")

    # Modules
    if modules:
        lines.append("**Processing Steps:**")
        lines.append("")
        for mod in modules:
            cred = ""
            if mod.get("credential_placeholder"):
                cred = f" (requires {mod['app']} authentication)"
            lines.append(f"{mod['id']}. **{mod['label']}** — `{mod['module']}`{cred}")
        lines.append("")

    # Routing
    routers = [m for m in modules if m.get("module_type") == "flow_control"]
    if routers:
        lines.append("**Routing:**")
        lines.append("")
        for r in routers:
            outgoing = [c for c in connections if c.get("from") == r["id"]]
            lines.append(f"- Router [{r['id']}] {r['label']}: {len(outgoing)} route(s)")
            for c in outgoing:
                target = next((m for m in modules if m["id"] == c["to"]), None)
                target_label = target["label"] if target else f"Module {c['to']}"
                filt = c.get("filter")
                if filt:
                    lines.append(f"  - → {target_label} (filter: {filt.get('name', 'unnamed')})")
                else:
                    lines.append(f"  - → {target_label} (fallback/all)")
        lines.append("")

    # Timeline
    lines.extend([
        "---",
        "",
        "## Implementation Timeline",
        "",
        f"**Estimated Total:** {timeline.get('total_hours', '?')} hours",
        f"**Complexity:** {timeline.get('complexity_grade', '?').title()}",
        "",
        "| Phase | Hours |",
        "|-------|-------|",
    ])

    for phase, hours in timeline.get("phases", {}).items():
        phase_label = phase.replace("_", " ").title()
        lines.append(f"| {phase_label} | {hours} |")

    lines.append("")

    # Cost
    lines.extend([
        "---",
        "",
        "## Cost Estimate",
        "",
        f"**Operations per execution:** {cost.get('ops_per_execution', '?')}",
        f"**Estimated executions/month:** {cost.get('executions_per_month', '?'):,}",
        f"**Operations/month:** {cost.get('ops_per_month', '?'):,}",
        "",
        f"**Recommended Plan:** Make.com {cost.get('recommended_plan', {}).get('name', '?')} "
        f"(${cost.get('monthly_operational_cost', 0):.2f}/month)",
        f"**Annual Estimate:** ${cost.get('annual_estimate', 0):.2f}",
        "",
    ])

    # Integration setup
    setup = cost.get("integration_setup", {})
    if setup:
        lines.append("**Integration Setup:**")
        lines.append("")
        for app, app_cost in setup.items():
            lines.append(f"- {app}: ${app_cost:.2f}")
        lines.append("")

    # Credentials needed
    cred_apps = []
    if trigger.get("credential_placeholder"):
        cred_apps.append(trigger.get("module", "trigger"))
    for mod in modules:
        if mod.get("credential_placeholder"):
            cred_apps.append(f"{mod['app']} ({mod['label']})")
    if cred_apps:
        lines.extend([
            "---",
            "",
            "## Credentials Required",
            "",
        ])
        for app in cred_apps:
            lines.append(f"- {app}")
        lines.extend([
            "",
            "*Credential placeholders are used during build. Real credentials must be configured in Make.com after import.*",
            "",
        ])

    # Artifacts
    lines.extend([
        "---",
        "",
        "## Deliverables",
        "",
        f"All artifacts are in: `{pipeline_result.get('output_path', '/output/<slug>/vN/')}`",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `make_export.json` | Make.com blueprint — import directly into Make.com |",
        "| `canonical_spec.json` | Canonical workflow specification |",
        "| `validation_report.json` | Canonical spec validation results |",
        "| `export_validation_report.json` | Blueprint validation results |",
        "| `confidence.json` | Build confidence score and factors |",
        "| `timeline.json` | Implementation timeline estimate |",
        "| `cost_estimate.json` | Cost and operations estimate |",
        "| `build_log.md` | Technical build log |",
        "",
        "---",
        "",
        f"*Generated by Agentic Make Builder*",
    ])

    return "\n".join(lines)


def _build_pack_json(spec, pipeline_result, timeline, cost):
    """Build the machine-readable delivery package."""
    scenario = spec.get("scenario", {})
    trigger = spec.get("trigger", {})
    modules = spec.get("modules", [])

    return {
        "delivery_version": "1.0.0",
        "scenario": {
            "name": scenario.get("name", "Untitled"),
            "slug": scenario.get("slug", "unknown"),
            "description": scenario.get("description", ""),
            "version": pipeline_result.get("version"),
            "output_path": pipeline_result.get("output_path")
        },
        "structure": {
            "trigger": {
                "type": trigger.get("type"),
                "module": trigger.get("module"),
                "label": trigger.get("label")
            },
            "module_count": len(modules),
            "connection_count": len(spec.get("connections", [])),
            "apps_used": sorted(set(
                [m.get("app", "unknown") for m in modules]
                + [trigger.get("module", "").split(":")[0]]
            )),
            "has_routing": any(m.get("module_type") == "flow_control" for m in modules),
            "has_iteration": any(m.get("module_type") == "iterator" for m in modules),
            "credentials_required": sorted(set(
                [m.get("credential_placeholder") for m in modules if m.get("credential_placeholder")]
                + ([trigger.get("credential_placeholder")] if trigger.get("credential_placeholder") else [])
            ))
        },
        "quality": {
            "confidence_score": pipeline_result.get("confidence", {}).get("score"),
            "confidence_grade": pipeline_result.get("confidence", {}).get("grade"),
            "canonical_validation": pipeline_result.get("canonical_validation"),
            "export_validation": pipeline_result.get("export_validation"),
            "self_healed": pipeline_result.get("heal_result") is not None
        },
        "timeline": {
            "total_hours": timeline.get("total_hours"),
            "complexity_grade": timeline.get("complexity_grade"),
            "phases": timeline.get("phases")
        },
        "cost": {
            "ops_per_execution": cost.get("ops_per_execution"),
            "ops_per_month": cost.get("ops_per_month"),
            "recommended_plan": cost.get("recommended_plan", {}).get("name"),
            "monthly_cost": cost.get("monthly_operational_cost"),
            "annual_estimate": cost.get("annual_estimate")
        },
        "artifacts": [
            "make_export.json",
            "canonical_spec.json",
            "validation_report.json",
            "export_validation_report.json",
            "confidence.json",
            "timeline.json",
            "cost_estimate.json",
            "customer_delivery_summary.md",
            "delivery_pack.json",
            "build_log.md"
        ]
    }


# --- Self-check ---
if __name__ == "__main__":
    import json

    print("=== Delivery Packager Self-Check ===\n")

    # Mock inputs
    spec = {
        "scenario": {"name": "Test Scenario", "slug": "test-scenario", "description": "A test."},
        "trigger": {"id": 1, "type": "webhook", "module": "gateway:CustomWebHook",
                     "label": "Webhook", "credential_placeholder": None},
        "modules": [
            {"id": 2, "label": "Parse", "app": "json", "module": "json:ParseJSON",
             "module_type": "transformer", "credential_placeholder": None},
            {"id": 3, "label": "Notify", "app": "slack", "module": "slack:PostMessage",
             "module_type": "action", "credential_placeholder": "__SLACK_CONNECTION__"}
        ],
        "connections": [
            {"from": "trigger", "to": 2, "filter": None},
            {"from": 2, "to": 3, "filter": None}
        ],
        "metadata": {"original_request": "Test.", "agent_notes": []}
    }

    pipeline_result = {
        "success": True,
        "slug": "test-scenario",
        "version": 1,
        "output_path": "/output/test-scenario/v1",
        "confidence": {"score": 0.98, "grade": "A", "explanation": "Good."},
        "canonical_validation": {"valid": True, "checks_run": 48, "checks_passed": 48, "errors": 0, "warnings": 0},
        "export_validation": {"valid": True, "checks_run": 43, "checks_passed": 43, "errors": 0, "warnings": 0},
        "heal_result": None
    }

    timeline = {
        "total_hours": 3.5,
        "complexity_grade": "simple",
        "phases": {"design_and_planning": 0.53, "module_configuration": 1.05,
                   "connection_and_routing": 0.53, "credential_setup": 0.53,
                   "testing_and_validation": 0.7, "documentation": 0.18},
        "notes": ["Base: 1h", "2 modules: +1h"]
    }

    cost = {
        "ops_per_execution": 3,
        "executions_per_month": 100,
        "ops_per_month": 300,
        "integration_setup": {"gateway": 0.0, "json": 0.0, "slack": 0.0},
        "total_setup_cost": 0.0,
        "recommended_plan": {"name": "Free", "ops_per_month": 1000, "monthly_cost": 0.0},
        "monthly_operational_cost": 0.0,
        "annual_estimate": 0.0,
        "notes": ["Recommended: Free"]
    }

    # Test 1: Generate delivery
    print("Test 1: Generate delivery package")
    result = package_delivery(spec, pipeline_result, timeline, cost)

    assert "summary_md" in result
    assert "pack_json" in result
    assert isinstance(result["summary_md"], str)
    assert isinstance(result["pack_json"], dict)
    print(f"  Summary: {len(result['summary_md'])} chars")
    print(f"  Pack keys: {list(result['pack_json'].keys())}")
    print("  [OK]")

    # Test 2: Summary contains key sections
    print("\nTest 2: Summary content")
    md = result["summary_md"]
    assert "# Scenario Delivery" in md
    assert "Test Scenario" in md
    assert "## Workflow Structure" in md
    assert "## Implementation Timeline" in md
    assert "## Cost Estimate" in md
    assert "## Deliverables" in md
    assert "make_export.json" in md
    print("  [OK] All expected sections present")

    # Test 3: Pack JSON structure
    print("\nTest 3: Pack JSON structure")
    pack = result["pack_json"]
    assert pack["delivery_version"] == "1.0.0"
    assert pack["scenario"]["name"] == "Test Scenario"
    assert pack["scenario"]["version"] == 1
    assert pack["structure"]["module_count"] == 2
    assert pack["structure"]["has_routing"] is False
    assert pack["quality"]["confidence_score"] == 0.98
    assert pack["timeline"]["total_hours"] == 3.5
    assert pack["cost"]["ops_per_execution"] == 3
    assert len(pack["artifacts"]) == 10
    print(f"  Delivery version: {pack['delivery_version']}")
    print(f"  Artifacts: {len(pack['artifacts'])}")
    print("  [OK]")

    # Test 4: JSON serializable
    print("\nTest 4: JSON serializable")
    json_str = json.dumps(result["pack_json"], indent=2)
    roundtrip = json.loads(json_str)
    assert roundtrip == result["pack_json"]
    print(f"  Pack JSON: {len(json_str)} bytes")
    print("  [OK]")

    # Test 5: Credentials section appears when needed
    print("\nTest 5: Credentials section")
    assert "Credentials Required" in md
    assert "slack" in md.lower()
    print("  [OK]")

    # Test 6: Determinism
    print("\nTest 6: Determinism")
    r6a = package_delivery(spec, pipeline_result, timeline, cost)
    r6b = package_delivery(spec, pipeline_result, timeline, cost)
    assert r6a["summary_md"] == r6b["summary_md"]
    assert r6a["pack_json"] == r6b["pack_json"]
    print("  [OK]")

    print("\n=== All delivery_packager checks passed ===")
