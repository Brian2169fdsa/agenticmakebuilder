"""
SOW & Proposal Generator

Reads existing pipeline artifacts and renders a complete business document package:
  1. Statement of Work (Markdown → rendered for delivery)
  2. Executive Summary (1-page, non-technical)
  3. Technical Spec (for internal delivery team)
  4. Implementation Checklist (step-by-step deployment)

All data comes from existing artifacts — nothing is generated fresh.
This just renders what the pipeline already knows into client-ready documents.

Usage:
    from tools.sow_generator import generate_sow_package

    result = generate_sow_package(
        artifacts_dir="/path/to/output/acme-form-to-slack/v1",
        customer_name="Acme Corp",
        output_dir="/path/to/output/acme-form-to-slack/v1/sow"
    )

    # Or from DB artifacts dict
    result = generate_sow_package_from_dict(artifacts, customer_name, output_dir)
"""

import json
import os
import re
from datetime import datetime, timezone, timedelta


# ─── Cost and timeline helpers ───

MAKE_PLAN_OPS = {
    "Core": 10_000,
    "Pro": 100_000,
    "Teams": 1_500_000,
    "Enterprise": 10_000_000,
}

HOURLY_RATE = float(os.environ.get("MANAGEAI_HOURLY_RATE", "150"))
SETUP_BASE_HOURS = 4.0
HOURS_PER_STEP = 0.75
HOURS_PER_COMPLEX_STEP = 1.5


def _estimate_delivery_hours(canonical_spec: dict) -> float:
    steps = canonical_spec.get("steps", [])
    hours = SETUP_BASE_HOURS
    for step in steps:
        module = step.get("module", "")
        # Complex modules take longer
        if any(x in module.lower() for x in ["ai", "claude", "openai", "parse", "transform"]):
            hours += HOURS_PER_COMPLEX_STEP
        else:
            hours += HOURS_PER_STEP
    return round(hours, 1)


def _recommend_make_plan(estimated_monthly_ops: int) -> str:
    for plan, ops in MAKE_PLAN_OPS.items():
        if estimated_monthly_ops <= ops:
            return plan
    return "Enterprise"


def _calculate_timeline(hours: float) -> dict:
    now = datetime.now(timezone.utc)
    # 1 day = 6 billable hours
    business_days = max(1, round(hours / 6))
    delivery_date = now + timedelta(days=business_days + 2)  # +2 for review buffer
    return {
        "estimated_hours": hours,
        "estimated_days": business_days,
        "delivery_date": delivery_date.strftime("%B %d, %Y"),
        "start_date": now.strftime("%B %d, %Y"),
    }


# ─── Document renderers ───

def _render_sow(data: dict) -> str:
    spec = data.get("canonical_spec", {})
    timeline = data.get("timeline", {})
    cost = data.get("cost_estimate", {})
    confidence = data.get("confidence", {})
    modules = [s.get("module", "") for s in spec.get("steps", [])]

    risk_level = "Low"
    if isinstance(confidence, dict):
        score = confidence.get("score", 1.0)
        if score < 0.70:
            risk_level = "High"
        elif score < 0.85:
            risk_level = "Medium"

    lines = [
        f"# Statement of Work",
        f"",
        f"**Prepared by:** ManageAI",
        f"**Prepared for:** {data['customer_name']}",
        f"**Date:** {datetime.now().strftime('%B %d, %Y')}",
        f"**Project:** {spec.get('scenario_name', data.get('slug', 'Automation Project'))}",
        f"**Version:** v{data.get('version', 1)}",
        f"",
        f"---",
        f"",
        f"## 1. Project Overview",
        f"",
        f"{spec.get('scenario_description', data.get('original_request', ''))}",
        f"",
        f"**Business Objective:** {data.get('business_objective', spec.get('scenario_description', 'Automate a business workflow to reduce manual effort and improve reliability.'))}",
        f"",
        f"---",
        f"",
        f"## 2. Scope of Work",
        f"",
        f"ManageAI will design, build, test, and deliver the following automation:",
        f"",
        f"**Trigger:** {spec.get('trigger', {}).get('type', 'webhook').title()} — {spec.get('trigger', {}).get('label', 'Incoming event')}",
        f"",
        f"**Automation Steps:**",
        f"",
    ]

    for i, step in enumerate(spec.get("steps", []), 1):
        lines.append(f"{i}. {step.get('label', step.get('module', f'Step {i}'))}")

    lines += [
        f"",
        f"**Error Handling:** {spec.get('error_handling', {}).get('default_strategy', 'ignore').title()} strategy",
        f"",
        f"**Out of Scope:**",
        f"- Changes to existing third-party system configurations",
        f"- Custom API development not listed above",
        f"- Ongoing monitoring beyond the 30-day warranty period",
        f"- Training or change management",
        f"",
        f"---",
        f"",
        f"## 3. Deliverables",
        f"",
        f"Upon completion, {data['customer_name']} will receive:",
        f"",
        f"1. **Working automation scenario** deployed in Make.com",
        f"2. **Blueprint file** (JSON) — portable, version-controlled",
        f"3. **Technical documentation** — architecture, module config, credentials required",
        f"4. **Test results** — validation report confirming the scenario works end-to-end",
        f"5. **Implementation checklist** — step-by-step deployment guide",
        f"6. **30-day warranty** — bug fixes at no additional charge",
        f"",
        f"---",
        f"",
        f"## 4. Timeline",
        f"",
        f"| Milestone | Date |",
        f"|-----------|------|",
        f"| Project Start | {timeline.get('start_date', 'TBD')} |",
        f"| Build Complete | {timeline.get('delivery_date', 'TBD')} |",
        f"| Client Review | +2 business days |",
        f"| Final Delivery | +1 business day |",
        f"",
        f"**Estimated Effort:** {timeline.get('estimated_hours', '?')} hours over {timeline.get('estimated_days', '?')} business day(s)",
        f"",
        f"---",
        f"",
        f"## 5. Investment",
        f"",
        f"| Item | Hours | Rate | Amount |",
        f"|------|-------|------|--------|",
        f"| Scenario Design & Build | {timeline.get('estimated_hours', '?')} | ${HOURLY_RATE:.0f}/hr | ${timeline.get('estimated_hours', 0) * HOURLY_RATE:,.0f} |",
        f"| Testing & QA | 1.0 | ${HOURLY_RATE:.0f}/hr | ${HOURLY_RATE:,.0f} |",
        f"| Documentation | 0.5 | ${HOURLY_RATE:.0f}/hr | ${HOURLY_RATE * 0.5:,.0f} |",
        f"",
        f"**Total Investment:** ${(timeline.get('estimated_hours', 0) + 1.5) * HOURLY_RATE:,.0f}",
        f"",
        f"**Make.com Plan Recommendation:** {cost.get('recommended_plan', 'Pro')} — estimated {cost.get('monthly_ops_estimate', '?')} operations/month",
        f"",
        f"*Payment terms: 50% upon SOW signature, 50% upon delivery.*",
        f"",
        f"---",
        f"",
        f"## 6. Risk Assessment",
        f"",
        f"**Overall Risk Level:** {risk_level}",
        f"",
    ]

    agent_notes = data.get("agent_notes", [])
    if agent_notes:
        lines.append("**Assumptions & Notes:**")
        lines.append("")
        for note in agent_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## 7. Acceptance Criteria",
        f"",
        f"The automation is considered complete when:",
        f"",
        f"1. The trigger fires correctly on the expected input",
        f"2. All steps execute without errors on a test payload",
        f"3. Outputs match the agreed-upon format",
        f"4. The Make.com scenario is active and accessible to {data['customer_name']}",
        f"",
        f"---",
        f"",
        f"## 8. Signatures",
        f"",
        f"By signing below, both parties agree to the scope, timeline, and investment above.",
        f"",
        f"**ManageAI**",
        f"Signature: _______________________  Date: ___________",
        f"",
        f"**{data['customer_name']}**",
        f"Signature: _______________________  Date: ___________",
        f"",
    ]

    return "\n".join(lines)


def _render_exec_summary(data: dict) -> str:
    spec = data.get("canonical_spec", {})
    timeline = data.get("timeline", {})
    cost = data.get("cost_estimate", {})
    total_cost = (timeline.get("estimated_hours", 0) + 1.5) * HOURLY_RATE

    lines = [
        f"# Executive Summary",
        f"**{spec.get('scenario_name', 'Automation Project')}**",
        f"",
        f"Prepared for: **{data['customer_name']}** | {datetime.now().strftime('%B %d, %Y')}",
        f"",
        f"---",
        f"",
        f"## What We're Building",
        f"",
        f"{spec.get('scenario_description', data.get('original_request', ''))}",
        f"",
        f"## The Business Problem",
        f"",
        f"This automation eliminates the need for manual intervention in this workflow. "
        f"Once deployed, it runs automatically every time it is triggered — no human needed.",
        f"",
        f"## What You Get",
        f"",
        f"A fully automated workflow deployed in Make.com that:",
    ]

    for step in spec.get("steps", [])[:5]:
        lines.append(f"- {step.get('label', step.get('module', ''))}")

    lines += [
        f"",
        f"## Investment & Timeline",
        f"",
        f"| | |",
        f"|---|---|",
        f"| **Total Investment** | ${total_cost:,.0f} |",
        f"| **Delivery Time** | {timeline.get('estimated_days', '?')} business day(s) |",
        f"| **Delivery Date** | {timeline.get('delivery_date', 'TBD')} |",
        f"| **Monthly Platform Cost** | Make.com {cost.get('recommended_plan', 'Pro')} plan |",
        f"",
        f"## Next Steps",
        f"",
        f"1. Sign the Statement of Work",
        f"2. ManageAI begins build immediately",
        f"3. You receive working automation + all documentation",
        f"4. 30-day warranty period — any bugs fixed at no charge",
        f"",
        f"---",
        f"*Questions? Contact your ManageAI account manager.*",
    ]

    return "\n".join(lines)


def _render_tech_spec(data: dict) -> str:
    spec = data.get("canonical_spec", {})
    confidence = data.get("confidence", {})

    lines = [
        f"# Technical Specification",
        f"**{spec.get('scenario_name', 'Automation')}** — Internal Reference",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Customer | {data['customer_name']} |",
        f"| Slug | {data.get('slug', '?')} |",
        f"| Version | v{data.get('version', 1)} |",
        f"| Built | {datetime.now().strftime('%Y-%m-%d')} |",
        f"| Confidence | {confidence.get('score', '?')} ({confidence.get('grade', '?')}) |",
        f"",
        f"---",
        f"",
        f"## Trigger",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Type | {spec.get('trigger', {}).get('type', '?')} |",
        f"| Module | `{spec.get('trigger', {}).get('module', '?')}` |",
        f"| Label | {spec.get('trigger', {}).get('label', '?')} |",
        f"",
        f"## Steps",
        f"",
    ]

    for i, step in enumerate(spec.get("steps", []), 1):
        lines += [
            f"### Step {i}: {step.get('label', step.get('module', f'Step {i}'))}",
            f"",
            f"**Module:** `{step.get('module', '?')}`",
            f"",
        ]
        if step.get("mapper"):
            lines.append("**Mappings:**")
            lines.append("```json")
            lines.append(json.dumps(step["mapper"], indent=2))
            lines.append("```")
            lines.append("")
        if step.get("parameters"):
            lines.append("**Parameters:**")
            lines.append("```json")
            lines.append(json.dumps(step["parameters"], indent=2))
            lines.append("```")
            lines.append("")

    lines += [
        f"## Error Handling",
        f"",
        f"Strategy: **{spec.get('error_handling', {}).get('default_strategy', 'ignore')}**",
        f"",
        f"## Agent Notes",
        f"",
    ]
    for note in data.get("agent_notes", []):
        lines.append(f"- {note}")

    if not data.get("agent_notes"):
        lines.append("*No assumptions recorded.*")

    lines += [
        f"",
        f"## Credentials Required",
        f"",
        f"The following connections must be configured in Make.com before activation:",
        f"",
    ]

    # Extract unique credential placeholders from modules
    all_modules = set()
    for step in spec.get("steps", []):
        module = step.get("module", "")
        if module:
            pkg = module.split(":")[0]
            all_modules.add(pkg)

    trigger_module = spec.get("trigger", {}).get("module", "")
    if trigger_module:
        all_modules.add(trigger_module.split(":")[0])

    # Filter out non-credential packages
    skip_pkgs = {"gateway", "json", "tools", "filter", "flow"}
    credential_pkgs = [p for p in sorted(all_modules) if p not in skip_pkgs]
    for pkg in credential_pkgs:
        lines.append(f"- **{pkg.title()}** connection")

    return "\n".join(lines)


def _render_checklist(data: dict) -> str:
    spec = data.get("canonical_spec", {})

    all_modules = set()
    for step in spec.get("steps", []):
        m = step.get("module", "")
        if m:
            all_modules.add(m.split(":")[0])
    trigger_mod = spec.get("trigger", {}).get("module", "")
    if trigger_mod:
        all_modules.add(trigger_mod.split(":")[0])
    skip = {"gateway", "json", "tools", "filter", "flow"}
    cred_pkgs = [p for p in sorted(all_modules) if p not in skip]

    lines = [
        f"# Implementation Checklist",
        f"**{spec.get('scenario_name', 'Automation')}** — Deployment Guide",
        f"",
        f"Follow these steps in order. Check off each item as you complete it.",
        f"",
        f"---",
        f"",
        f"## Phase 1 — Prerequisites",
        f"",
        f"- [ ] Make.com account active with {data.get('cost_estimate', {}).get('recommended_plan', 'Pro')} plan or higher",
        f"- [ ] Admin access to Make.com workspace",
    ]

    for pkg in cred_pkgs:
        lines.append(f"- [ ] **{pkg.title()}** credentials / API key available")

    lines += [
        f"",
        f"---",
        f"",
        f"## Phase 2 — Import Scenario",
        f"",
        f"- [ ] Download `make_export.json` from your delivery package",
        f"- [ ] In Make.com: Scenarios → Create a new scenario → Import Blueprint",
        f"- [ ] Upload `make_export.json`",
        f"- [ ] Verify scenario name matches: **{spec.get('scenario_name', 'Automation')}**",
        f"",
        f"---",
        f"",
        f"## Phase 3 — Configure Connections",
        f"",
    ]

    for pkg in cred_pkgs:
        lines += [
            f"### {pkg.title()}",
            f"- [ ] Open scenario and click the {pkg.title()} module",
            f"- [ ] Click 'Add' next to Connection",
            f"- [ ] Authenticate with your {pkg.title()} credentials",
            f"- [ ] Click 'Save'",
            f"",
        ]

    lines += [
        f"---",
        f"",
        f"## Phase 4 — Test",
        f"",
        f"- [ ] Click 'Run once' in the scenario editor",
        f"- [ ] Trigger the scenario manually (send test webhook / open test record)",
        f"- [ ] Verify each module shows a green checkmark",
        f"- [ ] Confirm output is as expected",
        f"- [ ] Review execution log for any warnings",
        f"",
        f"---",
        f"",
        f"## Phase 5 — Activate",
        f"",
        f"- [ ] Set scheduling / activation as agreed",
        f"- [ ] Toggle scenario to **ON**",
        f"- [ ] Monitor first 5 live executions",
        f"- [ ] Confirm no errors in execution history",
        f"",
        f"---",
        f"",
        f"## Phase 6 — Handoff",
        f"",
        f"- [ ] Share scenario access with {data['customer_name']} team if needed",
        f"- [ ] Archive delivery package to shared drive",
        f"- [ ] Schedule 30-day check-in",
        f"- [ ] Close project ticket",
        f"",
        f"---",
        f"*Built by ManageAI — questions? Contact your account manager.*",
    ]

    return "\n".join(lines)


# ─── Main entry points ───

def generate_sow_package(
    artifacts_dir: str,
    customer_name: str,
    output_dir: str,
    original_request: str = "",
    business_objective: str = "",
) -> dict:
    """
    Generate full SOW package from an artifacts directory.

    Reads: canonical_spec.json, cost_estimate.json, confidence.json,
           delivery_pack.json, timeline.json (if exists)

    Writes to output_dir:
        sow.md, exec_summary.md, tech_spec.md, checklist.md
    """
    # Load all artifacts
    def _load(filename):
        path = os.path.join(artifacts_dir, filename)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return {}

    canonical_spec = _load("canonical_spec.json")
    cost_estimate = _load("cost_estimate.json")
    confidence = _load("confidence.json")
    delivery_pack = _load("delivery_pack.json")

    # Extract slug and version from path
    parts = artifacts_dir.rstrip("/").split("/")
    version_str = parts[-1] if parts[-1].startswith("v") else "v1"
    slug = parts[-2] if len(parts) >= 2 else "scenario"
    version = int(version_str.lstrip("v")) if version_str.lstrip("v").isdigit() else 1

    # Build timeline from cost estimate or canonical spec
    estimated_hours = _estimate_delivery_hours(canonical_spec)
    timeline = _calculate_timeline(estimated_hours)

    # Enrich cost estimate
    steps = canonical_spec.get("steps", [])
    ops_per_run = max(1, len(steps))
    runs_per_month = cost_estimate.get("runs_per_month", 1000)
    monthly_ops = ops_per_run * runs_per_month
    cost_estimate["monthly_ops_estimate"] = monthly_ops
    cost_estimate["recommended_plan"] = _recommend_make_plan(monthly_ops)

    data = {
        "customer_name": customer_name,
        "slug": slug,
        "version": version,
        "canonical_spec": canonical_spec,
        "cost_estimate": cost_estimate,
        "confidence": confidence,
        "timeline": timeline,
        "agent_notes": delivery_pack.get("agent_notes", []),
        "original_request": original_request or delivery_pack.get("original_request", ""),
        "business_objective": business_objective,
    }

    return _write_sow_package(data, output_dir)


def generate_sow_package_from_dict(
    artifacts: dict,
    customer_name: str,
    output_dir: str,
    slug: str = "scenario",
    version: int = 1,
    original_request: str = "",
) -> dict:
    """
    Generate SOW package from in-memory artifacts dict.
    Used by the API endpoint when artifacts are in DB.
    """
    canonical_spec = artifacts.get("canonical_spec", {})
    estimated_hours = _estimate_delivery_hours(canonical_spec)
    timeline = _calculate_timeline(estimated_hours)

    cost_estimate = artifacts.get("cost_estimate", {})
    steps = canonical_spec.get("steps", [])
    ops_per_run = max(1, len(steps))
    monthly_ops = ops_per_run * cost_estimate.get("runs_per_month", 1000)
    cost_estimate["monthly_ops_estimate"] = monthly_ops
    cost_estimate["recommended_plan"] = _recommend_make_plan(monthly_ops)

    delivery_pack = artifacts.get("delivery_pack", {})

    data = {
        "customer_name": customer_name,
        "slug": slug,
        "version": version,
        "canonical_spec": canonical_spec,
        "cost_estimate": cost_estimate,
        "confidence": artifacts.get("confidence", {}),
        "timeline": timeline,
        "agent_notes": delivery_pack.get("agent_notes", []),
        "original_request": original_request or delivery_pack.get("original_request", ""),
        "business_objective": "",
    }

    return _write_sow_package(data, output_dir)


def _write_sow_package(data: dict, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "sow.md": _render_sow(data),
        "exec_summary.md": _render_exec_summary(data),
        "tech_spec.md": _render_tech_spec(data),
        "checklist.md": _render_checklist(data),
    }

    written = {}
    for filename, content in files.items():
        path = os.path.join(output_dir, filename)
        with open(path, "w") as f:
            f.write(content)
        written[filename] = path

    return {
        "success": True,
        "customer_name": data["customer_name"],
        "slug": data["slug"],
        "version": data["version"],
        "output_dir": output_dir,
        "files": written,
        "timeline": data["timeline"],
        "total_investment": (data["timeline"].get("estimated_hours", 0) + 1.5) * HOURLY_RATE,
        "recommended_plan": data["cost_estimate"].get("recommended_plan", "Pro"),
    }


if __name__ == "__main__":
    import tempfile, shutil
    print("=== SOW Generator Self-Check ===\n")
    test_dir = tempfile.mkdtemp(prefix="amb_sow_test_")
    artifacts_dir = os.path.join(test_dir, "acme-form-to-slack", "v1")
    os.makedirs(artifacts_dir)
    output_dir = os.path.join(artifacts_dir, "sow")

    # Write mock artifacts
    mock_spec = {
        "scenario_name": "Acme — Form to Slack Notifier",
        "scenario_description": "When a form is submitted, parse the data and post a Slack notification.",
        "trigger": {"type": "webhook", "module": "gateway:CustomWebHook", "label": "Form Submitted"},
        "steps": [
            {"module": "json:ParseJSON", "label": "Parse Form Data", "mapper": {}, "parameters": {}},
            {"module": "slack:PostMessage", "label": "Notify Slack Channel", "mapper": {"text": "{{1.data}}"}, "parameters": {"connection": "__SLACK_CONNECTION__"}},
        ],
        "error_handling": {"default_strategy": "ignore"},
    }
    with open(os.path.join(artifacts_dir, "canonical_spec.json"), "w") as f:
        json.dump(mock_spec, f)
    with open(os.path.join(artifacts_dir, "cost_estimate.json"), "w") as f:
        json.dump({"runs_per_month": 500}, f)
    with open(os.path.join(artifacts_dir, "confidence.json"), "w") as f:
        json.dump({"score": 0.94, "grade": "A"}, f)
    with open(os.path.join(artifacts_dir, "delivery_pack.json"), "w") as f:
        json.dump({"agent_notes": ["Assumed webhook URL from gateway module"], "original_request": "Post form submissions to Slack"}, f)

    try:
        print("Test 1: Generate from filesystem")
        result = generate_sow_package(artifacts_dir, "Acme Corp", output_dir,
                                       original_request="Post form submissions to Slack")
        assert result["success"]
        assert len(result["files"]) == 4
        print(f"  Files: {list(result['files'].keys())}")
        print(f"  Investment: ${result['total_investment']:,.0f}")
        print(f"  Plan: {result['recommended_plan']}")
        print("  [OK]")

        print("Test 2: All 4 files written and non-empty")
        for fname, fpath in result["files"].items():
            assert os.path.exists(fpath)
            content = open(fpath).read()
            assert len(content) > 100, f"{fname} is too short"
            print(f"  {fname}: {len(content)} chars")
        print("  [OK]")

        print("Test 3: SOW contains required sections")
        sow = open(result["files"]["sow.md"]).read()
        for section in ["Statement of Work", "Scope of Work", "Deliverables", "Timeline", "Investment", "Signatures"]:
            assert section in sow, f"Missing: {section}"
        print("  [OK]")

        print("Test 4: Customer name appears in all docs")
        for fname, fpath in result["files"].items():
            content = open(fpath).read()
            assert "Acme Corp" in content, f"{fname} missing customer name"
        print("  [OK]")

        print("Test 5: generate_sow_package_from_dict")
        artifacts = {
            "canonical_spec": mock_spec,
            "cost_estimate": {"runs_per_month": 200},
            "confidence": {"score": 0.88, "grade": "B"},
            "delivery_pack": {"agent_notes": ["Test note"]},
        }
        result2 = generate_sow_package_from_dict(
            artifacts, "Beta Inc", os.path.join(test_dir, "sow2"),
            slug="beta-test", version=2
        )
        assert result2["success"]
        print(f"  Investment: ${result2['total_investment']:,.0f}")
        print("  [OK]")

    finally:
        shutil.rmtree(test_dir)

    print("\n=== All SOW generator checks passed ===")
