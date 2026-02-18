"""
Blueprint Generator

Produces a detailed human-readable build blueprint from a canonical spec and make_export.
This is the step-by-step reconstruction guide a customer or engineer can use to manually
rebuild the scenario in Make.com without needing the export file.

IMPORTANT: This reads the canonical spec format produced by normalize_to_canonical_spec,
NOT the raw plan_dict format. Key differences:
  - spec["scenario"]["name"]          (not spec["scenario_name"])
  - spec["scenario"]["description"]   (not spec["scenario_description"])
  - spec["modules"]                   (not spec["steps"])
  - spec["metadata"]["agent_notes"]   (not spec["agent_notes"])
  - Each module has: id, module, label, mapper, parameters

Input:
    output_path (str) — path to the versioned artifact directory
    customer_name (str) — client name for the document header

Output:
    build_blueprint.md written to output_path
"""

import json
import os
from datetime import datetime, timezone


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _module_label(module_str):
    if not module_str or ":" not in module_str:
        return module_str or "Unknown Module"
    app, action = module_str.split(":", 1)
    return f"{app.capitalize()} → {action}"


def _format_mapper(mapper: dict) -> str:
    if not mapper:
        return "_No mapper fields configured._"
    lines = []
    for key, value in mapper.items():
        lines.append(f"  - `{key}`: `{value}`")
    return "\n".join(lines)


def _format_parameters(parameters: dict) -> str:
    if not parameters:
        return "_No static parameters configured._"
    lines = []
    for key, value in parameters.items():
        if key == "hook":
            lines.append(f"  - `{key}`: `{value}` _(replace with your actual webhook ID)_")
        else:
            lines.append(f"  - `{key}`: `{value}`")
    return "\n".join(lines)


def generate_blueprint(output_path: str, customer_name: str) -> str:
    """
    Generate build_blueprint.md from artifacts in output_path.

    Reads canonical_spec.json which uses the NORMALIZED format:
      spec["scenario"]["name"]        ← NOT spec["scenario_name"]
      spec["modules"]                 ← NOT spec["steps"]
      spec["metadata"]["agent_notes"] ← NOT spec["agent_notes"]
    """
    canonical_path = os.path.join(output_path, "canonical_spec.json")
    export_path = os.path.join(output_path, "make_export.json")
    timeline_path = os.path.join(output_path, "timeline.json")
    cost_path = os.path.join(output_path, "cost_estimate.json")
    confidence_path = os.path.join(output_path, "confidence.json")

    canonical = _load_json(canonical_path)
    export = _load_json(export_path)
    timeline = _load_json(timeline_path) if os.path.exists(timeline_path) else {}
    cost = _load_json(cost_path) if os.path.exists(cost_path) else {}
    confidence = _load_json(confidence_path) if os.path.exists(confidence_path) else {}

    # --- CANONICAL SPEC FORMAT (normalized) ---
    # These keys come from normalize_to_canonical_spec, not the raw plan_dict
    scenario = canonical.get("scenario", {})
    scenario_name = scenario.get("name", "Unnamed Scenario")           # spec["scenario"]["name"]
    scenario_description = scenario.get("description", "")            # spec["scenario"]["description"]
    trigger = canonical.get("trigger", {})
    modules = canonical.get("modules", [])                             # spec["modules"] not spec["steps"]
    error_handling = canonical.get("error_handling", {})
    metadata = canonical.get("metadata", {})
    agent_notes = metadata.get("agent_notes", [])                      # spec["metadata"]["agent_notes"]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    slug = os.path.basename(os.path.dirname(output_path))
    version = os.path.basename(output_path)

    lines = []

    # Header
    lines.append(f"# Build Blueprint — {customer_name}")
    lines.append(f"**Scenario:** {scenario_name}")
    lines.append(f"**Description:** {scenario_description}")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Slug:** {slug} | **Version:** {version}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("> This document is a complete step-by-step guide to manually recreating")
    lines.append("> this Make.com scenario from scratch. Follow each section in order.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Confidence
    score = confidence.get("score", "N/A")
    grade = confidence.get("grade", "N/A")
    lines.append("## Build Quality")
    lines.append(f"- **Confidence Score:** {score}")
    lines.append(f"- **Grade:** {grade}")
    if confidence.get("explanation"):
        lines.append(f"- **Note:** {confidence['explanation']}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overview
    lines.append("## Scenario Overview")
    lines.append("")
    lines.append(f"This scenario has **{len(modules)} processing module(s)** plus a trigger.")
    lines.append("")
    lines.append("| Position | Module | Label |")
    lines.append("|----------|--------|-------|")

    trigger_module = trigger.get("module", "Unknown")
    trigger_label = trigger.get("label", "Trigger")
    lines.append(f"| 1 (Trigger) | `{trigger_module}` | {trigger_label} |")

    for i, mod in enumerate(modules, start=2):
        mod_name = mod.get("module", "Unknown")
        mod_label = mod.get("label", f"Step {i-1}")
        lines.append(f"| {i} | `{mod_name}` | {mod_label} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Step 1
    lines.append("## Step 1 — Create a New Scenario in Make.com")
    lines.append("")
    lines.append("1. Log in to [Make.com](https://make.com)")
    lines.append("2. Navigate to your target Team and Organization")
    lines.append("3. Click **Create a new scenario**")
    lines.append(f"4. Name the scenario: `{scenario_name}`")
    lines.append("5. Do not add any modules yet — proceed to Step 2")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Step 2: Trigger
    lines.append("## Step 2 — Configure the Trigger")
    lines.append("")
    lines.append(f"**Module:** `{trigger_module}`")
    lines.append(f"**Label:** {trigger_label}")
    lines.append(f"**Trigger Type:** {trigger.get('type', 'webhook')}")
    lines.append("")

    trigger_lower = trigger_module.lower()
    if "webhook" in trigger_lower or "gateway" in trigger_lower:
        lines.append("### Instructions")
        lines.append("1. Click the **+** button to add the first module")
        lines.append("2. Search for **Webhooks** and select **Custom webhook**")
        lines.append("3. Click **Add** to create a new webhook")
        lines.append(f"4. Name the webhook: `{trigger_label}`")
        lines.append("5. Copy the generated webhook URL — you will need this for your upstream system")
        lines.append("6. Click **OK** to save")
        lines.append(f"7. Set the label to: `{trigger_label}`")
    elif "schedule" in trigger_lower:
        lines.append("### Instructions")
        lines.append("1. Click the **+** button to add the first module")
        lines.append("2. Select **Schedule** as the trigger type")
        lines.append("3. Configure the interval based on your requirements")
    else:
        lines.append("### Instructions")
        lines.append(f"1. Add the `{trigger_module}` module as the trigger")
        lines.append("2. Configure according to the app's connection requirements")

    params = trigger.get("parameters", {})
    if params:
        lines.append("")
        lines.append("**Static Parameters:**")
        lines.append(_format_parameters(params))

    lines.append("")
    lines.append("---")
    lines.append("")

    # Step 3: Processing modules
    lines.append("## Step 3 — Add Processing Modules")
    lines.append("")
    lines.append("Add each module in order by clicking the **+** after the previous module.")
    lines.append("")

    for i, mod in enumerate(modules, start=1):
        module = mod.get("module", "Unknown")
        label = mod.get("label", f"Step {i}")
        mapper = mod.get("mapper", {})
        parameters = mod.get("parameters", {})
        app_name = module.split(":")[0] if ":" in module else module

        lines.append(f"### Module {i+1} — {label}")
        lines.append(f"**App:** {app_name.capitalize()}")
        lines.append(f"**Module:** `{module}`")
        lines.append(f"**Label:** {label}")
        lines.append("")
        lines.append("**How to add:**")
        lines.append(f"1. Click **+** after the previous module")
        lines.append(f"2. Search for **{app_name.capitalize()}**")
        lines.append(f"3. Select the action: `{module.split(':')[1] if ':' in module else module}`")

        if app_name.lower() not in ["json", "tools", "gateway", "builtin", "util"]:
            lines.append(f"4. Connect your {app_name.capitalize()} account (OAuth or API key)")
            lines.append(f"5. Set the label to: `{label}`")
        else:
            lines.append(f"4. Set the label to: `{label}`")

        if parameters:
            lines.append("")
            lines.append("**Static Parameters:**")
            lines.append(_format_parameters(parameters))

        if mapper:
            lines.append("")
            lines.append("**Field Mappings:**")
            lines.append(_format_mapper(mapper))
            lines.append("")
            lines.append("_Map these fields by clicking into each field and selecting the")
            lines.append("corresponding output from the previous module in the mapping panel._")

        lines.append("")

    lines.append("---")
    lines.append("")

    # Step 4: Error handling
    lines.append("## Step 4 — Configure Error Handling")
    lines.append("")
    strategy = error_handling.get("default_strategy", "ignore")
    max_errors = error_handling.get("max_errors", 3)
    lines.append(f"**Strategy:** {strategy}")
    lines.append(f"**Max consecutive errors:** {max_errors}")
    lines.append("")
    if strategy == "ignore":
        lines.append("1. Click the **wrench icon** on the scenario")
        lines.append("2. Under **Error handling**, select **Ignore**")
    elif strategy == "rollback":
        lines.append("1. Click the **wrench icon** on the scenario")
        lines.append("2. Under **Error handling**, select **Rollback**")
    else:
        lines.append("1. Click the **wrench icon** on the scenario")
        lines.append("2. Configure error handling per your team's standard")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Step 5: Connections
    apps_needed = set()
    for mod in modules:
        mod_str = mod.get("module", "")
        if ":" in mod_str:
            app = mod_str.split(":")[0].lower()
            if app not in ["json", "tools", "gateway", "builtin", "util"]:
                apps_needed.add(app)

    if apps_needed:
        lines.append("## Step 5 — Set Up Connections")
        lines.append("")
        lines.append("The following app connections are required:")
        lines.append("")
        for app in sorted(apps_needed):
            lines.append(f"### {app.capitalize()}")
            lines.append(f"1. Go to **Connections** in your Make.com left sidebar")
            lines.append(f"2. Click **Add connection** and search for **{app.capitalize()}**")
            lines.append(f"3. Follow the OAuth or API key flow to authorize the connection")
            lines.append(f"4. Return to the scenario and assign this connection to the `{app}` module(s)")
            lines.append("")
        lines.append("---")
        lines.append("")

    # Timeline
    if timeline:
        lines.append("## Estimated Build Time")
        lines.append("")
        lines.append(f"- **Total:** {timeline.get('total_hours', 'N/A')} hours")
        lines.append(f"- **Complexity:** {timeline.get('complexity_grade', timeline.get('complexity', 'N/A'))}")
        breakdown = timeline.get("breakdown", {})
        if breakdown:
            lines.append("")
            lines.append("**Breakdown:**")
            for phase, hours in breakdown.items():
                lines.append(f"- {phase}: {hours} hours")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Cost
    if cost:
        lines.append("## Make.com Operations Estimate")
        lines.append("")
        recommended = cost.get("recommended_plan", {})
        plan_name = recommended.get("name", "N/A") if isinstance(recommended, dict) else recommended
        monthly_cost = cost.get("monthly_operational_cost", cost.get("estimated_monthly_cost_usd", 0))
        lines.append(f"- **Operations per execution:** {cost.get('ops_per_execution', 'N/A')}")
        lines.append(f"- **Operations per month:** {cost.get('ops_per_month', 'N/A')}")
        lines.append(f"- **Recommended plan:** {plan_name}")
        lines.append(f"- **Estimated monthly cost:** ${monthly_cost}/mo")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Agent notes (from metadata, NOT from spec root)
    if agent_notes:
        lines.append("## Build Notes")
        lines.append("")
        lines.append("The following assumptions were made during the build:")
        lines.append("")
        for note in agent_notes:
            lines.append(f"- {note}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Checklist
    lines.append("## Final Checklist Before Activating")
    lines.append("")
    lines.append("- [ ] All modules are connected and show no error indicators")
    lines.append("- [ ] All required app connections are authorized")
    lines.append("- [ ] Webhook URL has been added to the upstream system (if applicable)")
    lines.append("- [ ] Field mappings reference the correct source module outputs")
    lines.append("- [ ] Run once manually using **Run once** to confirm data flows correctly")
    lines.append("- [ ] Review the output bundle at each module to verify expected values")
    lines.append("- [ ] Activate the scenario using the toggle in the bottom-left")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_Blueprint generated by ManageAI Agentic Make Build Engine — {now}_")

    output_file = os.path.join(output_path, "build_blueprint.md")
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    return output_file


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python blueprint_generator.py <output_path> <customer_name>")
        sys.exit(1)
    result = generate_blueprint(sys.argv[1], sys.argv[2])
    print(f"Blueprint written to: {result}")
