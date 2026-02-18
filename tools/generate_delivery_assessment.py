"""
Generate Delivery Assessment

Converts a structured Customer Success intake into:
  1. A delivery assessment report (summary, scope, risks, assumptions)
  2. A deterministic plan_dict compatible with normalize_to_canonical_spec

The agent (reasoning layer) is responsible for reading raw tickets,
documentation, and knowledge vault text, then producing the structured
intake dict. This tool performs deterministic validation, ambiguity
detection, and plan generation — no AI reasoning, no network calls.

Input:
    intake (dict) — structured CS intake (see schema below)
    registry (dict) — loaded module registry

Output:
    dict with:
        - ready_for_build: bool
        - clarification_questions: list[str] | None
        - delivery_report: dict | None
        - plan_dict: dict | None (compatible with normalize_to_canonical_spec)

Intake schema:
    {
        "customer_name": str,
        "ticket_summary": str,
        "business_objective": str,
        "trigger_type": "webhook" | "schedule" | "app_event" | null,
        "trigger_description": str,
        "processing_steps": [
            {
                "description": str,
                "app": str | null,      — registry app name (e.g., "slack", "json")
                "module": str | null,   — registry module_id (e.g., "slack:PostMessage")
                "inputs": str | null,   — what data this step consumes
                "outputs": str | null   — what data this step produces
            }
        ],
        "routing": {
            "needed": bool,
            "conditions": [
                {"description": str, "field": str, "operator": str, "value": str}
            ] | null
        },
        "error_handling_preference": "ignore" | "break" | "resume" | null,
        "expected_outputs": [str],
        "estimated_frequency": str | null,     — e.g., "100/day", "on-demand"
        "customer_notes": str | null,
        "research_summary": str | null         — required if non-registry apps detected
    }

Deterministic. No network calls. No AI reasoning.
"""

from tools.module_registry_loader import load_module_registry

# Apps in the curated registry (extracted at module level for lookups)
_REGISTRY_APPS = None

# Valid trigger types
VALID_TRIGGER_TYPES = {"webhook", "schedule", "app_event"}

# Default trigger module per type
TRIGGER_MODULE_MAP = {
    "webhook": "gateway:CustomWebHook",
    # schedule and app_event not yet in our curated registry
}

# App → likely module mapping for common intents
APP_MODULE_HINTS = {
    "slack": {"post": "slack:PostMessage", "send": "slack:PostMessage", "notify": "slack:PostMessage",
              "message": "slack:PostMessage", "default": "slack:PostMessage"},
    "google-sheets": {"add": "google-sheets:addRow", "append": "google-sheets:addRow",
                      "log": "google-sheets:addRow", "get": "google-sheets:getRow",
                      "read": "google-sheets:getRow", "update": "google-sheets:updateRow",
                      "default": "google-sheets:addRow"},
    "json": {"parse": "json:ParseJSON", "create": "json:TransformToJSON",
             "aggregate": "json:AggregateToJSON", "default": "json:ParseJSON"},
    "http": {"request": "http:ActionSendData", "send": "http:ActionSendData",
             "get": "http:ActionGetFile", "download": "http:ActionGetFile",
             "fetch": "http:ActionSendData", "default": "http:ActionSendData"},
    "util": {"variable": "util:SetVariable", "set": "util:SetVariable",
             "multiple": "util:SetMultipleVariables", "default": "util:SetVariable"},
}


def _default_mapper(module_id, prev_module_id):
    """Generate default mapper values for a module based on its required fields.

    Uses the previous module's ID for data references ({{N.field}} format).
    Returns dict with required fields populated, or empty dict if unknown module.
    """
    p = str(prev_module_id)
    defaults = {
        "json:ParseJSON": {"json": "{{" + p + ".body}}"},
        "json:TransformToJSON": {},
        "json:AggregateToJSON": {"array": "{{" + p + "}}"},
        "slack:PostMessage": {"channel": "#notifications", "text": "{{" + p + ".value}}"},
        "google-sheets:addRow": {"values": ["{{" + p + ".value}}"]},
        "google-sheets:getRow": {"row": "{{" + p + ".id}}"},
        "google-sheets:updateRow": {"row": "{{" + p + ".id}}", "values": ["{{" + p + ".value}}"]},
        "http:ActionSendData": {"url": "{{" + p + ".url}}", "method": "GET"},
        "http:ActionGetFile": {"url": "{{" + p + ".url}}"},
        "util:SetVariable": {"name": "variable", "value": "{{" + p + ".value}}"},
        "util:SetMultipleVariables": {"variables": [{"name": "var1", "value": "{{" + p + ".value}}"}]},
        "gateway:WebhookResponse": {"status": "200", "body": "{{" + p + ".value}}"},
    }
    return defaults.get(module_id, {})


def _default_parameters(module_id, prev_module_id):
    """Generate default parameters for a module based on its required params.

    Returns dict with required parameters populated, or empty dict if none needed.
    """
    p = str(prev_module_id)
    defaults = {
        "builtin:BasicAggregator": {"feeder": prev_module_id},
        "json:AggregateToJSON": {"feeder": prev_module_id},
        "http:ActionSendData": {"handleErrors": True},
        "http:ActionGetFile": {"handleErrors": True},
        "google-sheets:addRow": {"spreadsheetId": "__SPREADSHEET_ID__", "sheetId": "__SHEET_ID__"},
        "google-sheets:getRow": {"spreadsheetId": "__SPREADSHEET_ID__", "sheetId": "__SHEET_ID__"},
        "google-sheets:updateRow": {"spreadsheetId": "__SPREADSHEET_ID__", "sheetId": "__SHEET_ID__"},
    }
    return defaults.get(module_id, {})


def generate_delivery_assessment(intake, registry):
    """Generate a delivery assessment from a structured CS intake.

    Args:
        intake: Structured intake dict from agent pre-processing.
        registry: Loaded module registry.

    Returns:
        dict with ready_for_build, clarification_questions,
        delivery_report, plan_dict.
    """
    modules_dict = registry.get("modules", {})
    registry_apps = set()
    for mod_id, mod in modules_dict.items():
        registry_apps.add(mod.get("app", ""))

    questions = []
    risks = []
    assumptions = []

    # === Phase A: Validate required intake fields ===
    _check_required_field(intake, "customer_name", questions)
    _check_required_field(intake, "ticket_summary", questions)
    _check_required_field(intake, "business_objective", questions)
    _check_required_field(intake, "trigger_description", questions)

    if not intake.get("processing_steps"):
        questions.append("What processing steps should the scenario perform after the trigger fires?")

    if not intake.get("expected_outputs"):
        questions.append("What are the expected outputs or end results of this scenario?")

    # === Phase B: Validate trigger ===
    trigger_type = intake.get("trigger_type")
    if trigger_type is None:
        questions.append(
            "What should trigger this scenario? Options: "
            "webhook (incoming HTTP request), schedule (time-based), "
            "or app_event (triggered by an external app)."
        )
    elif trigger_type not in VALID_TRIGGER_TYPES:
        questions.append(
            f"Trigger type '{trigger_type}' is not recognized. "
            f"Valid options: webhook, schedule, app_event."
        )
    elif trigger_type != "webhook":
        # Only webhook is in our curated registry currently
        risks.append(
            f"Trigger type '{trigger_type}' is not yet in the curated module registry. "
            f"Only 'webhook' triggers are currently supported."
        )
        questions.append(
            f"Trigger type '{trigger_type}' is not currently supported. "
            f"Can this scenario use a webhook trigger instead?"
        )

    # === Phase C: Validate processing steps and resolve modules ===
    resolved_steps = []
    non_registry_apps = set()

    for i, step in enumerate(intake.get("processing_steps", [])):
        desc = step.get("description", "")
        app = step.get("app")
        module = step.get("module")

        # If module is explicitly provided, validate it
        if module:
            if module not in modules_dict:
                questions.append(
                    f"Step {i+1} ('{desc}'): Module '{module}' is not in the curated registry. "
                    f"Is there an alternative module that could be used?"
                )
                non_registry_apps.add(module.split(":")[0] if ":" in module else module)
            resolved_steps.append({**step, "_resolved_module": module})
            continue

        # If only app is provided, try to resolve module from description
        if app:
            if app in APP_MODULE_HINTS:
                resolved_module = _resolve_module_from_description(app, desc)
                resolved_steps.append({**step, "_resolved_module": resolved_module})
                assumptions.append(
                    f"Step {i+1}: Mapped '{app}' with description '{desc}' "
                    f"to module '{resolved_module}'"
                )
            elif app in registry_apps:
                # App exists in registry but not in our hint map — find first match
                candidates = [
                    mid for mid, m in modules_dict.items()
                    if m.get("app") == app and m.get("category") != "trigger"
                ]
                if candidates:
                    resolved_steps.append({**step, "_resolved_module": candidates[0]})
                    assumptions.append(
                        f"Step {i+1}: Auto-selected module '{candidates[0]}' for app '{app}'"
                    )
                else:
                    questions.append(
                        f"Step {i+1} ('{desc}'): App '{app}' found in registry but "
                        f"no non-trigger modules available. Which module should be used?"
                    )
                    resolved_steps.append({**step, "_resolved_module": None})
            else:
                non_registry_apps.add(app)
                questions.append(
                    f"Step {i+1} ('{desc}'): App '{app}' is not in the curated registry. "
                    f"Supported apps: {', '.join(sorted(registry_apps - {'gateway', 'builtin'}))}. "
                    f"Is there an alternative?"
                )
                resolved_steps.append({**step, "_resolved_module": None})
        else:
            # Neither app nor module provided
            questions.append(
                f"Step {i+1} ('{desc}'): Which app/module should handle this step? "
                f"Supported: slack, google-sheets, http, json, util."
            )
            resolved_steps.append({**step, "_resolved_module": None})

    # === Phase D: Third-party / non-registry app check ===
    if non_registry_apps and not intake.get("research_summary"):
        questions.append(
            f"Non-registry app(s) detected: {', '.join(sorted(non_registry_apps))}. "
            f"Please provide a research_summary with API details and module mappings, "
            f"or choose from supported apps."
        )

    # === Phase E: Routing validation ===
    routing = intake.get("routing", {})
    if routing.get("needed") and not routing.get("conditions"):
        questions.append(
            "Routing is needed but no conditions were specified. "
            "What field and value should determine each route?"
        )

    # === Phase F: Ambiguity check — stop if questions exist ===
    if questions:
        return {
            "ready_for_build": False,
            "clarification_questions": questions,
            "delivery_report": None,
            "plan_dict": None
        }

    # === Phase G: Build delivery report ===
    delivery_report = _build_delivery_report(
        intake, resolved_steps, risks, assumptions, registry_apps
    )

    # === Phase H: Generate plan_dict ===
    plan_dict = _build_plan_dict(intake, resolved_steps, modules_dict)

    return {
        "ready_for_build": True,
        "clarification_questions": None,
        "delivery_report": delivery_report,
        "plan_dict": plan_dict
    }


def _check_required_field(intake, field, questions):
    """Check that a required intake field is present and non-empty."""
    val = intake.get(field)
    if not val or (isinstance(val, str) and not val.strip()):
        label = field.replace("_", " ").title()
        questions.append(f"Missing required field: {label}. Please provide this information.")


def _resolve_module_from_description(app, description):
    """Resolve a module ID from app name + step description keywords."""
    hints = APP_MODULE_HINTS.get(app, {})
    desc_lower = description.lower()

    for keyword, module in hints.items():
        if keyword != "default" and keyword in desc_lower:
            return module

    return hints.get("default", f"{app}:unknown")


def _build_delivery_report(intake, resolved_steps, risks, assumptions, registry_apps):
    """Build the structured delivery assessment report."""
    # Identify credential-requiring apps
    cred_apps = set()
    for step in resolved_steps:
        mod = step.get("_resolved_module", "")
        if mod:
            app = mod.split(":")[0] if ":" in mod else ""
            if app in ("slack", "google-sheets"):
                cred_apps.add(app)

    # Scope items
    scope = []
    trigger_desc = intake.get("trigger_description", "")
    scope.append(f"Configure {intake.get('trigger_type', 'webhook')} trigger: {trigger_desc}")
    for i, step in enumerate(resolved_steps):
        mod = step.get("_resolved_module", "unknown")
        scope.append(f"Step {i+1}: {step.get('description', '')} ({mod})")

    routing = intake.get("routing", {})
    if routing.get("needed"):
        conditions = routing.get("conditions", [])
        scope.append(f"Configure router with {len(conditions)} routing condition(s)")

    for app in sorted(cred_apps):
        scope.append(f"Set up {app} OAuth/API credentials")

    scope.append("Validate and test complete scenario")

    # Risks
    if not risks:
        risks.append("No significant risks identified.")

    # Timeline assumptions
    timeline_assumptions = [
        "Customer provides credentials promptly",
        "All specified integrations use standard APIs",
        "No custom data transformation beyond standard module capabilities"
    ]
    if cred_apps:
        timeline_assumptions.append(
            f"OAuth setup for {', '.join(sorted(cred_apps))} takes < 30 minutes each"
        )

    # Cost assumptions
    freq = intake.get("estimated_frequency", "100 executions/month")
    cost_assumptions = [
        f"Estimated execution frequency: {freq}",
        "Standard Make.com pricing applies",
        "No premium connector costs"
    ]

    return {
        "customer_name": intake.get("customer_name", "Unknown"),
        "summary": (
            f"Build a Make.com scenario for {intake.get('customer_name', 'the customer')}: "
            f"{intake.get('business_objective', intake.get('ticket_summary', ''))}"
        ),
        "business_objective": intake.get("business_objective", ""),
        "scope": scope,
        "integrations": [
            step.get("_resolved_module", "unknown") for step in resolved_steps
        ],
        "risks": risks,
        "assumptions": assumptions,
        "timeline_assumptions": timeline_assumptions,
        "cost_assumptions": cost_assumptions,
        "expected_outputs": intake.get("expected_outputs", [])
    }


def _build_plan_dict(intake, resolved_steps, modules_dict):
    """Generate a plan_dict compatible with normalize_to_canonical_spec.

    Plan dict format:
        scenario_name, scenario_description, trigger, steps[],
        connections[] (optional), error_handling, agent_notes[]
    """
    trigger_type = intake.get("trigger_type", "webhook")
    trigger_module = TRIGGER_MODULE_MAP.get(trigger_type, "gateway:CustomWebHook")

    # Build trigger
    trigger = {
        "type": trigger_type,
        "module": trigger_module,
        "label": intake.get("trigger_description", "Trigger"),
        "parameters": {"hook": "__WEBHOOK_ID__"} if trigger_type == "webhook" else {}
    }

    # Build steps
    steps = []
    routing = intake.get("routing", {})
    has_router = routing.get("needed", False)
    router_step_index = None

    # If routing is needed, insert router after the last pre-routing step
    # Convention: router goes before the steps that branch
    if has_router:
        # Find where to insert the router.
        # If the first step looks like parsing/preprocessing, router goes after it.
        # Otherwise router is the first step.
        pre_router_count = 0
        for step in resolved_steps:
            mod = step.get("_resolved_module", "")
            # Parsing/variable-setting steps go before the router
            if mod.startswith("json:") or mod.startswith("util:"):
                pre_router_count += 1
            else:
                break

        # Add pre-router steps
        for i, step in enumerate(resolved_steps[:pre_router_count]):
            prev_mid = i + 1 if i > 0 else 1  # step 0 refs trigger(1), step i refs module i+1
            steps.append(_step_from_resolved(step, prev_mid))

        # Add router
        router_step_index = len(steps)
        steps.append({
            "module": "builtin:BasicRouter",
            "label": "Route by Condition"
        })

        # Add post-router steps (these become route branches)
        # Branch steps reference the last pre-router module for data
        last_pre_router_mid = pre_router_count + 1 if pre_router_count > 0 else 1
        for step in resolved_steps[pre_router_count:]:
            steps.append(_step_from_resolved(step, last_pre_router_mid))
    else:
        # Linear: just add all steps
        for i, step in enumerate(resolved_steps):
            prev_mid = i + 1 if i > 0 else 1
            steps.append(_step_from_resolved(step, prev_mid))

    # Build connections
    connections = None
    if has_router and router_step_index is not None:
        connections = []

        # trigger → first step (or router if no pre-router steps)
        connections.append({"from": "trigger", "to": 0})

        # Chain pre-router steps linearly
        for i in range(router_step_index):
            if i < router_step_index - 1:
                connections.append({"from": i, "to": i + 1})
            else:
                # Last pre-router step → router
                connections.append({"from": i, "to": router_step_index})

        # If router is the first step, trigger connects directly to it
        if router_step_index == 0:
            pass  # trigger → 0 already added

        # Router → each branch step
        conditions = routing.get("conditions", [])
        branch_steps = list(range(router_step_index + 1, len(steps)))

        for bi, branch_idx in enumerate(branch_steps):
            conn = {"from": router_step_index, "to": branch_idx}

            # Apply filter from conditions if available
            if bi < len(conditions):
                cond = conditions[bi]
                conn["filter"] = {
                    "name": cond.get("description", f"Condition {bi+1}"),
                    "conditions": [[{
                        "a": "{{" + _make_ref_field(router_step_index, cond.get("field", "value")) + "}}",
                        "b": cond.get("value", ""),
                        "o": _map_operator(cond.get("operator", "equals"))
                    }]]
                }
                conn["label"] = cond.get("description", f"Route {bi+1}")

            connections.append(conn)

    # Error handling
    eh_pref = intake.get("error_handling_preference", "ignore") or "ignore"
    error_handling = {
        "default_strategy": eh_pref,
        "max_errors": 3
    }

    # Agent notes
    agent_notes = []
    for step in resolved_steps:
        if step.get("_resolved_module") and step.get("_resolved_module") != step.get("module"):
            agent_notes.append(
                f"Auto-resolved '{step.get('description', '')}' → {step.get('_resolved_module')}"
            )
    if intake.get("customer_notes"):
        agent_notes.append(f"Customer note: {intake['customer_notes']}")
    if intake.get("research_summary"):
        agent_notes.append(f"Research: {intake['research_summary']}")

    plan = {
        "scenario_name": _generate_scenario_name(intake),
        "scenario_description": intake.get("business_objective", intake.get("ticket_summary", "")),
        "trigger": trigger,
        "steps": steps,
        "error_handling": error_handling,
        "agent_notes": agent_notes
    }

    if connections is not None:
        plan["connections"] = connections

    return plan


def _step_from_resolved(step, prev_module_id=1):
    """Convert a resolved intake step to a plan_dict step.

    Args:
        step: Resolved intake step with _resolved_module.
        prev_module_id: Module ID of the upstream module for data references.
    """
    module = step.get("_resolved_module")
    if not module:
        module = "unknown:Unknown"

    result = {
        "module": module,
        "label": step.get("description", "Step")
    }

    # Use explicit mapper if provided, otherwise generate defaults
    if step.get("mapper"):
        result["mapper"] = step["mapper"]
    else:
        default = _default_mapper(module, prev_module_id)
        if default:
            result["mapper"] = default

    # Use explicit parameters if provided, otherwise generate defaults
    if step.get("parameters"):
        result["parameters"] = step["parameters"]
    else:
        default_params = _default_parameters(module, prev_module_id)
        if default_params:
            result["parameters"] = default_params

    return result


def _generate_scenario_name(intake):
    """Generate a scenario name from the intake."""
    customer = intake.get("customer_name", "")
    summary = intake.get("ticket_summary", "")

    if customer and summary:
        # Truncate to reasonable length
        name = f"{customer}: {summary}"
        if len(name) > 100:
            name = name[:97] + "..."
        return name
    return summary or "Untitled Scenario"


def _make_ref_field(step_index, field):
    """Build a data reference field.

    For filters on router connections, we reference the upstream module.
    step_index is 0-based; the actual module ID = step_index + 2.
    But since connections use 0-based indices and normalize converts them,
    we need the reference to use the module ID that the data came from.

    For a router at step_index, data typically comes from the step before it
    (step_index - 1 → module ID step_index + 1) or from the trigger (module 1).
    """
    # Reference the step before the router, or the trigger
    if step_index > 0:
        source_module_id = step_index + 1  # 0-based step index → module ID
    else:
        source_module_id = 1  # Trigger

    return f"{source_module_id}.{field}"


def _map_operator(op_str):
    """Map a human-readable operator to Make.com's operator format."""
    op_map = {
        "equals": "text:equal",
        "equal": "text:equal",
        "not_equals": "text:notEqual",
        "contains": "text:contain",
        "starts_with": "text:startsWith",
        "ends_with": "text:endsWith",
        "greater_than": "number:greater",
        "less_than": "number:less",
        "exists": "exist",
        "not_exists": "notExist",
    }
    return op_map.get(op_str.lower().replace(" ", "_"), "text:equal")


# --- Self-check ---
if __name__ == "__main__":
    import json

    print("=== Generate Delivery Assessment Self-Check ===\n")

    registry = load_module_registry()

    # --- Test 1: Complete intake → ready_for_build ---
    print("Test 1: Complete linear intake (webhook → parse → slack)")
    intake1 = {
        "customer_name": "Acme Corp",
        "ticket_summary": "Form submission notifications to Slack",
        "business_objective": "Automatically notify team on Slack when a web form is submitted.",
        "trigger_type": "webhook",
        "trigger_description": "Incoming form submission webhook",
        "processing_steps": [
            {
                "description": "Parse the incoming JSON payload",
                "app": "json",
                "module": None,
                "inputs": "Raw webhook body",
                "outputs": "Structured form fields"
            },
            {
                "description": "Post notification to Slack channel",
                "app": "slack",
                "module": None,
                "inputs": "Parsed form data",
                "outputs": "Slack message"
            }
        ],
        "routing": {"needed": False, "conditions": None},
        "error_handling_preference": "ignore",
        "expected_outputs": ["Slack message in #form-submissions channel"],
        "estimated_frequency": "50/day",
        "customer_notes": "Channel is #form-submissions",
        "research_summary": None
    }

    r1 = generate_delivery_assessment(intake1, registry)
    assert r1["ready_for_build"] is True, f"Should be ready, got questions: {r1['clarification_questions']}"
    assert r1["clarification_questions"] is None
    assert r1["delivery_report"] is not None
    assert r1["plan_dict"] is not None

    # Verify plan_dict structure
    pd = r1["plan_dict"]
    assert pd["scenario_name"] == "Acme Corp: Form submission notifications to Slack"
    assert pd["trigger"]["type"] == "webhook"
    assert pd["trigger"]["module"] == "gateway:CustomWebHook"
    assert len(pd["steps"]) == 2
    assert pd["steps"][0]["module"] == "json:ParseJSON"
    assert pd["steps"][1]["module"] == "slack:PostMessage"
    assert "connections" not in pd  # Linear flow — no explicit connections
    assert pd["error_handling"]["default_strategy"] == "ignore"

    # Verify delivery report
    dr = r1["delivery_report"]
    assert dr["customer_name"] == "Acme Corp"
    assert len(dr["scope"]) > 0
    assert "50/day" in dr["cost_assumptions"][0]

    print(f"  ready_for_build: {r1['ready_for_build']}")
    print(f"  plan_dict steps: {len(pd['steps'])}")
    print(f"  scenario_name: {pd['scenario_name']}")
    print(f"  delivery_report scope items: {len(dr['scope'])}")
    print("  [OK] Complete intake produces valid plan_dict")

    # --- Test 2: Intake with routing ---
    print("\nTest 2: Router intake (webhook → parse → router → slack/sheets)")
    intake2 = {
        "customer_name": "Beta Inc",
        "ticket_summary": "Priority-based alert routing",
        "business_objective": "Route high-priority events to Slack, log all to Google Sheets.",
        "trigger_type": "webhook",
        "trigger_description": "Event webhook from monitoring system",
        "processing_steps": [
            {
                "description": "Parse event payload",
                "app": "json",
                "module": None,
                "inputs": "Raw event",
                "outputs": "Parsed event with priority field"
            },
            {
                "description": "Send urgent alert to Slack",
                "app": "slack",
                "module": None,
                "inputs": "Parsed event",
                "outputs": "Slack notification"
            },
            {
                "description": "Log event to tracking spreadsheet",
                "app": "google-sheets",
                "module": "google-sheets:addRow",
                "inputs": "Parsed event",
                "outputs": "New row in sheet"
            }
        ],
        "routing": {
            "needed": True,
            "conditions": [
                {"description": "High Priority", "field": "priority", "operator": "equals", "value": "high"}
            ]
        },
        "error_handling_preference": "break",
        "expected_outputs": ["Slack alert for high priority", "All events logged to sheet"],
        "estimated_frequency": "200/day",
        "customer_notes": None,
        "research_summary": None
    }

    r2 = generate_delivery_assessment(intake2, registry)
    assert r2["ready_for_build"] is True, f"Should be ready, got: {r2['clarification_questions']}"

    pd2 = r2["plan_dict"]
    # Should have: parse, router, slack, sheets = 4 steps
    assert len(pd2["steps"]) == 4
    assert pd2["steps"][0]["module"] == "json:ParseJSON"
    assert pd2["steps"][1]["module"] == "builtin:BasicRouter"
    assert pd2["steps"][2]["module"] == "slack:PostMessage"
    assert pd2["steps"][3]["module"] == "google-sheets:addRow"
    assert "connections" in pd2  # Routing requires explicit connections
    assert pd2["error_handling"]["default_strategy"] == "break"

    # Verify filter on router connection
    router_conns = [c for c in pd2["connections"] if c.get("filter")]
    assert len(router_conns) == 1
    assert router_conns[0]["filter"]["name"] == "High Priority"

    print(f"  Steps: {[s['module'] for s in pd2['steps']]}")
    print(f"  Connections: {len(pd2['connections'])}")
    print(f"  Filtered connections: {len(router_conns)}")
    print("  [OK] Router intake with conditions")

    # --- Test 3: Incomplete intake → clarification questions ---
    print("\nTest 3: Incomplete intake (missing fields)")
    intake3 = {
        "customer_name": "Gamma LLC",
        "ticket_summary": "",
        "business_objective": "",
        "trigger_type": None,
        "trigger_description": "",
        "processing_steps": [],
        "routing": {"needed": False},
        "error_handling_preference": None,
        "expected_outputs": [],
        "estimated_frequency": None,
        "customer_notes": None,
        "research_summary": None
    }

    r3 = generate_delivery_assessment(intake3, registry)
    assert r3["ready_for_build"] is False
    assert r3["clarification_questions"] is not None
    assert len(r3["clarification_questions"]) >= 4  # Multiple missing fields
    assert r3["plan_dict"] is None

    print(f"  Questions: {len(r3['clarification_questions'])}")
    for q in r3["clarification_questions"]:
        print(f"    - {q[:80]}...")
    print("  [OK] Missing fields generate clarification questions")

    # --- Test 4: Non-registry app → requires research_summary ---
    print("\nTest 4: Non-registry app without research_summary")
    intake4 = {
        "customer_name": "Delta Corp",
        "ticket_summary": "Salesforce sync to Slack",
        "business_objective": "When Salesforce lead created, notify Slack.",
        "trigger_type": "webhook",
        "trigger_description": "Salesforce webhook",
        "processing_steps": [
            {
                "description": "Process Salesforce data",
                "app": "salesforce",
                "module": None,
                "inputs": "Salesforce lead",
                "outputs": "Processed lead"
            },
            {
                "description": "Post to Slack",
                "app": "slack",
                "module": None,
                "inputs": "Lead data",
                "outputs": "Slack message"
            }
        ],
        "routing": {"needed": False},
        "error_handling_preference": "ignore",
        "expected_outputs": ["Slack notification"],
        "estimated_frequency": "20/day",
        "customer_notes": None,
        "research_summary": None
    }

    r4 = generate_delivery_assessment(intake4, registry)
    assert r4["ready_for_build"] is False
    assert any("salesforce" in q.lower() for q in r4["clarification_questions"])
    assert any("research_summary" in q.lower() or "registry" in q.lower()
               for q in r4["clarification_questions"])
    print(f"  Questions: {len(r4['clarification_questions'])}")
    for q in r4["clarification_questions"]:
        print(f"    - {q[:80]}...")
    print("  [OK] Non-registry app blocked without research_summary")

    # --- Test 5: Unsupported trigger type ---
    print("\nTest 5: Unsupported trigger type")
    intake5 = {
        "customer_name": "Epsilon",
        "ticket_summary": "Scheduled report to Slack",
        "business_objective": "Every morning, send a report to Slack.",
        "trigger_type": "schedule",
        "trigger_description": "Daily at 8am",
        "processing_steps": [
            {"description": "Post report", "app": "slack", "module": None,
             "inputs": "Report data", "outputs": "Slack message"}
        ],
        "routing": {"needed": False},
        "error_handling_preference": "ignore",
        "expected_outputs": ["Daily Slack message"],
        "estimated_frequency": "1/day",
        "customer_notes": None,
        "research_summary": None
    }

    r5 = generate_delivery_assessment(intake5, registry)
    assert r5["ready_for_build"] is False
    assert any("schedule" in q.lower() for q in r5["clarification_questions"])
    print(f"  Questions: {len(r5['clarification_questions'])}")
    print("  [OK] Unsupported trigger type flagged")

    # --- Test 6: plan_dict is pipeline-compatible ---
    print("\nTest 6: plan_dict feeds into build_scenario_pipeline")
    from tools.build_scenario_pipeline import build_scenario_pipeline
    import tempfile
    import shutil

    test_dir = tempfile.mkdtemp(prefix="amb_assess_test_")
    try:
        result = build_scenario_pipeline(
            r1["plan_dict"], registry,
            intake1["business_objective"],
            base_output_dir=test_dir
        )
        assert result["success"] is True, f"Pipeline failed: {result['failure_reason']}"
        assert result["canonical_validation"]["valid"] is True
        assert result["export_validation"]["valid"] is True
        print(f"  Pipeline: SUCCESS (v{result['version']})")
        print(f"  Spec: {result['canonical_validation']['checks_passed']}/{result['canonical_validation']['checks_run']}")
        print(f"  Export: {result['export_validation']['checks_passed']}/{result['export_validation']['checks_run']}")
        print(f"  Confidence: {result['confidence']['score']} ({result['confidence']['grade']})")
    finally:
        shutil.rmtree(test_dir)
    print("  [OK] plan_dict → pipeline → SUCCESS")

    # --- Test 7: Router plan_dict pipeline-compatible ---
    print("\nTest 7: Router plan_dict feeds into pipeline")
    test_dir2 = tempfile.mkdtemp(prefix="amb_assess_router_test_")
    try:
        result2 = build_scenario_pipeline(
            r2["plan_dict"], registry,
            intake2["business_objective"],
            base_output_dir=test_dir2
        )
        assert result2["success"] is True, f"Router pipeline failed: {result2['failure_reason']}"
        print(f"  Pipeline: SUCCESS (v{result2['version']})")
        print(f"  Spec: {result2['canonical_validation']['checks_passed']}/{result2['canonical_validation']['checks_run']}")
        print(f"  Confidence: {result2['confidence']['score']} ({result2['confidence']['grade']})")
    finally:
        shutil.rmtree(test_dir2)
    print("  [OK] Router plan_dict → pipeline → SUCCESS")

    # --- Test 8: Determinism ---
    print("\nTest 8: Determinism")
    r8a = generate_delivery_assessment(intake1, registry)
    r8b = generate_delivery_assessment(intake1, registry)
    assert r8a["ready_for_build"] == r8b["ready_for_build"]
    assert r8a["plan_dict"] == r8b["plan_dict"]
    assert r8a["delivery_report"] == r8b["delivery_report"]
    print("  [OK] Deterministic output")

    print("\n=== All generate_delivery_assessment checks passed ===")
