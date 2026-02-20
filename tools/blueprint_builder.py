"""
Blueprint Builder — Converts plan_dict to Make.com scenario blueprints.

Takes the plan_dict structure produced by POST /plan and generates
a valid Make.com blueprint JSON that can be imported via the API.
"""

from typing import Optional


# ── Module builders ──────────────────────────────────────────

def build_webhook_trigger(label: str = "Webhook", hook_id: Optional[int] = None) -> dict:
    """Build a Custom Webhook trigger module."""
    module = {
        "id": 1,
        "module": "gateway:CustomWebHook",
        "version": 1,
        "metadata": {"designer": {"x": 0, "y": 0}},
        "parameters": {},
        "mapper": {},
    }
    if hook_id:
        module["parameters"]["hook"] = hook_id
    if label:
        module["metadata"]["designer"]["name"] = label
    return module


def build_http_request_module(
    id: int,
    label: str,
    url: str = "",
    method: str = "GET",
    body: Optional[str] = None,
) -> dict:
    """Build an HTTP Request module."""
    mapper = {"url": url, "method": method}
    if body:
        mapper["body"] = body
    return {
        "id": id,
        "module": "http:ActionSendData",
        "version": 3,
        "metadata": {"designer": {"x": 300 * (id - 1), "y": 0, "name": label}},
        "parameters": {},
        "mapper": mapper,
    }


def build_router_module(id: int, label: str = "Router") -> dict:
    """Build a Basic Router module."""
    return {
        "id": id,
        "module": "builtin:BasicRouter",
        "version": 1,
        "metadata": {"designer": {"x": 300 * (id - 1), "y": 0, "name": label}},
        "parameters": {},
        "mapper": {},
        "routes": [{"flow": []}],
    }


def build_set_variable_module(
    id: int, label: str, variables: Optional[list] = None
) -> dict:
    """Build a Set Variable module."""
    mapper = {}
    if variables:
        mapper["variables"] = [
            {"name": v.get("name", f"var{i}"), "value": v.get("value", "")}
            for i, v in enumerate(variables)
        ]
    return {
        "id": id,
        "module": "util:SetVariable2",
        "version": 1,
        "metadata": {"designer": {"x": 300 * (id - 1), "y": 0, "name": label}},
        "parameters": {},
        "mapper": mapper,
    }


def build_json_parse_module(
    id: int, label: str, json_string_var: str = ""
) -> dict:
    """Build a JSON Parse module."""
    return {
        "id": id,
        "module": "json:ParseJSON",
        "version": 1,
        "metadata": {"designer": {"x": 300 * (id - 1), "y": 0, "name": label}},
        "parameters": {},
        "mapper": {"json": json_string_var},
    }


def build_slack_module(
    id: int,
    label: str,
    channel: str = "",
    message: str = "",
    connection_id: Optional[int] = None,
) -> dict:
    """Build a Slack Create Message module."""
    module = {
        "id": id,
        "module": "slack:ActionCreateMessage",
        "version": 1,
        "metadata": {"designer": {"x": 300 * (id - 1), "y": 0, "name": label}},
        "parameters": {},
        "mapper": {"channel": channel, "text": message},
    }
    if connection_id:
        module["parameters"]["connection"] = connection_id
    return module


def build_email_module(id: int, label: str, to: str = "", subject: str = "", body: str = "") -> dict:
    """Build an Email Send module."""
    return {
        "id": id,
        "module": "email:ActionSendEmail",
        "version": 1,
        "metadata": {"designer": {"x": 300 * (id - 1), "y": 0, "name": label}},
        "parameters": {},
        "mapper": {"to": to, "subject": subject, "html": body},
    }


def build_google_sheets_module(id: int, label: str, action: str = "addRow") -> dict:
    """Build a Google Sheets module."""
    return {
        "id": id,
        "module": f"google-sheets:{action}",
        "version": 2,
        "metadata": {"designer": {"x": 300 * (id - 1), "y": 0, "name": label}},
        "parameters": {},
        "mapper": {},
    }


# ── Plan → Module extraction ─────────────────────────────────

_KEYWORD_MAP = [
    (["webhook", "trigger", "incoming"], "webhook"),
    (["http", "request", "fetch", "api", "rest", "url", "curl"], "http"),
    (["slack", "message channel"], "slack"),
    (["email", "send email", "mail", "smtp"], "email"),
    (["parse", "json", "transform"], "json"),
    (["filter", "route", "branch", "condition", "if"], "router"),
    (["google sheet", "spreadsheet", "gsheet"], "sheets"),
    (["variable", "set", "store", "assign"], "variable"),
]


def _classify_step(step: dict) -> str:
    """Classify a processing step by keywords in description/app fields."""
    text = " ".join([
        step.get("description", ""),
        step.get("app", ""),
        step.get("module", ""),
    ]).lower()

    for keywords, module_type in _KEYWORD_MAP:
        for kw in keywords:
            if kw in text:
                return module_type
    return "http"  # Default to HTTP module for unrecognized steps


def extract_modules_from_plan(plan_dict: dict) -> list:
    """Convert plan_dict processing_steps to Make.com module dicts.

    Reads plan_dict["processing_steps"] and maps each to an appropriate
    Make.com module type based on keyword matching.

    Args:
        plan_dict: The plan dict from POST /plan response.

    Returns:
        List of Make.com module dicts with sequential IDs starting from 1.
    """
    steps = plan_dict.get("processing_steps", [])
    if not steps:
        return []

    modules = []
    for i, step in enumerate(steps):
        step_id = i + 1
        label = step.get("description", f"Step {step_id}")[:60]
        module_type = _classify_step(step)

        if module_type == "webhook":
            mod = build_webhook_trigger(label=label)
            mod["id"] = step_id
        elif module_type == "http":
            url = step.get("inputs", "")
            mod = build_http_request_module(step_id, label, url=url)
        elif module_type == "slack":
            mod = build_slack_module(step_id, label)
        elif module_type == "email":
            mod = build_email_module(step_id, label)
        elif module_type == "json":
            mod = build_json_parse_module(step_id, label)
        elif module_type == "router":
            mod = build_router_module(step_id, label)
        elif module_type == "sheets":
            mod = build_google_sheets_module(step_id, label)
        else:
            mod = build_set_variable_module(step_id, label)

        modules.append(mod)

    return modules


# ── Blueprint assembly ───────────────────────────────────────

def build_blueprint(
    plan_dict: dict,
    scenario_name: str,
    webhook_url: Optional[str] = None,
) -> dict:
    """Build a complete Make.com blueprint from a plan_dict.

    Args:
        plan_dict: The plan dict from POST /plan response.
        scenario_name: Name for the Make.com scenario.
        webhook_url: Optional webhook URL if a trigger webhook exists.

    Returns:
        Valid Make.com blueprint JSON structure.
    """
    modules = extract_modules_from_plan(plan_dict)

    # If no modules were extracted, add a placeholder webhook trigger
    if not modules:
        modules = [build_webhook_trigger(label=f"{scenario_name} Trigger")]

    # Ensure first module is a trigger if none exists
    first_module_str = modules[0].get("module", "")
    if "gateway" not in first_module_str.lower() and "webhook" not in first_module_str.lower():
        trigger = build_webhook_trigger(label=f"{scenario_name} Trigger")
        trigger["id"] = 0
        # Shift existing module IDs
        for m in modules:
            m["id"] = m["id"] + 1
        modules.insert(0, trigger)

    blueprint = {
        "name": scenario_name,
        "flow": [
            {
                "id": modules[0]["id"],
                "module": modules[0]["module"],
                "version": modules[0].get("version", 1),
                "metadata": modules[0].get("metadata", {}),
                "parameters": modules[0].get("parameters", {}),
                "mapper": modules[0].get("mapper", {}),
            }
        ],
        "metadata": {
            "instant": False,
            "version": 1,
            "scenario": {
                "roundtrips": 1,
                "maxErrors": 3,
                "autoCommit": True,
                "autoCommitTriggerLast": True,
                "sequential": False,
                "confidential": False,
                "dataloss": False,
                "dlq": False,
                "freshVariables": False,
            },
        },
    }

    # Add remaining modules to the flow
    for mod in modules[1:]:
        blueprint["flow"].append({
            "id": mod["id"],
            "module": mod["module"],
            "version": mod.get("version", 1),
            "metadata": mod.get("metadata", {}),
            "parameters": mod.get("parameters", {}),
            "mapper": mod.get("mapper", {}),
        })

    return blueprint


def validate_blueprint(blueprint: dict) -> tuple[bool, list]:
    """Validate a Make.com blueprint structure.

    Args:
        blueprint: Blueprint dict to validate.

    Returns:
        Tuple of (is_valid, list_of_error_strings).
    """
    errors = []

    if not blueprint.get("name"):
        errors.append("Blueprint missing 'name'")

    flow = blueprint.get("flow")
    if not flow:
        errors.append("Blueprint missing 'flow' or flow is empty")
    elif not isinstance(flow, list):
        errors.append("Blueprint 'flow' must be a list")
    else:
        for i, module in enumerate(flow):
            if not module.get("id") and module.get("id") != 0:
                errors.append(f"Module at index {i} missing 'id'")
            if not module.get("module"):
                errors.append(f"Module at index {i} missing 'module' type")

    return (len(errors) == 0, errors)
