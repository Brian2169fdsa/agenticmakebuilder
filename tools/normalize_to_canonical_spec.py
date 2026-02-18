"""
Normalize to Canonical Spec

Deterministically transforms an agent plan dictionary into a canonical spec
that conforms to canonical_spec_schema.json.

Input:  plan (dict) — structured plan from agent reasoning
        registry (dict) — loaded module registry
        original_request (str) — user's natural language request
        created_at (str, optional) — ISO 8601 timestamp (for deterministic testing)

Output: dict — canonical spec conforming to the schema

This tool does NOT validate the spec (that is validate_canonical_spec's job).
This tool does NOT generate Make.com JSON (that is Phase 2).

Deterministic: identical input always produces identical output.
No randomness. No network calls. No conversation context.
"""

import re
from datetime import datetime, timezone

from tools.module_registry_loader import load_module_registry, get_module


SPEC_VERSION = "1.0.0"


def normalize_to_canonical_spec(plan, registry, original_request, created_at=None):
    """Transform an agent plan dict into a canonical spec.

    The plan dict has this structure:
    {
        "scenario_name": str (required),
        "scenario_description": str (required),
        "trigger": {
            "type": "webhook" | "schedule" | "app_event",
            "module": "gateway:CustomWebHook" (Make module ID),
            "label": str,
            "parameters": dict (optional),
            "webhook": dict (optional — name, data_structure)
        },
        "steps": [
            {
                "module": "json:ParseJSON" (Make module ID),
                "label": str,
                "parameters": dict (optional),
                "mapper": dict (optional),
                "onerror": str | None (optional),
                "resume_value": dict | None (optional)
            },
            ...
        ],
        "connections": [  # optional — if omitted, assumes linear flow
            {
                "from": "trigger" | int (0-based step index),
                "to": int (0-based step index),
                "filter": dict | None (optional),
                "label": str | None (optional)
            },
            ...
        ],
        "error_handling": {
            "default_strategy": str (required),
            "max_errors": int (optional, default 3),
            "module_overrides": list (optional)
        },
        "agent_notes": list[str] (optional)
    }

    Step indices in 'connections' are 0-based into the 'steps' array.
    They get converted to module IDs (trigger=1, steps[0]=2, steps[1]=3, ...).

    If 'connections' is omitted, a linear flow is generated:
    trigger → steps[0] → steps[1] → ... → steps[N-1]

    Args:
        plan: Structured plan dict from agent reasoning.
        registry: Loaded module registry from module_registry_loader.
        original_request: User's original natural language request.
        created_at: Optional ISO 8601 timestamp. If None, uses current UTC time.

    Returns:
        dict — canonical spec conforming to canonical_spec_schema.json.
    """
    if created_at is None:
        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    modules_dict = registry.get("modules", {})

    # --- Scenario ---
    scenario_name = plan.get("scenario_name", "Untitled Scenario")
    scenario_description = plan.get("scenario_description", "")
    slug = _generate_slug(scenario_name)

    scenario = {
        "name": scenario_name,
        "description": scenario_description,
        "slug": slug
    }

    scheduling = plan.get("scheduling")
    if scheduling:
        scenario["scheduling"] = scheduling

    # --- Trigger ---
    plan_trigger = plan.get("trigger", {})
    trigger_module_id = plan_trigger.get("module", "gateway:CustomWebHook")
    reg_trigger = modules_dict.get(trigger_module_id, {})

    trigger = {
        "id": 1,
        "type": plan_trigger.get("type", "webhook"),
        "module": trigger_module_id,
        "label": plan_trigger.get("label", reg_trigger.get("label", "Trigger")),
        "version": reg_trigger.get("version", 1),
        "parameters": plan_trigger.get("parameters", {}),
        "credential_placeholder": _resolve_credential(plan_trigger, reg_trigger)
    }

    # Include webhook config if present
    webhook = plan_trigger.get("webhook")
    if webhook:
        trigger["webhook"] = webhook

    # --- Modules ---
    steps = plan.get("steps", [])
    modules = []

    for i, step in enumerate(steps):
        module_id = i + 2  # trigger=1, first step=2
        step_module = step.get("module", "")
        reg_entry = modules_dict.get(step_module, {})

        app = reg_entry.get("app", _extract_app(step_module))

        mod = {
            "id": module_id,
            "label": step.get("label", reg_entry.get("label", f"Step {i + 1}")),
            "app": app,
            "module": step_module,
            "module_type": reg_entry.get("category", "action"),
            "version": step.get("version", reg_entry.get("version", 1)),
            "parameters": step.get("parameters", {}),
            "mapper": step.get("mapper", {}),
            "credential_placeholder": _resolve_credential(step, reg_entry),
            "onerror": step.get("onerror", None),
            "resume_value": step.get("resume_value", None)
        }
        modules.append(mod)

    # --- Connections ---
    plan_connections = plan.get("connections")

    if plan_connections is None:
        # Linear flow: trigger → step[0] → step[1] → ... → step[N-1]
        connections = _build_linear_connections(len(steps))
    else:
        connections = _normalize_connections(plan_connections, len(steps))

    # --- Error Handling ---
    plan_eh = plan.get("error_handling", {})
    error_handling = {
        "default_strategy": plan_eh.get("default_strategy", "ignore"),
        "max_errors": plan_eh.get("max_errors", 3),
        "module_overrides": _normalize_module_overrides(
            plan_eh.get("module_overrides", []),
            len(steps)
        )
    }

    # --- Metadata ---
    metadata = {
        "created_at": created_at,
        "updated_at": None,
        "original_request": original_request,
        "agent_notes": plan.get("agent_notes", []),
        "build_version": None
    }

    return {
        "spec_version": SPEC_VERSION,
        "scenario": scenario,
        "trigger": trigger,
        "modules": modules,
        "connections": connections,
        "error_handling": error_handling,
        "metadata": metadata
    }


def _generate_slug(name):
    """Generate a deterministic kebab-case slug from a scenario name.

    Args:
        name: Scenario name string.

    Returns:
        Kebab-case slug string.
    """
    # Lowercase, replace non-alphanumeric with hyphens, collapse multiple hyphens
    slug = name.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug or "untitled"


def _extract_app(module_id):
    """Extract app namespace from a module identifier.

    'slack:PostMessage' → 'slack'
    'google-sheets:addRow' → 'google-sheets'
    """
    if ":" in module_id:
        return module_id.split(":")[0]
    return "unknown"


def _resolve_credential(step_or_trigger, registry_entry):
    """Determine the credential placeholder for a module.

    If the plan explicitly sets credential_placeholder, use it.
    Otherwise, derive from registry.

    Returns:
        str or None.
    """
    explicit = step_or_trigger.get("credential_placeholder")
    if explicit is not None:
        return explicit

    if registry_entry.get("requires_credential"):
        return registry_entry.get("credential_placeholder")

    return None


def _build_linear_connections(step_count):
    """Build a linear connection chain: trigger → 2 → 3 → ... → N+1.

    Args:
        step_count: Number of steps (modules).

    Returns:
        List of connection dicts.
    """
    connections = []
    if step_count == 0:
        return connections

    # trigger → first module
    connections.append({
        "from": "trigger",
        "to": 2,
        "filter": None,
        "label": None
    })

    # module[i] → module[i+1]
    for i in range(step_count - 1):
        connections.append({
            "from": i + 2,
            "to": i + 3,
            "filter": None,
            "label": None
        })

    return connections


def _normalize_connections(plan_connections, step_count):
    """Convert plan connections (using step indices) to canonical connections (using module IDs).

    Plan connections use:
      "trigger" → stays as "trigger"
      int (0-based step index) → converted to module ID (index + 2)

    Args:
        plan_connections: List of connection dicts from the plan.
        step_count: Number of steps for bounds context.

    Returns:
        List of normalized connection dicts.
    """
    connections = []

    for conn in plan_connections:
        from_val = conn.get("from")
        to_val = conn.get("to")

        # Normalize 'from'
        if from_val == "trigger":
            norm_from = "trigger"
        elif isinstance(from_val, int):
            norm_from = from_val + 2  # step index → module ID
        else:
            norm_from = from_val  # pass through (let validator catch issues)

        # Normalize 'to'
        if isinstance(to_val, int):
            norm_to = to_val + 2  # step index → module ID
        else:
            norm_to = to_val  # pass through

        normalized = {
            "from": norm_from,
            "to": norm_to,
            "filter": conn.get("filter", None),
            "label": conn.get("label", None)
        }

        connections.append(normalized)

    return connections


def _normalize_module_overrides(overrides, step_count):
    """Normalize error handling module overrides.

    If overrides use step indices (0-based), convert to module IDs.

    Args:
        overrides: List of override dicts from the plan.
        step_count: Number of steps.

    Returns:
        List of normalized override dicts.
    """
    normalized = []

    for ov in overrides:
        mid = ov.get("module_id")
        # If module_id looks like a step index (0-based, less than step_count),
        # and not already a module ID (>= 2), convert it
        if isinstance(mid, int) and mid < 2 and mid < step_count:
            mid = mid + 2

        entry = {
            "module_id": mid,
            "strategy": ov.get("strategy", "ignore")
        }

        rv = ov.get("resume_value")
        if rv is not None:
            entry["resume_value"] = rv

        normalized.append(entry)

    return normalized


# --- Self-check ---
if __name__ == "__main__":
    import json

    print("=== Normalize to Canonical Spec Self-Check ===\n")

    registry = load_module_registry()
    fixed_ts = "2026-02-16T12:00:00Z"

    # Test 1: Simple linear plan (webhook → parse → slack)
    print("Test 1: Linear plan (webhook → parse JSON → Slack)")
    plan1 = {
        "scenario_name": "Form Submission to Slack",
        "scenario_description": "Receives form data via webhook, parses JSON, posts to Slack.",
        "trigger": {
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive Form",
            "parameters": {"hook": "__WEBHOOK_ID__"},
            "webhook": {
                "name": "Form Webhook",
                "data_structure": {"name": {"type": "string"}, "email": {"type": "string"}}
            }
        },
        "steps": [
            {
                "module": "json:ParseJSON",
                "label": "Parse Form Data",
                "mapper": {"json": "{{1.body}}"}
            },
            {
                "module": "slack:PostMessage",
                "label": "Notify Slack",
                "mapper": {"channel": "#notifications", "text": "New: {{2.name}} ({{2.email}})"},
                "onerror": "ignore"
            }
        ],
        # connections omitted → linear flow
        "error_handling": {"default_strategy": "ignore"},
        "agent_notes": ["Assumed webhook payload has name and email fields"]
    }

    spec1 = normalize_to_canonical_spec(plan1, registry, "Send form data to Slack", created_at=fixed_ts)

    # Verify structure
    assert spec1["spec_version"] == "1.0.0"
    assert spec1["scenario"]["slug"] == "form-submission-to-slack"
    assert spec1["trigger"]["id"] == 1
    assert spec1["trigger"]["module"] == "gateway:CustomWebHook"
    assert spec1["trigger"]["credential_placeholder"] is None
    assert len(spec1["modules"]) == 2
    assert spec1["modules"][0]["id"] == 2
    assert spec1["modules"][1]["id"] == 3
    assert spec1["modules"][0]["module_type"] == "transformer"  # from registry
    assert spec1["modules"][1]["credential_placeholder"] == "__SLACK_CONNECTION__"  # from registry
    assert len(spec1["connections"]) == 2
    assert spec1["connections"][0] == {"from": "trigger", "to": 2, "filter": None, "label": None}
    assert spec1["connections"][1] == {"from": 2, "to": 3, "filter": None, "label": None}
    assert spec1["metadata"]["created_at"] == fixed_ts
    assert spec1["metadata"]["original_request"] == "Send form data to Slack"
    print("  [OK] Structure correct")

    # Verify determinism: same input → same output
    spec1b = normalize_to_canonical_spec(plan1, registry, "Send form data to Slack", created_at=fixed_ts)
    assert json.dumps(spec1, sort_keys=True) == json.dumps(spec1b, sort_keys=True)
    print("  [OK] Deterministic (identical input → identical output)")

    # Test 2: Router plan with explicit connections
    print("\nTest 2: Router plan (webhook → router → [slack, sheets])")
    plan2 = {
        "scenario_name": "Priority Router: Slack + Sheets",
        "scenario_description": "Routes high priority to Slack, logs all to Google Sheets.",
        "trigger": {
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive Event",
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
                "parameters": {"spreadsheetId": "__SPREADSHEET_ID__", "sheetId": "__SHEET_ID__"},
                "mapper": {"values": ["{{1.priority}}", "{{1.title}}"]}
            }
        ],
        "connections": [
            {"from": "trigger", "to": 0},
            {
                "from": 0, "to": 1,
                "filter": {
                    "name": "High Priority",
                    "conditions": [[{"a": "{{1.priority}}", "b": "high", "o": "text:equal"}]]
                },
                "label": "High Priority"
            },
            {"from": 0, "to": 2, "label": "All Items"}
        ],
        "error_handling": {
            "default_strategy": "ignore",
            "module_overrides": [
                {"module_id": 3, "strategy": "break"}
            ]
        },
        "agent_notes": ["Router fallback route logs all items to Sheets"]
    }

    spec2 = normalize_to_canonical_spec(plan2, registry, "Route by priority", created_at=fixed_ts)

    assert spec2["scenario"]["slug"] == "priority-router-slack-sheets"
    assert len(spec2["modules"]) == 3
    assert spec2["modules"][0]["module_type"] == "flow_control"
    assert spec2["modules"][0]["id"] == 2  # router
    assert spec2["modules"][1]["id"] == 3  # slack
    assert spec2["modules"][2]["id"] == 4  # sheets
    assert spec2["modules"][2]["credential_placeholder"] == "__GOOGLE_SHEETS_CONNECTION__"

    # Connections: step indices (0,1,2) → module IDs (2,3,4)
    assert spec2["connections"][0] == {"from": "trigger", "to": 2, "filter": None, "label": None}
    assert spec2["connections"][1]["from"] == 2
    assert spec2["connections"][1]["to"] == 3
    assert spec2["connections"][1]["filter"]["name"] == "High Priority"
    assert spec2["connections"][2] == {"from": 2, "to": 4, "filter": None, "label": "All Items"}

    # Error handling overrides preserved as-is (module_id=3 already a module ID)
    assert spec2["error_handling"]["module_overrides"][0]["module_id"] == 3
    print("  [OK] Router structure correct")
    print("  [OK] Connections converted from step indices to module IDs")
    print("  [OK] Registry defaults applied (module_type, credential_placeholder)")

    # Test 3: Slug generation edge cases
    print("\nTest 3: Slug generation")
    assert _generate_slug("Hello World") == "hello-world"
    assert _generate_slug("  Multiple   Spaces  ") == "multiple-spaces"
    assert _generate_slug("Special!@#$Characters") == "special-characters"
    assert _generate_slug("already-kebab-case") == "already-kebab-case"
    assert _generate_slug("CamelCase Test") == "camelcase-test"
    assert _generate_slug("") == "untitled"
    assert _generate_slug("123 Numbers First") == "123-numbers-first"
    print("  [OK] All slug edge cases pass")

    # Test 4: Empty plan (trigger only, no steps)
    print("\nTest 4: Empty plan (trigger only)")
    plan4 = {
        "scenario_name": "Webhook Only",
        "scenario_description": "Just a webhook, no steps.",
        "trigger": {
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive"
        },
        "steps": [],
        "error_handling": {"default_strategy": "ignore"}
    }
    spec4 = normalize_to_canonical_spec(plan4, registry, "Just a webhook", created_at=fixed_ts)
    assert len(spec4["modules"]) == 0
    assert len(spec4["connections"]) == 0
    print("  [OK] Empty plan handled correctly")

    # Test 5: Verify full spec can be serialized to JSON (no non-serializable types)
    print("\nTest 5: JSON serialization")
    json_str = json.dumps(spec1, indent=2)
    roundtrip = json.loads(json_str)
    assert roundtrip == spec1
    print(f"  [OK] Serializes cleanly ({len(json_str)} bytes)")

    print("\n=== All normalize_to_canonical_spec checks passed ===")
