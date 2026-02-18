"""
Self-Heal Make Export

Deterministically repairs structural errors in Make.com export JSON
based on the validation report from validate_make_export.py.

Input:
    blueprint (dict) — Make.com blueprint JSON (may be invalid)
    registry (dict, optional) — loaded module registry for credential/param defaults
    max_retries (int) — maximum repair passes (default 2)

Output:
    dict with:
        - success: bool — True if final validation has zero errors
        - retries_used: int — number of repair passes executed
        - repair_log: list of repair action dicts
        - final_validation: dict — validation report after last pass
        - blueprint: dict — the (potentially repaired) blueprint

Supported repairs by rule_id:
    MR-002: Missing/invalid name → "Untitled Scenario"
    MR-003: Missing/invalid flow → empty array
    MR-004: Missing/invalid metadata → default metadata object
    MF-001: Non-integer module ID → int conversion
    MF-002: Duplicate module IDs → reassign unique IDs
    MF-003: Trigger (id=1) not first in flow → reorder
    MF-006: Missing version field → default to 1
    MM-001: Missing metadata.designer → add default x,y
    MM-002: Missing parameters → add empty dict
    MM-004: onerror not array → wrap in array
    MM-005: Invalid onerror directive → remove invalid entries
    MT-001: Router missing routes → add empty routes array
    MT-003: Route missing flow → add empty flow array
    MC-003: Missing __IMTCONN__ for credential module → add from registry
    MD-001: Missing metadata.scenario → add default
    MD-002: Missing maxErrors → add default (3)
    MD-003: Missing designer.orphans → add empty array

Deterministic. No network calls. No AI reasoning. Pure structural repair.
"""

import copy
import re
from datetime import datetime

from tools.validate_make_export import validate_make_export

# Designer position defaults
X_SPACING = 300
Y_START = 0

VALID_ONERROR_DIRECTIVES = {"ignore", "resume", "break", "rollback", "commit"}
CREDENTIAL_PLACEHOLDER_PATTERN = re.compile(r"^__[A-Z_]+_CONNECTION__$")

# Rule IDs that this tool can repair
REPAIRABLE_RULES = {
    "MR-002", "MR-003", "MR-004",
    "MF-001", "MF-002", "MF-003", "MF-006",
    "MM-001", "MM-002", "MM-004", "MM-005",
    "MT-001", "MT-003",
    "MC-003",
    "MD-001", "MD-002", "MD-003",
}


def self_heal_make_export(blueprint, registry=None, max_retries=2):
    """Attempt to repair a Make.com blueprint based on validation errors.

    Runs validation, identifies repairable errors, applies deterministic fixes,
    then re-validates. Repeats up to max_retries times or until all errors are
    resolved (or only non-repairable errors remain).

    Args:
        blueprint: Make.com blueprint dict (will be deep-copied, not mutated).
        registry: Optional loaded module registry dict.
        max_retries: Maximum number of repair passes (default 2).

    Returns:
        dict with success, retries_used, repair_log, final_validation, blueprint.
    """
    bp = copy.deepcopy(blueprint)
    repair_log = []
    retries_used = 0

    modules_dict = {}
    if registry:
        modules_dict = registry.get("modules", {})

    for attempt in range(1, max_retries + 1):
        # Validate current state
        report = validate_make_export(bp, registry)

        if report["valid"]:
            # Already valid, no repairs needed
            return {
                "success": True,
                "retries_used": retries_used,
                "repair_log": repair_log,
                "final_validation": report,
                "blueprint": bp
            }

        # Identify repairable errors
        repairable = [e for e in report["errors"] if e["rule_id"] in REPAIRABLE_RULES]

        if not repairable:
            # Only non-repairable errors remain — stop
            return {
                "success": False,
                "retries_used": retries_used,
                "repair_log": repair_log,
                "final_validation": report,
                "blueprint": bp
            }

        retries_used = attempt

        # Apply repairs for this pass
        pass_repairs = _apply_repairs(bp, repairable, modules_dict)
        for r in pass_repairs:
            r["pass"] = attempt
        repair_log.extend(pass_repairs)

    # Final validation after all passes
    final_report = validate_make_export(bp, registry)

    return {
        "success": final_report["valid"],
        "retries_used": retries_used,
        "repair_log": repair_log,
        "final_validation": final_report,
        "blueprint": bp
    }


def _apply_repairs(bp, errors, modules_dict):
    """Apply deterministic repairs for a list of errors.

    Dispatches each error to its specific repair function. Modifies bp in place.

    Args:
        bp: Blueprint dict (mutated).
        errors: List of repairable error dicts.
        modules_dict: Registry modules dict for lookups.

    Returns:
        List of repair action dicts documenting what was changed.
    """
    repairs = []

    # Group errors by rule_id to batch related repairs
    errors_by_rule = {}
    for e in errors:
        errors_by_rule.setdefault(e["rule_id"], []).append(e)

    # Apply repairs in a deterministic order (sorted by rule_id)
    for rule_id in sorted(errors_by_rule.keys()):
        rule_errors = errors_by_rule[rule_id]
        handler = _REPAIR_HANDLERS.get(rule_id)
        if handler:
            result = handler(bp, rule_errors, modules_dict)
            repairs.extend(result)

    return repairs


# ===== Root Structure Repairs =====

def _repair_missing_name(bp, errors, modules_dict):
    """MR-002: Missing or invalid name → set default."""
    bp["name"] = bp.get("name") if isinstance(bp.get("name"), str) else "Untitled Scenario"
    if not isinstance(bp.get("name"), str):
        bp["name"] = "Untitled Scenario"
    return [{"rule_id": "MR-002", "action": "Set blueprint name to 'Untitled Scenario'"}]


def _repair_missing_flow(bp, errors, modules_dict):
    """MR-003: Missing or invalid flow → set empty array."""
    if not isinstance(bp.get("flow"), list):
        bp["flow"] = []
        return [{"rule_id": "MR-003", "action": "Initialized missing flow as empty array"}]
    return []


def _repair_missing_metadata(bp, errors, modules_dict):
    """MR-004: Missing or invalid metadata → set default metadata object."""
    if not isinstance(bp.get("metadata"), dict):
        bp["metadata"] = {
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
                "freshVariables": False
            },
            "designer": {
                "orphans": []
            }
        }
        return [{"rule_id": "MR-004", "action": "Created default metadata object"}]
    return []


# ===== Flow Integrity Repairs =====

def _repair_non_integer_id(bp, errors, modules_dict):
    """MF-001: Non-integer module ID → attempt int conversion."""
    repairs = []
    all_modules = _collect_all_modules(bp)

    for mod in all_modules:
        mid = mod.get("id")
        if mid is not None and not isinstance(mid, int):
            try:
                mod["id"] = int(mid)
                repairs.append({
                    "rule_id": "MF-001",
                    "action": f"Converted module ID '{mid}' to integer {mod['id']}",
                    "module_id": mod["id"]
                })
            except (ValueError, TypeError):
                # Can't convert — skip (will remain as error)
                pass

    return repairs


def _repair_duplicate_ids(bp, errors, modules_dict):
    """MF-002: Duplicate module IDs → reassign unique IDs."""
    repairs = []
    all_modules = _collect_all_modules(bp)

    seen = set()
    max_id = 0
    for mod in all_modules:
        mid = mod.get("id")
        if isinstance(mid, int) and mid > max_id:
            max_id = mid

    for mod in all_modules:
        mid = mod.get("id")
        if mid in seen:
            max_id += 1
            old_id = mid
            mod["id"] = max_id
            repairs.append({
                "rule_id": "MF-002",
                "action": f"Reassigned duplicate module ID {old_id} → {max_id}",
                "module_id": max_id
            })
        seen.add(mod.get("id"))

    return repairs


def _repair_trigger_not_first(bp, errors, modules_dict):
    """MF-003: Trigger (id=1) not first in flow → reorder."""
    flow = bp.get("flow")
    if not isinstance(flow, list) or not flow:
        return []

    # Find the trigger module (id=1) in the flow
    trigger_idx = None
    for i, mod in enumerate(flow):
        if isinstance(mod, dict) and mod.get("id") == 1:
            trigger_idx = i
            break

    if trigger_idx is not None and trigger_idx != 0:
        trigger = flow.pop(trigger_idx)
        flow.insert(0, trigger)
        return [{
            "rule_id": "MF-003",
            "action": f"Moved trigger (id=1) from index {trigger_idx} to index 0"
        }]

    return []


def _repair_missing_version(bp, errors, modules_dict):
    """MF-006: Missing version field → default to 1."""
    repairs = []
    all_modules = _collect_all_modules(bp)

    for mod in all_modules:
        mid = mod.get("id", "?")
        if "version" not in mod or not isinstance(mod.get("version"), int):
            # Check registry for correct version
            mod_type = mod.get("module", "")
            reg_entry = modules_dict.get(mod_type, {})
            mod["version"] = reg_entry.get("version", 1)
            repairs.append({
                "rule_id": "MF-006",
                "action": f"Set module {mid} version to {mod['version']}",
                "module_id": mid
            })

    return repairs


# ===== Module Structure Repairs =====

def _repair_missing_designer(bp, errors, modules_dict):
    """MM-001: Missing metadata.designer → add default x,y positions."""
    repairs = []
    all_modules = _collect_all_modules(bp)

    for i, mod in enumerate(all_modules):
        mid = mod.get("id", "?")
        mod_meta = mod.get("metadata")

        needs_repair = (
            not isinstance(mod_meta, dict)
            or not isinstance(mod_meta.get("designer"), dict)
            or "x" not in mod_meta.get("designer", {})
            or "y" not in mod_meta.get("designer", {})
        )

        if needs_repair:
            if not isinstance(mod_meta, dict):
                mod["metadata"] = {}
            mod["metadata"]["designer"] = {
                "x": i * X_SPACING,
                "y": Y_START
            }
            repairs.append({
                "rule_id": "MM-001",
                "action": f"Added designer position to module {mid} (x={i * X_SPACING}, y={Y_START})",
                "module_id": mid
            })

    return repairs


def _repair_missing_parameters(bp, errors, modules_dict):
    """MM-002: Missing parameters → add empty dict."""
    repairs = []
    all_modules = _collect_all_modules(bp)

    for mod in all_modules:
        mid = mod.get("id", "?")
        is_router = mod.get("module") == "builtin:BasicRouter"

        if not is_router and "parameters" not in mod:
            mod["parameters"] = {}
            repairs.append({
                "rule_id": "MM-002",
                "action": f"Added empty parameters dict to module {mid}",
                "module_id": mid
            })

    return repairs


def _repair_onerror_not_array(bp, errors, modules_dict):
    """MM-004: onerror not array → wrap in array."""
    repairs = []
    all_modules = _collect_all_modules(bp)

    for mod in all_modules:
        mid = mod.get("id", "?")
        onerror = mod.get("onerror")

        if onerror is not None and not isinstance(onerror, list):
            if isinstance(onerror, str) and onerror in VALID_ONERROR_DIRECTIVES:
                mod["onerror"] = [{"directive": onerror}]
                repairs.append({
                    "rule_id": "MM-004",
                    "action": f"Wrapped module {mid} onerror '{onerror}' in array",
                    "module_id": mid
                })
            elif isinstance(onerror, dict) and "directive" in onerror:
                mod["onerror"] = [onerror]
                repairs.append({
                    "rule_id": "MM-004",
                    "action": f"Wrapped module {mid} onerror dict in array",
                    "module_id": mid
                })
            else:
                # Unrecognized format — remove it
                del mod["onerror"]
                repairs.append({
                    "rule_id": "MM-004",
                    "action": f"Removed unrecognizable onerror from module {mid}",
                    "module_id": mid
                })

    return repairs


def _repair_invalid_onerror_directive(bp, errors, modules_dict):
    """MM-005: Invalid onerror directive → remove invalid entries."""
    repairs = []
    all_modules = _collect_all_modules(bp)

    for mod in all_modules:
        mid = mod.get("id", "?")
        onerror = mod.get("onerror")

        if isinstance(onerror, list):
            valid_entries = []
            removed = []
            for entry in onerror:
                directive = entry.get("directive") if isinstance(entry, dict) else None
                if directive in VALID_ONERROR_DIRECTIVES:
                    valid_entries.append(entry)
                else:
                    removed.append(str(directive))

            if removed:
                mod["onerror"] = valid_entries if valid_entries else None
                if not valid_entries:
                    del mod["onerror"]
                repairs.append({
                    "rule_id": "MM-005",
                    "action": f"Removed invalid onerror directives from module {mid}: {removed}",
                    "module_id": mid
                })

    return repairs


# ===== Router/Routes Repairs =====

def _repair_missing_routes_array(bp, errors, modules_dict):
    """MT-001: Router missing routes → add empty routes array."""
    repairs = []
    _repair_routes_recursive(bp.get("flow", []), "MT-001", repairs, _add_routes_array)
    return repairs


def _add_routes_array(mod, repairs):
    """Add an empty routes array to a router module."""
    if not isinstance(mod.get("routes"), list):
        mod["routes"] = []
        repairs.append({
            "rule_id": "MT-001",
            "action": f"Added empty routes array to router module {mod.get('id', '?')}",
            "module_id": mod.get("id")
        })


def _repair_missing_route_flow(bp, errors, modules_dict):
    """MT-003: Route missing flow → add empty flow array."""
    repairs = []
    _repair_routes_recursive(bp.get("flow", []), "MT-003", repairs, _add_route_flows)
    return repairs


def _add_route_flows(mod, repairs):
    """Add empty flow arrays to routes that are missing them."""
    routes = mod.get("routes")
    if isinstance(routes, list):
        for ri, route in enumerate(routes):
            if isinstance(route, dict) and not isinstance(route.get("flow"), list):
                route["flow"] = []
                repairs.append({
                    "rule_id": "MT-003",
                    "action": f"Added empty flow array to router {mod.get('id', '?')} route[{ri}]",
                    "module_id": mod.get("id")
                })


def _repair_routes_recursive(flow_items, rule_id, repairs, repair_fn):
    """Recursively find routers and apply a repair function."""
    for item in flow_items:
        if not isinstance(item, dict):
            continue

        if item.get("module") == "builtin:BasicRouter":
            repair_fn(item, repairs)

        # Recurse into routes
        routes = item.get("routes")
        if isinstance(routes, list):
            for route in routes:
                if isinstance(route, dict) and isinstance(route.get("flow"), list):
                    _repair_routes_recursive(route["flow"], rule_id, repairs, repair_fn)


# ===== Credential Repairs =====

def _repair_missing_credential(bp, errors, modules_dict):
    """MC-003: Missing __IMTCONN__ for credential module → add from registry."""
    if not modules_dict:
        return []

    repairs = []
    all_modules = _collect_all_modules(bp)

    for mod in all_modules:
        mid = mod.get("id", "?")
        mod_type = mod.get("module", "")
        reg_entry = modules_dict.get(mod_type)

        if reg_entry and reg_entry.get("requires_credential"):
            params = mod.get("parameters")
            if not isinstance(params, dict):
                mod["parameters"] = {}
                params = mod["parameters"]

            if "__IMTCONN__" not in params:
                placeholder = reg_entry.get("credential_placeholder", "__UNKNOWN_CONNECTION__")
                params["__IMTCONN__"] = placeholder
                repairs.append({
                    "rule_id": "MC-003",
                    "action": f"Added __IMTCONN__='{placeholder}' to module {mid} ({mod_type})",
                    "module_id": mid
                })

    return repairs


# ===== Designer/Metadata Repairs =====

def _repair_missing_scenario_metadata(bp, errors, modules_dict):
    """MD-001: Missing metadata.scenario → add default."""
    metadata = bp.get("metadata")
    if not isinstance(metadata, dict):
        # MR-004 handler should have fixed this already
        return []

    if not isinstance(metadata.get("scenario"), dict):
        metadata["scenario"] = {
            "roundtrips": 1,
            "maxErrors": 3,
            "autoCommit": True,
            "autoCommitTriggerLast": True,
            "sequential": False,
            "confidential": False,
            "dataloss": False,
            "dlq": False,
            "freshVariables": False
        }
        return [{"rule_id": "MD-001", "action": "Added default metadata.scenario object"}]
    return []


def _repair_missing_max_errors(bp, errors, modules_dict):
    """MD-002: Missing maxErrors → add default (3)."""
    metadata = bp.get("metadata", {})
    scenario = metadata.get("scenario", {})

    if isinstance(scenario, dict):
        if "maxErrors" not in scenario or not isinstance(scenario.get("maxErrors"), int):
            scenario["maxErrors"] = 3
            return [{"rule_id": "MD-002", "action": "Set metadata.scenario.maxErrors to 3"}]
    return []


def _repair_missing_designer_orphans(bp, errors, modules_dict):
    """MD-003: Missing designer.orphans → add empty array."""
    metadata = bp.get("metadata", {})

    if isinstance(metadata, dict):
        designer = metadata.get("designer")
        if not isinstance(designer, dict):
            metadata["designer"] = {"orphans": []}
            return [{"rule_id": "MD-003", "action": "Added metadata.designer with empty orphans array"}]
        elif not isinstance(designer.get("orphans"), list):
            designer["orphans"] = []
            return [{"rule_id": "MD-003", "action": "Added empty orphans array to metadata.designer"}]
    return []


# ===== Helpers =====

def _collect_all_modules(bp):
    """Collect all module dicts from the flow tree (including router routes).

    Returns a flat list of module dicts (references, not copies).
    """
    flow = bp.get("flow", [])
    if not isinstance(flow, list):
        return []

    modules = []
    _collect_recursive(flow, modules)
    return modules


def _collect_recursive(flow_items, modules):
    """Recursively collect modules from flow items."""
    for item in flow_items:
        if not isinstance(item, dict):
            continue
        modules.append(item)

        routes = item.get("routes")
        if isinstance(routes, list):
            for route in routes:
                if isinstance(route, dict) and isinstance(route.get("flow"), list):
                    _collect_recursive(route["flow"], modules)


# Repair handler dispatch table
_REPAIR_HANDLERS = {
    # Root structure
    "MR-002": _repair_missing_name,
    "MR-003": _repair_missing_flow,
    "MR-004": _repair_missing_metadata,
    # Flow integrity
    "MF-001": _repair_non_integer_id,
    "MF-002": _repair_duplicate_ids,
    "MF-003": _repair_trigger_not_first,
    "MF-006": _repair_missing_version,
    # Module structure
    "MM-001": _repair_missing_designer,
    "MM-002": _repair_missing_parameters,
    "MM-004": _repair_onerror_not_array,
    "MM-005": _repair_invalid_onerror_directive,
    # Router/Routes
    "MT-001": _repair_missing_routes_array,
    "MT-003": _repair_missing_route_flow,
    # Credentials
    "MC-003": _repair_missing_credential,
    # Designer/Metadata
    "MD-001": _repair_missing_scenario_metadata,
    "MD-002": _repair_missing_max_errors,
    "MD-003": _repair_missing_designer_orphans,
}


# --- Self-check ---
if __name__ == "__main__":
    import json
    from tools.module_registry_loader import load_module_registry
    from tools.generate_make_export import generate_make_export

    print("=== Self-Heal Make Export Self-Check ===\n")

    registry = load_module_registry()

    # --- Test 1: Already valid blueprint (no repairs needed) ---
    print("Test 1: Already valid blueprint")
    valid_spec = {
        "spec_version": "1.0.0",
        "scenario": {"name": "Valid", "description": "Test.", "slug": "valid-test"},
        "trigger": {
            "id": 1, "type": "webhook", "module": "gateway:CustomWebHook",
            "label": "WH", "version": 1,
            "parameters": {"hook": "__WEBHOOK_ID__"}, "credential_placeholder": None
        },
        "modules": [
            {
                "id": 2, "label": "Parse", "app": "json", "module": "json:ParseJSON",
                "module_type": "transformer", "version": 1,
                "parameters": {}, "mapper": {"json": "{{1.body}}"},
                "credential_placeholder": None, "onerror": None, "resume_value": None
            }
        ],
        "connections": [{"from": "trigger", "to": 2, "filter": None, "label": None}],
        "error_handling": {"default_strategy": "ignore", "max_errors": 3, "module_overrides": []},
        "metadata": {"created_at": "2026-02-16T12:00:00Z", "original_request": "Test."}
    }
    valid_bp = generate_make_export(valid_spec)
    result1 = self_heal_make_export(valid_bp, registry)

    assert result1["success"] is True
    assert result1["retries_used"] == 0
    assert len(result1["repair_log"]) == 0
    print(f"  Success: {result1['success']}, Retries: {result1['retries_used']}, Repairs: {len(result1['repair_log'])}")
    print("  [OK] No repairs needed for valid blueprint")

    # --- Test 2: Missing metadata, designer, credentials ---
    print("\nTest 2: Missing metadata, designer positions, credentials")
    broken_bp2 = {
        "name": "Broken",
        "flow": [
            {
                "id": 1, "module": "gateway:CustomWebHook", "version": 1,
                "parameters": {"hook": "__WEBHOOK_ID__"}
                # Missing: metadata.designer
            },
            {
                "id": 2, "module": "json:ParseJSON", "version": 1,
                "mapper": {"json": "{{1.body}}"}
                # Missing: parameters, metadata.designer
            },
            {
                "id": 3, "module": "slack:PostMessage", "version": 1,
                "mapper": {"channel": "#test", "text": "{{2.msg}}"}
                # Missing: parameters, __IMTCONN__, metadata.designer
            }
        ]
        # Missing: metadata (entire)
    }

    result2 = self_heal_make_export(broken_bp2, registry)
    print(f"  Success: {result2['success']}, Retries: {result2['retries_used']}")
    print(f"  Repairs applied: {len(result2['repair_log'])}")
    for r in result2["repair_log"]:
        print(f"    [{r['rule_id']}] {r['action']}")
    assert result2["success"] is True, f"Should heal, got errors: {[e['message'] for e in result2['final_validation']['errors']]}"
    # Verify credential was added
    slack_mod = result2["blueprint"]["flow"][2]
    assert "__IMTCONN__" in slack_mod["parameters"]
    assert slack_mod["parameters"]["__IMTCONN__"] == "__SLACK_CONNECTION__"
    print("  [OK] Healed: metadata, designer positions, parameters, credentials")

    # --- Test 3: Duplicate IDs + trigger not first ---
    print("\nTest 3: Duplicate IDs + trigger not first")
    broken_bp3 = {
        "name": "Dup IDs",
        "flow": [
            {
                "id": 2, "module": "json:ParseJSON", "version": 1,
                "parameters": {}, "mapper": {"json": "{{1.body}}"},
                "metadata": {"designer": {"x": 300, "y": 0}}
            },
            {
                "id": 1, "module": "gateway:CustomWebHook", "version": 1,
                "parameters": {"hook": "__WEBHOOK_ID__"}, "mapper": {},
                "metadata": {"designer": {"x": 0, "y": 0}}
            },
            {
                "id": 2, "module": "slack:PostMessage", "version": 1,
                "parameters": {"__IMTCONN__": "__SLACK_CONNECTION__"},
                "mapper": {"channel": "#test", "text": "hi"},
                "metadata": {"designer": {"x": 600, "y": 0}}
            }
        ],
        "metadata": {
            "version": 1,
            "scenario": {"roundtrips": 1, "maxErrors": 3, "autoCommit": True,
                         "autoCommitTriggerLast": True, "sequential": False,
                         "confidential": False, "dataloss": False, "dlq": False,
                         "freshVariables": False},
            "designer": {"orphans": []}
        }
    }

    result3 = self_heal_make_export(broken_bp3, registry)
    print(f"  Success: {result3['success']}, Retries: {result3['retries_used']}")
    for r in result3["repair_log"]:
        print(f"    [{r['rule_id']}] {r['action']}")
    assert result3["success"] is True, f"Should heal dup IDs, got errors: {[e['message'] for e in result3['final_validation']['errors']]}"
    # Verify trigger is now first
    assert result3["blueprint"]["flow"][0]["id"] == 1
    # Verify no duplicate IDs
    ids = [m["id"] for m in result3["blueprint"]["flow"]]
    assert len(ids) == len(set(ids)), f"Duplicate IDs remain: {ids}"
    print("  [OK] Healed: trigger reordered, duplicate IDs reassigned")

    # --- Test 4: Router with missing routes and route flows ---
    print("\nTest 4: Router with structural issues")
    broken_bp4 = {
        "name": "Router Issues",
        "flow": [
            {
                "id": 1, "module": "gateway:CustomWebHook", "version": 1,
                "parameters": {"hook": "__WEBHOOK_ID__"}, "mapper": {},
                "metadata": {"designer": {"x": 0, "y": 0}}
            },
            {
                "id": 2, "module": "builtin:BasicRouter", "version": 1,
                "mapper": None,
                "metadata": {"designer": {"x": 300, "y": 0}},
                "routes": [
                    {"flow": [
                        {
                            "id": 3, "module": "slack:PostMessage", "version": 1,
                            "parameters": {"__IMTCONN__": "__SLACK_CONNECTION__"},
                            "mapper": {"channel": "#test", "text": "hi"},
                            "metadata": {"designer": {"x": 600, "y": -100}}
                        }
                    ]},
                    {"not_flow": "broken"}  # Missing flow key
                ]
            }
        ],
        "metadata": {
            "version": 1,
            "scenario": {"roundtrips": 1, "maxErrors": 3, "autoCommit": True,
                         "autoCommitTriggerLast": True, "sequential": False,
                         "confidential": False, "dataloss": False, "dlq": False,
                         "freshVariables": False},
            "designer": {"orphans": []}
        }
    }

    result4 = self_heal_make_export(broken_bp4, registry)
    print(f"  Success: {result4['success']}, Retries: {result4['retries_used']}")
    for r in result4["repair_log"]:
        print(f"    [{r['rule_id']}] {r['action']}")
    # Route flow was added but route is empty → MT-004 is a warning, not error
    router = result4["blueprint"]["flow"][1]
    assert isinstance(router["routes"][1].get("flow"), list), "Missing route flow should be added"
    print("  [OK] Healed: missing route flow array added")

    # --- Test 5: onerror string instead of array ---
    print("\nTest 5: onerror string instead of array")
    broken_bp5 = {
        "name": "Onerror Fix",
        "flow": [
            {
                "id": 1, "module": "gateway:CustomWebHook", "version": 1,
                "parameters": {"hook": "__WEBHOOK_ID__"}, "mapper": {},
                "metadata": {"designer": {"x": 0, "y": 0}}
            },
            {
                "id": 2, "module": "json:ParseJSON", "version": 1,
                "parameters": {}, "mapper": {"json": "{{1.body}}"},
                "metadata": {"designer": {"x": 300, "y": 0}},
                "onerror": "ignore"  # Should be [{"directive": "ignore"}]
            }
        ],
        "metadata": {
            "version": 1,
            "scenario": {"roundtrips": 1, "maxErrors": 3, "autoCommit": True,
                         "autoCommitTriggerLast": True, "sequential": False,
                         "confidential": False, "dataloss": False, "dlq": False,
                         "freshVariables": False},
            "designer": {"orphans": []}
        }
    }

    result5 = self_heal_make_export(broken_bp5, registry)
    print(f"  Success: {result5['success']}, Retries: {result5['retries_used']}")
    for r in result5["repair_log"]:
        print(f"    [{r['rule_id']}] {r['action']}")
    assert result5["success"] is True
    # Verify onerror was wrapped
    assert result5["blueprint"]["flow"][1]["onerror"] == [{"directive": "ignore"}]
    print("  [OK] Healed: onerror string wrapped in array")

    # --- Test 6: Non-repairable errors (stops gracefully) ---
    print("\nTest 6: Non-repairable errors")
    non_repairable_bp = {
        "name": "Bad Modules",
        "flow": [
            {
                "id": 1, "module": "gateway:CustomWebHook", "version": 1,
                "parameters": {"hook": "__WEBHOOK_ID__"}, "mapper": {},
                "metadata": {"designer": {"x": 0, "y": 0}}
            },
            {
                "id": 2, "module": "nonexistent:FakeModule", "version": 1,
                "parameters": {}, "mapper": {},
                "metadata": {"designer": {"x": 300, "y": 0}}
            }
        ],
        "metadata": {
            "version": 1,
            "scenario": {"roundtrips": 1, "maxErrors": 3, "autoCommit": True,
                         "autoCommitTriggerLast": True, "sequential": False,
                         "confidential": False, "dataloss": False, "dlq": False,
                         "freshVariables": False},
            "designer": {"orphans": []}
        }
    }

    result6 = self_heal_make_export(non_repairable_bp, registry)
    print(f"  Success: {result6['success']}, Retries: {result6['retries_used']}")
    assert result6["success"] is False, "Should fail for non-repairable errors"
    assert result6["retries_used"] == 0, "Should not retry when only non-repairable errors exist"
    print("  [OK] Correctly stops on non-repairable errors (0 retries)")

    # --- Test 7: Max retries respected ---
    print("\nTest 7: Max retries cap")
    # Use an empty blueprint that has many issues
    result7 = self_heal_make_export({}, registry, max_retries=2)
    assert result7["retries_used"] <= 2
    print(f"  Retries: {result7['retries_used']} (max 2)")
    print("  [OK] Max retries respected")

    # --- Test 8: Determinism ---
    print("\nTest 8: Determinism")
    r8a = self_heal_make_export(broken_bp2, registry)
    r8b = self_heal_make_export(broken_bp2, registry)
    assert r8a["success"] == r8b["success"]
    assert r8a["retries_used"] == r8b["retries_used"]
    assert len(r8a["repair_log"]) == len(r8b["repair_log"])
    assert json.dumps(r8a["blueprint"], sort_keys=True) == json.dumps(r8b["blueprint"], sort_keys=True)
    print("  [OK] Deterministic output")

    # --- Test 9: Original blueprint not mutated ---
    print("\nTest 9: Input not mutated")
    original_copy = copy.deepcopy(broken_bp2)
    _ = self_heal_make_export(broken_bp2, registry)
    assert broken_bp2 == original_copy, "Original blueprint was mutated!"
    print("  [OK] Original blueprint unchanged")

    print(f"\n=== All self_heal_make_export checks passed ===")
