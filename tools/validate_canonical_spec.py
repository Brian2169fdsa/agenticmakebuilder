"""
Canonical Spec Validator

Runs all 35+ validation rules against a canonical spec and produces
a structured validation report.

Input: spec (dict), module_registry (dict from loader)
Output: structured validation report dict

Rule categories:
  SC — Schema Completeness (12 rules)
  MI — Module Integrity (10 rules)
  CI — Connection Integrity (10 rules)
  RR — Router Rules (3 rules)
  CR — Credential Rules (3 rules)
  DM — Data Mapping Rules (4 rules)
  SS — Structural Soundness (5 rules)

Deterministic. No network calls. No conversation context.
"""

import re
from datetime import datetime

from tools.graph_integrity_check import graph_integrity_check
from tools.data_mapping_extractor import extract_data_mappings, validate_data_mappings


SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
CREDENTIAL_PATTERN = re.compile(r"^__[A-Z_]+_CONNECTION__$")
SUSPICIOUS_CRED_PATTERNS = [
    re.compile(r"\b\d{10,}\b"),                # Long numeric IDs
    re.compile(r"sk[-_][a-zA-Z0-9]{20,}"),     # API key patterns
    re.compile(r"xox[bpsar]-[a-zA-Z0-9-]+"),   # Slack tokens
    re.compile(r"ya29\.[a-zA-Z0-9_-]+"),        # Google OAuth tokens
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+"),   # Bearer tokens
]

VALID_TRIGGER_TYPES = {"webhook", "schedule", "app_event"}
VALID_MODULE_TYPES = {"action", "search", "flow_control", "transformer", "aggregator", "iterator", "responder"}
VALID_ERROR_STRATEGIES = {"ignore", "resume", "break", "rollback", "commit"}


def validate_canonical_spec(spec, registry):
    """Run all validation rules against a canonical spec.

    Args:
        spec: The canonical spec dict to validate.
        registry: Loaded module registry dict (from module_registry_loader).

    Returns:
        dict with:
            - valid: bool — True if zero errors
            - errors: list of error dicts
            - warnings: list of warning dicts
            - checks_run: int
            - checks_passed: int
            - checks_failed: int
            - timestamp: str (ISO 8601)
    """
    errors = []
    warnings = []
    checks_run = 0
    checks_passed = 0

    def _error(rule_id, message, module_id=None, context=None):
        errors.append({
            "rule_id": rule_id,
            "severity": "error",
            "message": message,
            "module_id": module_id,
            "context": context or {}
        })

    def _warn(rule_id, message, module_id=None, context=None):
        warnings.append({
            "rule_id": rule_id,
            "severity": "warning",
            "message": message,
            "module_id": module_id,
            "context": context or {}
        })

    def _check(rule_id, condition, error_msg, warn=False, module_id=None, context=None):
        nonlocal checks_run, checks_passed
        checks_run += 1
        if condition:
            checks_passed += 1
            return True
        else:
            if warn:
                _warn(rule_id, error_msg, module_id, context)
            else:
                _error(rule_id, error_msg, module_id, context)
            return False

    modules_dict = registry.get("modules", {})

    # ===== SC — Schema Completeness =====

    # SC-001: spec_version present and matches semver
    sv = spec.get("spec_version")
    _check("SC-001", sv is not None and isinstance(sv, str) and SEMVER_PATTERN.match(sv),
           f"spec_version must be a semver string (X.Y.Z), got: {sv!r}")

    # SC-002: scenario has name, description, slug
    scenario = spec.get("scenario", {})
    _check("SC-002",
           isinstance(scenario, dict) and all(k in scenario for k in ["name", "description", "slug"]),
           "scenario must contain 'name', 'description', and 'slug'")

    # SC-003: scenario.slug matches kebab-case
    slug = scenario.get("slug", "")
    _check("SC-003", isinstance(slug, str) and SLUG_PATTERN.match(slug),
           f"scenario.slug must be kebab-case, got: {slug!r}")

    # SC-004: trigger has id, type, module, label
    trigger = spec.get("trigger", {})
    _check("SC-004",
           isinstance(trigger, dict) and all(k in trigger for k in ["id", "type", "module", "label"]),
           "trigger must contain 'id', 'type', 'module', and 'label'")

    # SC-005: trigger.id is exactly 1
    _check("SC-005", trigger.get("id") == 1,
           f"trigger.id must be 1, got: {trigger.get('id')}")

    # SC-006: trigger.type is valid
    _check("SC-006", trigger.get("type") in VALID_TRIGGER_TYPES,
           f"trigger.type must be one of {VALID_TRIGGER_TYPES}, got: {trigger.get('type')!r}")

    # SC-007: modules is an array
    modules = spec.get("modules")
    _check("SC-007", isinstance(modules, list),
           f"modules must be an array, got: {type(modules).__name__}")
    if not isinstance(modules, list):
        modules = []

    # SC-008: connections is an array
    connections = spec.get("connections")
    _check("SC-008", isinstance(connections, list),
           f"connections must be an array, got: {type(connections).__name__}")
    if not isinstance(connections, list):
        connections = []

    # SC-009: error_handling has default_strategy
    eh = spec.get("error_handling", {})
    _check("SC-009", isinstance(eh, dict) and "default_strategy" in eh,
           "error_handling must contain 'default_strategy'")

    # SC-010: default_strategy is valid
    _check("SC-010", eh.get("default_strategy") in VALID_ERROR_STRATEGIES,
           f"error_handling.default_strategy must be one of {VALID_ERROR_STRATEGIES}, "
           f"got: {eh.get('default_strategy')!r}")

    # SC-011: metadata has created_at and original_request
    metadata = spec.get("metadata", {})
    _check("SC-011",
           isinstance(metadata, dict) and "created_at" in metadata and "original_request" in metadata,
           "metadata must contain 'created_at' and 'original_request'")

    # SC-012: created_at is valid ISO 8601
    created_at = metadata.get("created_at", "")
    valid_ts = False
    if isinstance(created_at, str):
        try:
            datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            valid_ts = True
        except (ValueError, TypeError):
            pass
    _check("SC-012", valid_ts,
           f"metadata.created_at must be valid ISO 8601, got: {created_at!r}")

    # ===== MI — Module Integrity =====

    trigger_id = trigger.get("id", 1)
    all_module_ids = {trigger_id}
    module_id_list = []

    for mod in modules:
        mid = mod.get("id")
        module_id_list.append(mid)
        if mid is not None:
            all_module_ids.add(mid)

    # MI-001: Every module has required keys
    required_module_keys = {"id", "label", "app", "module", "module_type"}
    for mod in modules:
        mid = mod.get("id", "?")
        missing = required_module_keys - set(mod.keys())
        _check("MI-001", len(missing) == 0,
               f"Module {mid} missing required keys: {missing}",
               module_id=mid)

    # MI-002: All module IDs unique (including trigger)
    seen_ids = {trigger_id}
    for mod in modules:
        mid = mod.get("id")
        _check("MI-002", mid not in seen_ids or mid == trigger_id and mod is modules[0],
               f"Duplicate module ID: {mid}",
               module_id=mid)
        if mid is not None:
            seen_ids.add(mid)

    # MI-003: Module IDs sequential from 2
    expected_ids = list(range(2, 2 + len(modules)))
    actual_ids = [m.get("id") for m in modules]
    _check("MI-003", actual_ids == expected_ids,
           f"Module IDs must be sequential from 2, expected {expected_ids}, got {actual_ids}")

    # MI-004: Trigger ID is 1 (already checked in SC-005, but separate rule)
    _check("MI-004", trigger_id == 1,
           f"Trigger ID must be 1, got: {trigger_id}")

    # MI-005: Every module value exists in registry
    trigger_module = trigger.get("module", "")
    _check("MI-005", trigger_module in modules_dict,
           f"Trigger module '{trigger_module}' not found in module registry",
           module_id=trigger_id,
           context={"module": trigger_module})

    for mod in modules:
        mid = mod.get("id", "?")
        mod_name = mod.get("module", "")
        _check("MI-005", mod_name in modules_dict,
               f"Module '{mod_name}' not found in module registry",
               module_id=mid,
               context={"module": mod_name})

    # MI-006: module_type matches registry category
    for mod in modules:
        mid = mod.get("id", "?")
        mod_name = mod.get("module", "")
        mod_type = mod.get("module_type", "")
        reg_entry = modules_dict.get(mod_name)
        if reg_entry:
            _check("MI-006", mod_type == reg_entry["category"],
                   f"Module {mid} type '{mod_type}' does not match registry category "
                   f"'{reg_entry['category']}' for '{mod_name}'",
                   module_id=mid)

    # MI-007: Module version matches registry (warning)
    for mod in modules:
        mid = mod.get("id", "?")
        mod_name = mod.get("module", "")
        mod_ver = mod.get("version", 1)
        reg_entry = modules_dict.get(mod_name)
        if reg_entry:
            _check("MI-007", mod_ver == reg_entry["version"],
                   f"Module {mid} version {mod_ver} does not match registry version "
                   f"{reg_entry['version']} for '{mod_name}'",
                   warn=True, module_id=mid)

    # MI-008: Modules requiring credentials have non-null placeholder
    for mod in modules:
        mid = mod.get("id", "?")
        mod_name = mod.get("module", "")
        reg_entry = modules_dict.get(mod_name)
        if reg_entry and reg_entry.get("requires_credential"):
            _check("MI-008", mod.get("credential_placeholder") is not None,
                   f"Module {mid} ('{mod_name}') requires credential but credential_placeholder is null",
                   module_id=mid)

    # MI-009: Modules not requiring credentials have null placeholder (warning)
    for mod in modules:
        mid = mod.get("id", "?")
        mod_name = mod.get("module", "")
        reg_entry = modules_dict.get(mod_name)
        if reg_entry and not reg_entry.get("requires_credential"):
            _check("MI-009", mod.get("credential_placeholder") is None,
                   f"Module {mid} ('{mod_name}') does not require credential but has placeholder set",
                   warn=True, module_id=mid)

    # MI-010: All required parameters per registry are present
    for mod in modules:
        mid = mod.get("id", "?")
        mod_name = mod.get("module", "")
        reg_entry = modules_dict.get(mod_name)
        if reg_entry:
            params = mod.get("parameters", {})
            mapper = mod.get("mapper", {})
            for rp in reg_entry.get("required_parameters", []):
                _check("MI-010", rp in params,
                       f"Module {mid} missing required parameter: '{rp}'",
                       module_id=mid, context={"parameter": rp})
            for rm in reg_entry.get("required_mapper_fields", []):
                _check("MI-010", rm in mapper,
                       f"Module {mid} missing required mapper field: '{rm}'",
                       module_id=mid, context={"mapper_field": rm})

    # ===== CI — Connection Integrity =====

    module_ids_set = set(m.get("id") for m in modules if m.get("id") is not None)

    # Build graph result for graph-based checks
    graph = graph_integrity_check(
        trigger_id=trigger_id,
        module_ids=list(module_ids_set),
        connections=connections
    )

    # CI-001: Every from is "trigger" or valid module ID
    for i, conn in enumerate(connections):
        f = conn.get("from")
        valid = (f == "trigger") or (isinstance(f, int) and f in all_module_ids)
        _check("CI-001", valid,
               f"Connection {i} 'from' value '{f}' is not 'trigger' or a valid module ID",
               context={"connection_index": i})

    # CI-002: Every to is a valid module ID
    for i, conn in enumerate(connections):
        t = conn.get("to")
        _check("CI-002", isinstance(t, int) and t in module_ids_set,
               f"Connection {i} 'to' value '{t}' is not a valid module ID",
               context={"connection_index": i})

    # CI-003: No orphan modules
    for orphan in graph["orphan_nodes"]:
        if orphan != trigger_id:
            _check("CI-003", False,
                   f"Module {orphan} is not reachable from trigger (orphan)",
                   module_id=orphan)
    if not graph["orphan_nodes"]:
        checks_run += 1
        checks_passed += 1

    # CI-004: Trigger has at least one outgoing connection
    trigger_out = [c for c in connections if c.get("from") == "trigger" or c.get("from") == trigger_id]
    _check("CI-004", len(trigger_out) > 0 or len(modules) == 0,
           "Trigger has no outgoing connections")

    # CI-005: Flow is DAG (no cycles)
    _check("CI-005", graph["is_dag"],
           f"Flow graph contains cycles involving nodes: {graph['cycle_nodes']}")

    # CI-006: No duplicate connections (warning)
    conn_pairs = []
    for conn in connections:
        pair = (str(conn.get("from")), conn.get("to"))
        _check("CI-006", pair not in conn_pairs,
               f"Duplicate connection from {pair[0]} to {pair[1]}",
               warn=True)
        conn_pairs.append(pair)

    # CI-007: Routers have >= 2 outgoing connections
    router_ids = set()
    for mod in modules:
        if mod.get("module_type") == "flow_control":
            router_ids.add(mod.get("id"))
    for rid in router_ids:
        out_count = graph["out_degree"].get(rid, 0)
        _check("CI-007", out_count >= 2,
               f"Router module {rid} has {out_count} outgoing connections (minimum 2 required)",
               module_id=rid)

    # CI-008: No self-loops
    for sl in graph["self_loops"]:
        _check("CI-008", False,
               f"Self-loop detected: module {sl.get('from')} connects to itself",
               module_id=sl.get("from"))
    if not graph["self_loops"]:
        checks_run += 1
        checks_passed += 1

    # CI-009: Terminal modules are valid terminal types (warning)
    valid_terminal_types = {"action", "search", "responder", "aggregator"}
    for tid in graph["terminal_nodes"]:
        if tid == trigger_id:
            continue
        mod = next((m for m in modules if m.get("id") == tid), None)
        if mod:
            _check("CI-009", mod.get("module_type") in valid_terminal_types,
                   f"Terminal module {tid} has type '{mod.get('module_type')}' "
                   f"which is not a typical terminal type",
                   warn=True, module_id=tid)

    # CI-010: WebhookResponse only in webhook-triggered scenarios
    for mod in modules:
        if mod.get("module") == "gateway:WebhookResponse":
            _check("CI-010", trigger.get("type") == "webhook",
                   f"Module {mod.get('id')} (WebhookResponse) is only valid with webhook triggers",
                   module_id=mod.get("id"))

    # ===== RR — Router Rules =====

    # RR-001: (same as CI-007, already checked above — skip duplicate)
    # We count it as a separate rule though
    for rid in router_ids:
        out_count = graph["out_degree"].get(rid, 0)
        _check("RR-001", out_count >= 2,
               f"Router {rid} must have at least 2 outgoing connections, has {out_count}",
               module_id=rid)

    # RR-002: At most one unfiltered fallback route per router (warning)
    for rid in router_ids:
        outgoing = [c for c in connections if (c.get("from") == rid or (c.get("from") == "trigger" and rid == trigger_id))]
        unfiltered = [c for c in outgoing if c.get("filter") is None]
        _check("RR-002", len(unfiltered) <= 1,
               f"Router {rid} has {len(unfiltered)} unfiltered routes (max 1 fallback recommended)",
               warn=True, module_id=rid)

    # RR-003: Routers have no data in mapper/parameters (warning)
    for mod in modules:
        if mod.get("id") in router_ids:
            params = mod.get("parameters", {})
            mapper = mod.get("mapper", {})
            _check("RR-003",
                   len(params) == 0 and len(mapper) == 0,
                   f"Router {mod.get('id')} should not have parameters or mapper data",
                   warn=True, module_id=mod.get("id"))

    # ===== CR — Credential Rules =====

    # CR-001: All credential placeholders match pattern
    all_cred_placeholders = []
    trig_cred = trigger.get("credential_placeholder")
    if trig_cred is not None:
        all_cred_placeholders.append((trigger_id, trig_cred))
    for mod in modules:
        mc = mod.get("credential_placeholder")
        if mc is not None:
            all_cred_placeholders.append((mod.get("id"), mc))

    for mid, cred in all_cred_placeholders:
        _check("CR-001", CREDENTIAL_PATTERN.match(cred),
               f"Module {mid} credential placeholder '{cred}' does not match pattern __APP_NAME_CONNECTION__",
               module_id=mid)

    # CR-002: No real credentials anywhere in spec
    _scan_for_real_credentials(spec, errors, checks_run_ref=[0], checks_passed_ref=[0])
    # We handle this as a batch check
    checks_run += 1
    cred_errors = [e for e in errors if e["rule_id"] == "CR-002"]
    if not cred_errors:
        checks_passed += 1

    # CR-003: Same app uses same placeholder throughout (warning)
    app_creds = {}
    for mod in modules:
        app = mod.get("app")
        cred = mod.get("credential_placeholder")
        if cred is not None:
            if app in app_creds and app_creds[app] != cred:
                _warn("CR-003",
                      f"App '{app}' uses inconsistent credential placeholders: "
                      f"'{app_creds[app]}' and '{cred}'",
                      module_id=mod.get("id"))
            app_creds[app] = cred
    checks_run += 1
    cr003_warns = [w for w in warnings if w["rule_id"] == "CR-003"]
    if not cr003_warns:
        checks_passed += 1

    # ===== DM — Data Mapping Rules =====

    dm_result = validate_data_mappings(spec, graph)
    for dm_err in dm_result["errors"]:
        errors.append(dm_err)
        checks_run += 1
    for dm_warn in dm_result.get("warnings", []):
        warnings.append(dm_warn)
        checks_run += 1

    # DM-004: Filter conditions follow same reference rules (handled by validate_data_mappings)
    if dm_result["references_checked"] > 0:
        checks_run += 1
        if dm_result["valid"]:
            checks_passed += 1

    # ===== SS — Structural Soundness =====

    # SS-001: At least one module (warning)
    _check("SS-001", len(modules) > 0,
           "Spec contains no modules", warn=True)

    # SS-002: At least one connection (warning)
    _check("SS-002", len(connections) > 0,
           "Spec contains no connections", warn=True)

    # SS-003: Aggregator modules have valid feeder parameter
    for mod in modules:
        if mod.get("module_type") in ("aggregator",):
            feeder = mod.get("parameters", {}).get("feeder")
            if feeder is not None:
                _check("SS-003", isinstance(feeder, int) and feeder in all_module_ids,
                       f"Aggregator module {mod.get('id')} has invalid feeder: {feeder}",
                       module_id=mod.get("id"))
            # feeder might be required — MI-010 already catches missing params

    # SS-004: Iterators have 1 incoming and >= 1 outgoing (warning)
    for mod in modules:
        if mod.get("module_type") == "iterator":
            mid = mod.get("id")
            _check("SS-004",
                   graph["in_degree"].get(mid, 0) >= 1 and graph["out_degree"].get(mid, 0) >= 1,
                   f"Iterator module {mid} should have >= 1 incoming and >= 1 outgoing connections",
                   warn=True, module_id=mid)

    # SS-005: At most one WebhookResponse module
    resp_count = sum(1 for m in modules if m.get("module") == "gateway:WebhookResponse")
    _check("SS-005", resp_count <= 1,
           f"Spec has {resp_count} WebhookResponse modules (max 1 allowed)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checks_run": checks_run,
        "checks_passed": checks_passed,
        "checks_failed": checks_run - checks_passed,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    }


def _scan_for_real_credentials(obj, errors, checks_run_ref, checks_passed_ref, path="root"):
    """Recursively scan for suspicious credential-like values."""
    if isinstance(obj, str):
        for pattern in SUSPICIOUS_CRED_PATTERNS:
            if pattern.search(obj):
                errors.append({
                    "rule_id": "CR-002",
                    "severity": "error",
                    "message": f"Suspicious credential-like value found at {path}",
                    "module_id": None,
                    "context": {"path": path, "pattern": pattern.pattern}
                })
                return
    elif isinstance(obj, dict):
        for key, val in obj.items():
            _scan_for_real_credentials(val, errors, checks_run_ref, checks_passed_ref, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _scan_for_real_credentials(item, errors, checks_run_ref, checks_passed_ref, f"{path}[{i}]")


# --- Self-check ---
if __name__ == "__main__":
    import json

    print("=== Canonical Spec Validator Self-Check ===\n")

    from tools.module_registry_loader import load_module_registry
    registry = load_module_registry()

    # Valid linear spec: webhook -> parse JSON -> post to Slack
    valid_spec = {
        "spec_version": "1.0.0",
        "scenario": {
            "name": "Test Webhook to Slack",
            "description": "Receives webhook, parses JSON, posts to Slack.",
            "slug": "test-webhook-to-slack"
        },
        "trigger": {
            "id": 1,
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive Webhook",
            "version": 1,
            "parameters": {"hook": "__WEBHOOK_ID__"},
            "credential_placeholder": None,
            "webhook": {
                "name": "Test Webhook",
                "data_structure": {"name": {"type": "string"}}
            }
        },
        "modules": [
            {
                "id": 2,
                "label": "Parse JSON",
                "app": "json",
                "module": "json:ParseJSON",
                "module_type": "transformer",
                "version": 1,
                "parameters": {},
                "mapper": {"json": "{{1.body}}"},
                "credential_placeholder": None,
                "onerror": None,
                "resume_value": None
            },
            {
                "id": 3,
                "label": "Post to Slack",
                "app": "slack",
                "module": "slack:PostMessage",
                "module_type": "action",
                "version": 1,
                "parameters": {},
                "mapper": {
                    "channel": "#general",
                    "text": "Hello {{2.name}}"
                },
                "credential_placeholder": "__SLACK_CONNECTION__",
                "onerror": "ignore",
                "resume_value": None
            }
        ],
        "connections": [
            {"from": "trigger", "to": 2, "filter": None, "label": None},
            {"from": 2, "to": 3, "filter": None, "label": None}
        ],
        "error_handling": {
            "default_strategy": "ignore",
            "max_errors": 3,
            "module_overrides": []
        },
        "metadata": {
            "created_at": "2026-02-16T12:00:00Z",
            "updated_at": None,
            "original_request": "Create a webhook that posts to Slack.",
            "agent_notes": [],
            "build_version": None
        }
    }

    print("Test 1: Valid linear spec")
    report = validate_canonical_spec(valid_spec, registry)
    print(f"  Valid: {report['valid']}")
    print(f"  Checks: {report['checks_run']} run, {report['checks_passed']} passed, {report['checks_failed']} failed")
    if report["errors"]:
        for e in report["errors"]:
            print(f"  ERROR [{e['rule_id']}]: {e['message']}")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  WARN  [{w['rule_id']}]: {w['message']}")
    assert report["valid"], f"Valid spec should pass, but got {len(report['errors'])} errors"
    print("  [OK] Valid spec passed all checks")

    # Test 2: Invalid spec with multiple issues
    print("\nTest 2: Invalid spec (missing fields, bad module)")
    invalid_spec = {
        "spec_version": "bad",
        "scenario": {"name": "Test", "slug": "INVALID SLUG"},
        "trigger": {"id": 2, "type": "invalid"},
        "modules": [
            {"id": 5, "label": "Bad", "app": "fake", "module": "fake:Nothing", "module_type": "action"}
        ],
        "connections": [],
        "error_handling": {"default_strategy": "explode"},
        "metadata": {"created_at": "not-a-date"}
    }
    report2 = validate_canonical_spec(invalid_spec, registry)
    print(f"  Valid: {report2['valid']}")
    print(f"  Errors: {len(report2['errors'])}")
    assert not report2["valid"], "Invalid spec should fail"
    error_rules = {e["rule_id"] for e in report2["errors"]}
    assert "SC-001" in error_rules, "Should catch bad spec_version"
    assert "SC-003" in error_rules, "Should catch bad slug"
    assert "SC-005" in error_rules, "Should catch trigger.id != 1"
    assert "MI-005" in error_rules, "Should catch unknown module"
    print(f"  [OK] Caught {len(report2['errors'])} errors across rules: {sorted(error_rules)}")

    # Test 3: Router spec
    print("\nTest 3: Valid router spec")
    router_spec = {
        "spec_version": "1.0.0",
        "scenario": {
            "name": "Router Test",
            "description": "Routes by priority.",
            "slug": "router-test"
        },
        "trigger": {
            "id": 1,
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive Event",
            "version": 1,
            "parameters": {"hook": "__WEBHOOK_ID__"},
            "credential_placeholder": None
        },
        "modules": [
            {
                "id": 2,
                "label": "Route",
                "app": "builtin",
                "module": "builtin:BasicRouter",
                "module_type": "flow_control",
                "version": 1,
                "parameters": {},
                "mapper": {},
                "credential_placeholder": None,
                "onerror": None,
                "resume_value": None
            },
            {
                "id": 3,
                "label": "Slack Alert",
                "app": "slack",
                "module": "slack:PostMessage",
                "module_type": "action",
                "version": 1,
                "parameters": {},
                "mapper": {"channel": "#urgent", "text": "Alert: {{1.title}}"},
                "credential_placeholder": "__SLACK_CONNECTION__",
                "onerror": None,
                "resume_value": None
            },
            {
                "id": 4,
                "label": "Log to Sheets",
                "app": "google-sheets",
                "module": "google-sheets:addRow",
                "module_type": "action",
                "version": 2,
                "parameters": {"spreadsheetId": "__SPREADSHEET_ID__", "sheetId": "__SHEET_ID__"},
                "mapper": {"values": ["{{1.title}}", "{{1.priority}}"]},
                "credential_placeholder": "__GOOGLE_SHEETS_CONNECTION__",
                "onerror": None,
                "resume_value": None
            }
        ],
        "connections": [
            {"from": "trigger", "to": 2, "filter": None, "label": None},
            {
                "from": 2, "to": 3,
                "filter": {
                    "name": "High Priority",
                    "conditions": [[{"a": "{{1.priority}}", "b": "high", "o": "text:equal"}]]
                },
                "label": "High Priority"
            },
            {"from": 2, "to": 4, "filter": None, "label": "All Items"}
        ],
        "error_handling": {
            "default_strategy": "ignore",
            "max_errors": 3,
            "module_overrides": []
        },
        "metadata": {
            "created_at": "2026-02-16T12:00:00Z",
            "updated_at": None,
            "original_request": "Route high priority to Slack, log all to Sheets.",
            "agent_notes": [],
            "build_version": None
        }
    }
    report3 = validate_canonical_spec(router_spec, registry)
    print(f"  Valid: {report3['valid']}")
    print(f"  Checks: {report3['checks_run']} run, {report3['checks_passed']} passed")
    if report3["errors"]:
        for e in report3["errors"]:
            print(f"  ERROR [{e['rule_id']}]: {e['message']}")
    if report3["warnings"]:
        for w in report3["warnings"]:
            print(f"  WARN  [{w['rule_id']}]: {w['message']}")
    assert report3["valid"], f"Router spec should pass, got {len(report3['errors'])} errors"
    print("  [OK] Router spec passed all checks")

    print(f"\n=== All validator self-checks passed ===")
