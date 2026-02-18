"""
Generate Make Export

Converts a validated canonical spec into a Make.com scenario blueprint JSON
that can be imported via Make.com's blueprint import feature.

Input:  spec (dict) — validated canonical spec
Output: dict — Make.com blueprint JSON

Make.com blueprint structure:
  {
    "name": str,
    "flow": [ ...modules... ],
    "metadata": { "scenario": {...}, "designer": {...} }
  }

Key conversions:
  - Canonical explicit connections → Make implicit flow ordering + router routes
  - Canonical connection filters → Make filters on receiving modules
  - Canonical credential_placeholder → Make parameters.__IMTCONN__
  - Canonical onerror string → Make onerror array [{"directive": str}]
  - Deterministic x,y designer positions

Deterministic. No network calls. No Make API calls. Pure JSON generation.
"""

from collections import defaultdict

from tools.graph_integrity_check import graph_integrity_check


# Designer layout constants (pixels in Make.com editor)
X_SPACING = 300
Y_SPACING = 200
X_START = 0
Y_START = 0


def generate_make_export(spec):
    """Convert a canonical spec into a Make.com blueprint JSON.

    Args:
        spec: Validated canonical spec dict.

    Returns:
        dict — Make.com blueprint ready for import.
    """
    trigger = spec.get("trigger", {})
    modules = spec.get("modules", [])
    connections = spec.get("connections", [])
    error_handling = spec.get("error_handling", {})

    # Index modules by ID for O(1) lookup
    modules_by_id = {m["id"]: m for m in modules}

    # Build adjacency and filter maps from connections
    adj = defaultdict(list)       # module_id → [(target_id, connection)]
    incoming_filter = {}          # target_id → filter dict (from connection)

    for conn in connections:
        from_val = conn["from"]
        from_id = trigger["id"] if from_val == "trigger" else from_val
        to_id = conn["to"]
        adj[from_id].append((to_id, conn))
        if conn.get("filter"):
            incoming_filter[to_id] = conn["filter"]

    # Identify router module IDs
    router_ids = {m["id"] for m in modules if m.get("module_type") == "flow_control"}

    # Build the flow tree starting from trigger
    visited = set()
    x_counter = [X_START]  # mutable counter for x position tracking

    trigger_make = _make_trigger_module(trigger, x_counter)
    visited.add(trigger["id"])

    # Get children of trigger
    trigger_children = adj.get(trigger["id"], [])

    if not trigger_children:
        # Trigger-only scenario
        flow = [trigger_make]
    else:
        # Build the main flow chain starting from trigger
        flow = [trigger_make]
        _build_flow_chain(
            trigger["id"], adj, modules_by_id, router_ids,
            incoming_filter, visited, x_counter, flow
        )

    # Build top-level metadata
    metadata = _make_scenario_metadata(error_handling, spec.get("scenario", {}))

    return {
        "name": spec.get("scenario", {}).get("name", "Untitled"),
        "flow": flow,
        "metadata": metadata
    }


def _build_flow_chain(current_id, adj, modules_by_id, router_ids,
                       incoming_filter, visited, x_counter, flow):
    """Recursively build the flow chain from a starting module.

    Appends modules to the flow list. Handles routers by creating
    nested routes with sub-flows.

    Args:
        current_id: ID of the current module (already in flow).
        adj: Adjacency map (id → [(target_id, connection)]).
        modules_by_id: Module lookup by ID.
        router_ids: Set of router module IDs.
        incoming_filter: Map of target_id → filter dict.
        visited: Set of visited module IDs.
        x_counter: Mutable list [current_x] for position tracking.
        flow: The flow list to append to (mutated in place).
    """
    children = adj.get(current_id, [])
    if not children:
        return

    # Sort children deterministically by target module ID
    children = sorted(children, key=lambda c: c[0])

    if len(children) == 1:
        # Linear: single next module
        target_id, conn = children[0]
        if target_id in visited:
            return
        visited.add(target_id)

        mod = modules_by_id.get(target_id)
        if mod is None:
            return

        if target_id in router_ids:
            # Next module is a router — build router with routes
            router_make = _make_router_module(
                mod, adj, modules_by_id, router_ids,
                incoming_filter, visited, x_counter
            )
            _apply_incoming_filter(router_make, target_id, incoming_filter)
            flow.append(router_make)
        else:
            # Regular module
            x_counter[0] += X_SPACING
            make_mod = _convert_module(mod, x_counter[0], Y_START)
            _apply_incoming_filter(make_mod, target_id, incoming_filter)
            flow.append(make_mod)

            # Continue chain
            _build_flow_chain(
                target_id, adj, modules_by_id, router_ids,
                incoming_filter, visited, x_counter, flow
            )
    else:
        # Multiple children from a non-router module — shouldn't happen
        # in well-formed specs, but handle gracefully by treating first child
        # as main chain
        for target_id, conn in children:
            if target_id in visited:
                continue
            visited.add(target_id)
            mod = modules_by_id.get(target_id)
            if mod is None:
                continue

            x_counter[0] += X_SPACING
            make_mod = _convert_module(mod, x_counter[0], Y_START)
            _apply_incoming_filter(make_mod, target_id, incoming_filter)
            flow.append(make_mod)
            _build_flow_chain(
                target_id, adj, modules_by_id, router_ids,
                incoming_filter, visited, x_counter, flow
            )
            break


def _make_router_module(router_mod, adj, modules_by_id, router_ids,
                         incoming_filter, visited, x_counter):
    """Build a Make.com router module with nested routes.

    Args:
        router_mod: Canonical spec module dict for the router.
        adj, modules_by_id, router_ids, incoming_filter, visited, x_counter:
            Same as _build_flow_chain.

    Returns:
        dict — Make.com router module with routes array.
    """
    x_counter[0] += X_SPACING
    router_x = x_counter[0]

    router_children = adj.get(router_mod["id"], [])
    # Sort routes deterministically by target module ID
    router_children = sorted(router_children, key=lambda c: c[0])

    routes = []
    for route_idx, (target_id, conn) in enumerate(router_children):
        if target_id in visited:
            continue
        visited.add(target_id)

        route_mod = modules_by_id.get(target_id)
        if route_mod is None:
            continue

        # Y offset for each route branch
        route_y = Y_START + (route_idx * Y_SPACING) - ((len(router_children) - 1) * Y_SPACING // 2)

        route_x_counter = [router_x]
        route_flow = []

        if target_id in router_ids:
            # Nested router
            nested_router = _make_router_module(
                route_mod, adj, modules_by_id, router_ids,
                incoming_filter, visited, route_x_counter
            )
            route_flow.append(nested_router)
        else:
            route_x_counter[0] += X_SPACING
            make_mod = _convert_module(route_mod, route_x_counter[0], route_y)
            route_flow.append(make_mod)

            # Continue the chain within this route
            _build_flow_chain(
                target_id, adj, modules_by_id, router_ids,
                incoming_filter, visited, route_x_counter, route_flow
            )

        route_entry = {"flow": route_flow}

        # Apply filter from the connection to this route's first module
        filt = conn.get("filter")
        if filt:
            if route_flow:
                route_flow[0]["filter"] = _convert_filter(filt)

        routes.append(route_entry)

    return {
        "id": router_mod["id"],
        "module": router_mod["module"],
        "version": router_mod.get("version", 1),
        "mapper": None,
        "metadata": {
            "designer": {"x": router_x, "y": Y_START}
        },
        "routes": routes
    }


def _make_trigger_module(trigger, x_counter):
    """Convert canonical trigger to a Make.com flow module.

    Args:
        trigger: Canonical spec trigger dict.
        x_counter: Mutable [x] counter.

    Returns:
        dict — Make.com module for the trigger.
    """
    params = dict(trigger.get("parameters", {}))

    # Credential placeholder → __IMTCONN__
    cred = trigger.get("credential_placeholder")
    if cred:
        params["__IMTCONN__"] = cred

    make_mod = {
        "id": trigger["id"],
        "module": trigger["module"],
        "version": trigger.get("version", 1),
        "parameters": params,
        "mapper": {},
        "metadata": {
            "designer": {"x": x_counter[0], "y": Y_START}
        }
    }

    return make_mod


def _convert_module(mod, x, y):
    """Convert a canonical spec module to a Make.com flow module.

    Args:
        mod: Canonical spec module dict.
        x: X position for designer.
        y: Y position for designer.

    Returns:
        dict — Make.com module.
    """
    params = dict(mod.get("parameters", {}))

    # Credential placeholder → __IMTCONN__
    cred = mod.get("credential_placeholder")
    if cred:
        params["__IMTCONN__"] = cred

    make_mod = {
        "id": mod["id"],
        "module": mod["module"],
        "version": mod.get("version", 1),
        "parameters": params,
        "mapper": mod.get("mapper", {}),
        "metadata": {
            "designer": {"x": x, "y": y}
        }
    }

    # Error handling
    onerror = mod.get("onerror")
    if onerror:
        onerror_entry = {"directive": onerror}
        if onerror == "resume" and mod.get("resume_value"):
            onerror_entry["resume"] = mod["resume_value"]
        make_mod["onerror"] = [onerror_entry]

    return make_mod


def _convert_filter(filt):
    """Convert a canonical spec filter to Make.com filter format.

    The format is identical (name + conditions 2D array), so this is
    mostly a pass-through with structure enforcement.

    Args:
        filt: Canonical spec filter dict.

    Returns:
        dict — Make.com filter.
    """
    if filt is None:
        return None

    return {
        "name": filt.get("name", "Filter"),
        "conditions": filt.get("conditions", [])
    }


def _apply_incoming_filter(make_mod, target_id, incoming_filter):
    """Apply an incoming connection's filter to a Make module.

    In Make.com, filters sit on the receiving module.

    Args:
        make_mod: The Make.com module dict to potentially add filter to.
        target_id: The module ID receiving the connection.
        incoming_filter: Map of target_id → filter dict.
    """
    filt = incoming_filter.get(target_id)
    if filt:
        make_mod["filter"] = _convert_filter(filt)


def _make_scenario_metadata(error_handling, scenario):
    """Build the top-level Make.com metadata object.

    Args:
        error_handling: Canonical spec error_handling dict.
        scenario: Canonical spec scenario dict.

    Returns:
        dict — Make.com metadata.
    """
    max_errors = error_handling.get("max_errors", 3)

    return {
        "version": 1,
        "scenario": {
            "roundtrips": 1,
            "maxErrors": max_errors,
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


# --- Self-check ---
if __name__ == "__main__":
    import json

    print("=== Generate Make Export Self-Check ===\n")

    # Test 1: Linear spec (webhook → parse → slack)
    print("Test 1: Linear flow (webhook → parse JSON → Slack)")
    linear_spec = {
        "spec_version": "1.0.0",
        "scenario": {
            "name": "Form to Slack",
            "description": "Receives form, posts to Slack.",
            "slug": "form-to-slack"
        },
        "trigger": {
            "id": 1,
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive Form",
            "version": 1,
            "parameters": {"hook": "__WEBHOOK_ID__"},
            "credential_placeholder": None
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
                "mapper": {"channel": "#general", "text": "Hello {{2.name}}"},
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
            "original_request": "Post form data to Slack."
        }
    }

    blueprint = generate_make_export(linear_spec)

    # Verify top-level structure
    assert "name" in blueprint
    assert "flow" in blueprint
    assert "metadata" in blueprint
    assert blueprint["name"] == "Form to Slack"

    # Verify flow has 3 modules (trigger + 2 modules)
    assert len(blueprint["flow"]) == 3, f"Expected 3 flow items, got {len(blueprint['flow'])}"

    # Verify trigger is first
    assert blueprint["flow"][0]["id"] == 1
    assert blueprint["flow"][0]["module"] == "gateway:CustomWebHook"

    # Verify module ordering
    assert blueprint["flow"][1]["id"] == 2
    assert blueprint["flow"][1]["module"] == "json:ParseJSON"
    assert blueprint["flow"][1]["mapper"] == {"json": "{{1.body}}"}

    assert blueprint["flow"][2]["id"] == 3
    assert blueprint["flow"][2]["module"] == "slack:PostMessage"
    assert blueprint["flow"][2]["mapper"] == {"channel": "#general", "text": "Hello {{2.name}}"}

    # Verify credential placeholder → __IMTCONN__
    assert "__IMTCONN__" in blueprint["flow"][2]["parameters"]
    assert blueprint["flow"][2]["parameters"]["__IMTCONN__"] == "__SLACK_CONNECTION__"

    # Verify onerror conversion
    assert blueprint["flow"][2].get("onerror") == [{"directive": "ignore"}]

    # Verify metadata
    assert blueprint["metadata"]["scenario"]["maxErrors"] == 3
    assert blueprint["metadata"]["designer"]["orphans"] == []

    # Verify designer positions are set
    for i, mod in enumerate(blueprint["flow"]):
        assert "metadata" in mod
        assert "designer" in mod["metadata"]
        assert "x" in mod["metadata"]["designer"]
        assert "y" in mod["metadata"]["designer"]

    print("  [OK] Linear flow structure correct")
    print("  [OK] Credential placeholder → __IMTCONN__")
    print("  [OK] onerror string → onerror array")
    print("  [OK] Designer positions assigned")

    # Test 2: Router spec (webhook → router → [slack, sheets])
    print("\nTest 2: Router flow (webhook → router → [slack, sheets])")
    router_spec = {
        "spec_version": "1.0.0",
        "scenario": {
            "name": "Priority Router",
            "description": "Routes by priority.",
            "slug": "priority-router"
        },
        "trigger": {
            "id": 1,
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive",
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
                "mapper": {"channel": "#urgent", "text": "ALERT: {{1.title}}"},
                "credential_placeholder": "__SLACK_CONNECTION__",
                "onerror": "break",
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
                "mapper": {"values": ["{{1.priority}}", "{{1.title}}"]},
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
            "max_errors": 5,
            "module_overrides": []
        },
        "metadata": {
            "created_at": "2026-02-16T12:00:00Z",
            "original_request": "Route by priority."
        }
    }

    bp2 = generate_make_export(router_spec)

    # Flow should be [trigger, router_with_routes]
    assert len(bp2["flow"]) == 2, f"Expected 2 top-level flow items, got {len(bp2['flow'])}"
    assert bp2["flow"][0]["id"] == 1  # trigger
    assert bp2["flow"][1]["id"] == 2  # router

    # Router should have routes
    router_mod = bp2["flow"][1]
    assert "routes" in router_mod, "Router module should have 'routes' array"
    assert len(router_mod["routes"]) == 2, f"Expected 2 routes, got {len(router_mod['routes'])}"

    # Route 1: Slack (module 3) — should have filter
    route1 = router_mod["routes"][0]
    assert len(route1["flow"]) == 1
    assert route1["flow"][0]["id"] == 3
    assert route1["flow"][0]["module"] == "slack:PostMessage"
    assert "filter" in route1["flow"][0], "First route should have filter on first module"
    assert route1["flow"][0]["filter"]["name"] == "High Priority"

    # Route 2: Sheets (module 4) — no filter
    route2 = router_mod["routes"][1]
    assert len(route2["flow"]) == 1
    assert route2["flow"][0]["id"] == 4
    assert route2["flow"][0]["module"] == "google-sheets:addRow"

    # Verify credential on route modules
    assert route1["flow"][0]["parameters"]["__IMTCONN__"] == "__SLACK_CONNECTION__"
    assert route2["flow"][0]["parameters"]["__IMTCONN__"] == "__GOOGLE_SHEETS_CONNECTION__"

    # Verify max_errors propagated
    assert bp2["metadata"]["scenario"]["maxErrors"] == 5

    print("  [OK] Router structure with routes array")
    print("  [OK] Filter placed on first module of filtered route")
    print("  [OK] Credentials on route modules")
    print("  [OK] maxErrors propagated to metadata")

    # Test 3: Determinism
    print("\nTest 3: Determinism")
    bp2b = generate_make_export(router_spec)
    assert json.dumps(bp2, sort_keys=True) == json.dumps(bp2b, sort_keys=True)
    print("  [OK] Identical input → identical output")

    # Test 4: Empty modules (trigger only)
    print("\nTest 4: Trigger-only scenario")
    empty_spec = {
        "spec_version": "1.0.0",
        "scenario": {"name": "Webhook Only", "description": ".", "slug": "webhook-only"},
        "trigger": {
            "id": 1, "type": "webhook", "module": "gateway:CustomWebHook",
            "label": "Receive", "version": 1, "parameters": {},
            "credential_placeholder": None
        },
        "modules": [],
        "connections": [],
        "error_handling": {"default_strategy": "ignore", "max_errors": 3},
        "metadata": {"created_at": "2026-02-16T12:00:00Z", "original_request": "Just a webhook."}
    }
    bp4 = generate_make_export(empty_spec)
    assert len(bp4["flow"]) == 1
    assert bp4["flow"][0]["id"] == 1
    print("  [OK] Trigger-only flow")

    # Test 5: JSON serialization
    print("\nTest 5: JSON serialization")
    json_str = json.dumps(bp2, indent=2)
    roundtrip = json.loads(json_str)
    assert roundtrip == bp2
    print(f"  [OK] Serializes cleanly ({len(json_str)} bytes)")

    # Print the router blueprint for visual inspection
    print("\n--- Router Blueprint (abridged) ---")
    print(f"name: {bp2['name']}")
    print(f"flow: [{bp2['flow'][0]['module']}, {bp2['flow'][1]['module']}]")
    for i, route in enumerate(bp2["flow"][1]["routes"]):
        mods = [m["module"] for m in route["flow"]]
        has_filter = "filter" in route["flow"][0] if route["flow"] else False
        print(f"  route[{i}]: {mods} (filter: {has_filter})")

    print("\n=== All generate_make_export checks passed ===")
