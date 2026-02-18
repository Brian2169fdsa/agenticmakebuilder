"""
Timeline Estimator

Produces a heuristic build/implementation timeline estimate based on
scenario complexity: module count, routers, iterators, external app integrations.

Input:  spec (dict) — canonical spec
Output: dict — timeline estimate with phases, total_hours, complexity_grade

Heuristics (config-driven, not AI reasoning):
  Base: 1 hour (any scenario)
  Per module: +0.5 hours
  Per router: +1.0 hours (branching adds complexity)
  Per iterator/aggregator: +0.75 hours
  Per external app (credential-requiring): +1.5 hours (auth setup, testing)
  Per connection with filter: +0.25 hours

Deterministic. No network calls. No AI reasoning.
"""

# Heuristic weights (hours)
BASE_HOURS = 1.0
PER_MODULE = 0.5
PER_ROUTER = 1.0
PER_ITERATOR_AGGREGATOR = 0.75
PER_EXTERNAL_APP = 1.5
PER_FILTERED_CONNECTION = 0.25

# Phase breakdown ratios
PHASE_RATIOS = {
    "design_and_planning": 0.15,
    "module_configuration": 0.30,
    "connection_and_routing": 0.15,
    "credential_setup": 0.15,
    "testing_and_validation": 0.20,
    "documentation": 0.05
}

COMPLEXITY_THRESHOLDS = {
    "simple": 4.0,
    "moderate": 7.0,
    "complex": 12.0,
    # above 12.0 = "advanced"
}


def estimate_timeline(spec):
    """Estimate implementation timeline from a canonical spec.

    Args:
        spec: Canonical spec dict.

    Returns:
        dict with:
            - total_hours: float
            - complexity_grade: str (simple, moderate, complex, advanced)
            - breakdown: dict of component contributions
            - phases: dict of phase → estimated hours
            - notes: list of strings explaining estimates
    """
    modules = spec.get("modules", [])
    connections = spec.get("connections", [])
    trigger = spec.get("trigger", {})

    breakdown = {}
    notes = []

    # Base
    breakdown["base"] = BASE_HOURS
    notes.append(f"Base setup: {BASE_HOURS}h")

    # Module count
    module_hours = len(modules) * PER_MODULE
    breakdown["modules"] = round(module_hours, 2)
    if modules:
        notes.append(f"{len(modules)} module(s): +{module_hours}h")

    # Routers
    routers = [m for m in modules if m.get("module_type") == "flow_control"]
    router_hours = len(routers) * PER_ROUTER
    breakdown["routers"] = round(router_hours, 2)
    if routers:
        notes.append(f"{len(routers)} router(s): +{router_hours}h (branching complexity)")

    # Iterators + Aggregators
    iter_agg = [m for m in modules if m.get("module_type") in ("iterator", "aggregator")]
    iter_hours = len(iter_agg) * PER_ITERATOR_AGGREGATOR
    breakdown["iterators_aggregators"] = round(iter_hours, 2)
    if iter_agg:
        notes.append(f"{len(iter_agg)} iterator/aggregator(s): +{iter_hours}h")

    # External apps (credential-requiring)
    cred_apps = set()
    if trigger.get("credential_placeholder"):
        cred_apps.add(trigger.get("module", "trigger"))
    for m in modules:
        if m.get("credential_placeholder"):
            cred_apps.add(m.get("app", m.get("module", "unknown")))

    ext_hours = len(cred_apps) * PER_EXTERNAL_APP
    breakdown["external_apps"] = round(ext_hours, 2)
    if cred_apps:
        notes.append(f"{len(cred_apps)} external app(s) requiring auth: +{ext_hours}h")
        for app in sorted(cred_apps):
            notes.append(f"  - {app}: credential setup + testing")

    # Filtered connections
    filtered = [c for c in connections if c.get("filter")]
    filter_hours = len(filtered) * PER_FILTERED_CONNECTION
    breakdown["filtered_connections"] = round(filter_hours, 2)
    if filtered:
        notes.append(f"{len(filtered)} filtered connection(s): +{filter_hours}h")

    # Total
    total = sum(breakdown.values())
    total = round(total, 2)

    # Complexity grade
    if total <= COMPLEXITY_THRESHOLDS["simple"]:
        complexity = "simple"
    elif total <= COMPLEXITY_THRESHOLDS["moderate"]:
        complexity = "moderate"
    elif total <= COMPLEXITY_THRESHOLDS["complex"]:
        complexity = "complex"
    else:
        complexity = "advanced"

    # Phase breakdown
    phases = {}
    for phase, ratio in PHASE_RATIOS.items():
        phases[phase] = round(total * ratio, 2)

    # Skip credential_setup phase if no external apps
    if not cred_apps:
        phases["credential_setup"] = 0.0
        # Redistribute to testing
        phases["testing_and_validation"] = round(
            total * (PHASE_RATIOS["testing_and_validation"] + PHASE_RATIOS["credential_setup"]), 2
        )

    return {
        "total_hours": total,
        "complexity_grade": complexity,
        "breakdown": breakdown,
        "phases": phases,
        "notes": notes
    }


# --- Self-check ---
if __name__ == "__main__":
    print("=== Timeline Estimator Self-Check ===\n")

    # Test 1: Simple linear scenario (webhook → parse → slack)
    print("Test 1: Simple linear (2 modules, 1 external app)")
    spec1 = {
        "trigger": {"id": 1, "module": "gateway:CustomWebHook", "credential_placeholder": None},
        "modules": [
            {"id": 2, "module": "json:ParseJSON", "module_type": "transformer",
             "credential_placeholder": None, "app": "json"},
            {"id": 3, "module": "slack:PostMessage", "module_type": "action",
             "credential_placeholder": "__SLACK_CONNECTION__", "app": "slack"}
        ],
        "connections": [
            {"from": "trigger", "to": 2, "filter": None},
            {"from": 2, "to": 3, "filter": None}
        ]
    }
    t1 = estimate_timeline(spec1)
    assert t1["total_hours"] > 0
    assert t1["complexity_grade"] == "simple"
    assert t1["breakdown"]["external_apps"] == PER_EXTERNAL_APP
    print(f"  Total: {t1['total_hours']}h, Grade: {t1['complexity_grade']}")
    print(f"  Breakdown: {t1['breakdown']}")
    print("  [OK]")

    # Test 2: Router scenario (more complex)
    print("\nTest 2: Router scenario (3 modules, router, 2 external apps, 1 filter)")
    spec2 = {
        "trigger": {"id": 1, "module": "gateway:CustomWebHook", "credential_placeholder": None},
        "modules": [
            {"id": 2, "module": "builtin:BasicRouter", "module_type": "flow_control",
             "credential_placeholder": None, "app": "builtin"},
            {"id": 3, "module": "slack:PostMessage", "module_type": "action",
             "credential_placeholder": "__SLACK_CONNECTION__", "app": "slack"},
            {"id": 4, "module": "google-sheets:addRow", "module_type": "action",
             "credential_placeholder": "__GOOGLE_SHEETS_CONNECTION__", "app": "google-sheets"}
        ],
        "connections": [
            {"from": "trigger", "to": 2, "filter": None},
            {"from": 2, "to": 3, "filter": {"name": "Urgent", "conditions": [[]]}},
            {"from": 2, "to": 4, "filter": None}
        ]
    }
    t2 = estimate_timeline(spec2)
    assert t2["total_hours"] > t1["total_hours"]
    assert t2["complexity_grade"] in ("moderate", "complex")
    assert t2["breakdown"]["routers"] == PER_ROUTER
    assert t2["breakdown"]["external_apps"] == 2 * PER_EXTERNAL_APP
    print(f"  Total: {t2['total_hours']}h, Grade: {t2['complexity_grade']}")
    print("  [OK]")

    # Test 3: Empty scenario (trigger only)
    print("\nTest 3: Trigger only")
    spec3 = {
        "trigger": {"id": 1, "module": "gateway:CustomWebHook", "credential_placeholder": None},
        "modules": [],
        "connections": []
    }
    t3 = estimate_timeline(spec3)
    assert t3["total_hours"] == BASE_HOURS
    assert t3["complexity_grade"] == "simple"
    print(f"  Total: {t3['total_hours']}h, Grade: {t3['complexity_grade']}")
    print("  [OK]")

    # Test 4: Phases sum approximately to total
    print("\nTest 4: Phase breakdown")
    phase_sum = sum(t2["phases"].values())
    assert abs(phase_sum - t2["total_hours"]) < 0.1, f"Phase sum {phase_sum} != total {t2['total_hours']}"
    print(f"  Phases: {t2['phases']}")
    print(f"  Sum: {phase_sum} ≈ {t2['total_hours']}")
    print("  [OK]")

    # Test 5: Determinism
    print("\nTest 5: Determinism")
    t5a = estimate_timeline(spec2)
    t5b = estimate_timeline(spec2)
    assert t5a == t5b
    print("  [OK]")

    print("\n=== All timeline_estimator checks passed ===")
