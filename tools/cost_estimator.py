"""
Cost Estimator

Produces a config-driven cost estimate based on integrations, module types,
and Make.com operations pricing.

Input:  spec (dict) — canonical spec
Output: dict — cost estimate with per-integration costs, Make ops notes, total

Cost table (config-driven, adjustable):
  Make.com operations:
    - Trigger execution: 1 operation
    - Each module execution: 1 operation
    - Router: 1 operation (regardless of route count)
    - Iterator: 1 operation per iteration

  Integration setup costs (one-time):
    - Slack: $0 (free tier available)
    - Google Sheets: $0 (free tier available)
    - HTTP (generic): $0 (no auth required)
    - JSON/Util: $0 (built-in, no setup)

  Monthly operational estimates based on execution frequency:
    - Make.com Free: 1,000 ops/month
    - Make.com Core: 10,000 ops/month ($9/mo)
    - Make.com Pro: 150,000 ops/month ($16/mo)

Deterministic. No network calls. No AI reasoning.
"""

# Operations per module type per execution
OPS_PER_MODULE = {
    "trigger": 1,
    "action": 1,
    "search": 1,
    "transformer": 1,
    "flow_control": 1,
    "iterator": 1,  # Note: multiplied by avg iteration count in practice
    "aggregator": 1,
    "responder": 1
}

# One-time integration setup costs (USD)
INTEGRATION_SETUP_COST = {
    "slack": 0.0,
    "google-sheets": 0.0,
    "http": 0.0,
    "json": 0.0,
    "util": 0.0,
    "builtin": 0.0,
    "gateway": 0.0
}

# Make.com plan thresholds
MAKE_PLANS = [
    {"name": "Free", "ops_per_month": 1000, "monthly_cost": 0.0},
    {"name": "Core", "ops_per_month": 10000, "monthly_cost": 9.0},
    {"name": "Pro", "ops_per_month": 150000, "monthly_cost": 16.0}
]

# Default execution frequency assumptions
DEFAULT_EXECUTIONS_PER_MONTH = 100


def estimate_cost(spec, executions_per_month=None):
    """Estimate costs for a Make.com scenario.

    Args:
        spec: Canonical spec dict.
        executions_per_month: Expected monthly executions. Default 100.

    Returns:
        dict with:
            - ops_per_execution: int (operations consumed per single run)
            - executions_per_month: int
            - ops_per_month: int (total monthly operations)
            - integration_setup: dict (app → setup cost)
            - total_setup_cost: float
            - recommended_plan: dict (name, ops_per_month, monthly_cost)
            - monthly_operational_cost: float
            - annual_estimate: float
            - notes: list of strings
    """
    if executions_per_month is None:
        executions_per_month = DEFAULT_EXECUTIONS_PER_MONTH

    modules = spec.get("modules", [])
    trigger = spec.get("trigger", {})

    notes = []

    # Calculate operations per execution
    ops = OPS_PER_MODULE.get("trigger", 1)  # Trigger always costs 1 op
    notes.append(f"Trigger ({trigger.get('module', 'unknown')}): 1 op")

    for mod in modules:
        mod_type = mod.get("module_type", "action")
        mod_ops = OPS_PER_MODULE.get(mod_type, 1)
        ops += mod_ops

    if modules:
        notes.append(f"{len(modules)} module(s): {len(modules)} op(s)")

    # Iterator note
    iterators = [m for m in modules if m.get("module_type") == "iterator"]
    if iterators:
        notes.append(
            f"Note: {len(iterators)} iterator(s) present. "
            f"Actual ops multiply by items iterated per execution."
        )

    ops_per_month = ops * executions_per_month

    # Integration setup costs
    apps_used = set()
    trigger_app = trigger.get("module", "").split(":")[0] if ":" in trigger.get("module", "") else "unknown"
    apps_used.add(trigger_app)
    for mod in modules:
        app = mod.get("app", "unknown")
        apps_used.add(app)

    integration_setup = {}
    for app in sorted(apps_used):
        cost = INTEGRATION_SETUP_COST.get(app, 0.0)
        integration_setup[app] = cost

    total_setup = sum(integration_setup.values())

    # Recommend Make.com plan
    recommended = MAKE_PLANS[0]  # Default to free
    for plan in MAKE_PLANS:
        if plan["ops_per_month"] >= ops_per_month:
            recommended = plan
            break
    else:
        # Exceeds all plans
        recommended = MAKE_PLANS[-1]
        notes.append(
            f"Warning: {ops_per_month} ops/month exceeds Pro plan ({MAKE_PLANS[-1]['ops_per_month']}). "
            f"Consider Teams or Enterprise plan."
        )

    monthly_cost = recommended["monthly_cost"]
    annual = round(monthly_cost * 12 + total_setup, 2)

    notes.append(f"Recommended plan: Make.com {recommended['name']} ({recommended['ops_per_month']:,} ops/month)")
    notes.append(f"Estimated {ops_per_month:,} ops/month at {executions_per_month:,} executions/month")

    # Credential-requiring apps note
    cred_apps = set()
    if trigger.get("credential_placeholder"):
        cred_apps.add(trigger_app)
    for mod in modules:
        if mod.get("credential_placeholder"):
            cred_apps.add(mod.get("app", "unknown"))
    if cred_apps:
        notes.append(f"Apps requiring OAuth/API key setup: {', '.join(sorted(cred_apps))}")

    return {
        "ops_per_execution": ops,
        "executions_per_month": executions_per_month,
        "ops_per_month": ops_per_month,
        "integration_setup": integration_setup,
        "total_setup_cost": total_setup,
        "recommended_plan": recommended,
        "monthly_operational_cost": monthly_cost,
        "annual_estimate": annual,
        "notes": notes
    }


# --- Self-check ---
if __name__ == "__main__":
    print("=== Cost Estimator Self-Check ===\n")

    # Test 1: Simple linear (trigger + 2 modules)
    print("Test 1: Simple linear (3 ops/execution)")
    spec1 = {
        "trigger": {"id": 1, "module": "gateway:CustomWebHook", "credential_placeholder": None},
        "modules": [
            {"id": 2, "module": "json:ParseJSON", "module_type": "transformer",
             "app": "json", "credential_placeholder": None},
            {"id": 3, "module": "slack:PostMessage", "module_type": "action",
             "app": "slack", "credential_placeholder": "__SLACK_CONNECTION__"}
        ]
    }
    c1 = estimate_cost(spec1, executions_per_month=100)
    assert c1["ops_per_execution"] == 3
    assert c1["ops_per_month"] == 300
    assert c1["recommended_plan"]["name"] == "Free"
    assert c1["monthly_operational_cost"] == 0.0
    print(f"  Ops/exec: {c1['ops_per_execution']}, Ops/month: {c1['ops_per_month']}")
    print(f"  Plan: {c1['recommended_plan']['name']}, Cost: ${c1['monthly_operational_cost']}/mo")
    print("  [OK]")

    # Test 2: High volume pushes to paid plan
    print("\nTest 2: High volume (500 exec/month)")
    c2 = estimate_cost(spec1, executions_per_month=500)
    assert c2["ops_per_month"] == 1500
    assert c2["recommended_plan"]["name"] == "Core"
    print(f"  Ops/month: {c2['ops_per_month']}, Plan: {c2['recommended_plan']['name']}")
    print("  [OK]")

    # Test 3: Router scenario (4 ops)
    print("\nTest 3: Router scenario")
    spec3 = {
        "trigger": {"id": 1, "module": "gateway:CustomWebHook", "credential_placeholder": None},
        "modules": [
            {"id": 2, "module": "builtin:BasicRouter", "module_type": "flow_control",
             "app": "builtin", "credential_placeholder": None},
            {"id": 3, "module": "slack:PostMessage", "module_type": "action",
             "app": "slack", "credential_placeholder": "__SLACK_CONNECTION__"},
            {"id": 4, "module": "google-sheets:addRow", "module_type": "action",
             "app": "google-sheets", "credential_placeholder": "__GOOGLE_SHEETS_CONNECTION__"}
        ]
    }
    c3 = estimate_cost(spec3, executions_per_month=100)
    assert c3["ops_per_execution"] == 4
    assert "slack" in c3["integration_setup"]
    assert "google-sheets" in c3["integration_setup"]
    print(f"  Ops/exec: {c3['ops_per_execution']}, Apps: {list(c3['integration_setup'].keys())}")
    print("  [OK]")

    # Test 4: Trigger only
    print("\nTest 4: Trigger only")
    spec4 = {
        "trigger": {"id": 1, "module": "gateway:CustomWebHook", "credential_placeholder": None},
        "modules": []
    }
    c4 = estimate_cost(spec4, executions_per_month=50)
    assert c4["ops_per_execution"] == 1
    assert c4["ops_per_month"] == 50
    print(f"  Ops/exec: {c4['ops_per_execution']}, Monthly: ${c4['monthly_operational_cost']}")
    print("  [OK]")

    # Test 5: Determinism
    print("\nTest 5: Determinism")
    c5a = estimate_cost(spec3, executions_per_month=100)
    c5b = estimate_cost(spec3, executions_per_month=100)
    assert c5a == c5b
    print("  [OK]")

    # Test 6: Iterator note present
    print("\nTest 6: Iterator note")
    spec6 = {
        "trigger": {"id": 1, "module": "gateway:CustomWebHook", "credential_placeholder": None},
        "modules": [
            {"id": 2, "module": "builtin:BasicIterator", "module_type": "iterator",
             "app": "builtin", "credential_placeholder": None}
        ]
    }
    c6 = estimate_cost(spec6)
    has_iter_note = any("iterator" in n.lower() for n in c6["notes"])
    assert has_iter_note, "Should have iterator warning note"
    print(f"  Notes: {[n for n in c6['notes'] if 'iterator' in n.lower()]}")
    print("  [OK]")

    print("\n=== All cost_estimator checks passed ===")
