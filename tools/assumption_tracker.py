"""
Assumption Tracker

Centralizes all assumptions introduced during the pipeline lifecycle:
  - generate_delivery_assessment (module resolution, field inference)
  - normalize_to_canonical_spec (default mapper injection)
  - parameter auto-population (default parameters)
  - self_heal_make_export (repair passes)

The tracker is a plain dict (no classes). Create one at pipeline start,
record events as they occur, then extract formatted outputs for delivery
artifacts.

API:
    create_tracker()                     → tracker dict
    record(tracker, ...)                 → mutates tracker in place
    get_ledger(tracker)                  → structured ledger dict
    compute_confidence_penalty(tracker)  → float (total penalty)
    classify_risk(tracker)               → str (low/moderate/elevated/high)
    format_for_delivery_report(tracker)  → dict (assumptions section)
    format_for_build_log(tracker)        → str (markdown section)
    format_for_delivery_pack(tracker)    → dict (machine-readable)

Deterministic. No network calls. No AI reasoning. No Make/validation logic.
"""

# --- Assumption event types and their per-event confidence penalties ---
TYPE_PENALTIES = {
    "module_resolution":    0.02,   # Auto-resolved module from app + description
    "mapper_default":       0.03,   # Injected default mapper value
    "parameter_default":    0.03,   # Injected default parameter value
    "repair":               0.05,   # Self-heal repair applied
    "field_inference":      0.01,   # Inferred a field value from context
}

# Maximum total penalty from assumptions (caps the sum)
MAX_TOTAL_PENALTY = 0.30

# Risk classification thresholds (by event count)
RISK_THRESHOLDS = {
    "low":      0,    # 0 assumptions
    "moderate": 1,    # 1–3
    "elevated": 4,    # 4–7
    "high":     8,    # 8+
}

# Valid sources
VALID_SOURCES = {
    "delivery_assessment",
    "normalize",
    "default_mapper",
    "default_parameters",
    "self_heal",
    "pipeline",
}


def create_tracker():
    """Create a new assumption tracker.

    Returns:
        dict — empty tracker ready to record events.
    """
    return {
        "events": [],
        "event_count": 0,
    }


def record(tracker, source, assumption_type, module, field, assumed_value, reason):
    """Record a single assumption event.

    Args:
        tracker: Tracker dict from create_tracker().
        source: Where the assumption was made (e.g., "delivery_assessment").
        assumption_type: Event type (e.g., "module_resolution", "mapper_default").
        module: Module identifier or step description.
        field: Which field was assumed.
        assumed_value: The value that was assumed.
        reason: Why this assumption was made.
    """
    event = {
        "id": tracker["event_count"] + 1,
        "source": source,
        "type": assumption_type,
        "module": str(module),
        "field": str(field),
        "assumed_value": assumed_value,
        "reason": str(reason),
        "penalty": TYPE_PENALTIES.get(assumption_type, 0.02),
    }
    tracker["events"].append(event)
    tracker["event_count"] += 1


def get_ledger(tracker):
    """Return a structured assumption ledger.

    Returns:
        dict with:
            - event_count: int
            - events: list of event dicts
            - by_source: dict[source] → count
            - by_type: dict[type] → count
            - total_penalty: float
            - risk_level: str
    """
    by_source = {}
    by_type = {}
    for ev in tracker["events"]:
        by_source[ev["source"]] = by_source.get(ev["source"], 0) + 1
        by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1

    return {
        "event_count": tracker["event_count"],
        "events": list(tracker["events"]),
        "by_source": by_source,
        "by_type": by_type,
        "total_penalty": compute_confidence_penalty(tracker),
        "risk_level": classify_risk(tracker),
    }


def compute_confidence_penalty(tracker):
    """Compute the total confidence penalty from all recorded assumptions.

    Returns:
        float — penalty value (always >= 0, capped at MAX_TOTAL_PENALTY).
    """
    raw = sum(ev["penalty"] for ev in tracker["events"])
    return round(min(raw, MAX_TOTAL_PENALTY), 4)


def classify_risk(tracker):
    """Classify risk level based on assumption count and severity.

    Returns:
        str — "low", "moderate", "elevated", or "high".
    """
    count = tracker["event_count"]

    # Heavy repairs or many mapper defaults escalate risk
    repair_count = sum(1 for ev in tracker["events"] if ev["type"] == "repair")
    if repair_count >= 3:
        return "high"

    if count >= RISK_THRESHOLDS["high"]:
        return "high"
    elif count >= RISK_THRESHOLDS["elevated"]:
        return "elevated"
    elif count >= RISK_THRESHOLDS["moderate"]:
        return "moderate"
    else:
        return "low"


def format_for_delivery_report(tracker):
    """Format assumptions for inclusion in the delivery report.

    Returns:
        dict with:
            - assumption_count: int
            - risk_level: str
            - items: list of human-readable assumption strings
            - recommendation: str
    """
    items = []
    for ev in tracker["events"]:
        items.append(
            f"[{ev['type']}] {ev['module']}.{ev['field']} = "
            f"{_format_value(ev['assumed_value'])} — {ev['reason']}"
        )

    risk = classify_risk(tracker)
    if risk == "low":
        recommendation = "No assumptions were made. Build is fully specified."
    elif risk == "moderate":
        recommendation = (
            "A small number of assumptions were made. "
            "Review the items below and confirm they match your requirements."
        )
    elif risk == "elevated":
        recommendation = (
            "Several assumptions were made during the build. "
            "We recommend reviewing each item carefully before deployment."
        )
    else:
        recommendation = (
            "A high number of assumptions were made. "
            "This build may not match your requirements without revision. "
            "Please review all items and provide clarifications."
        )

    return {
        "assumption_count": tracker["event_count"],
        "risk_level": risk,
        "items": items,
        "recommendation": recommendation,
    }


def format_for_build_log(tracker):
    """Format assumptions as a markdown section for the build log.

    Returns:
        str — markdown text.
    """
    if tracker["event_count"] == 0:
        return (
            "## Assumptions\n\n"
            "No assumptions were introduced during this build.\n"
        )

    lines = [
        "## Assumptions",
        "",
        f"**{tracker['event_count']}** assumption(s) recorded "
        f"(risk: **{classify_risk(tracker)}**, "
        f"confidence penalty: **-{compute_confidence_penalty(tracker):.2f}**)",
        "",
        "| # | Source | Type | Module | Field | Value | Reason |",
        "|---|--------|------|--------|-------|-------|--------|",
    ]

    for ev in tracker["events"]:
        val = _format_value(ev["assumed_value"])
        lines.append(
            f"| {ev['id']} | {ev['source']} | {ev['type']} "
            f"| {ev['module']} | {ev['field']} | {val} | {ev['reason']} |"
        )

    lines.append("")
    return "\n".join(lines)


def format_for_delivery_pack(tracker):
    """Format assumptions for inclusion in delivery_pack.json.

    Returns:
        dict — machine-readable assumptions section.
    """
    ledger = get_ledger(tracker)
    return {
        "assumption_count": ledger["event_count"],
        "risk_level": ledger["risk_level"],
        "confidence_penalty": ledger["total_penalty"],
        "by_source": ledger["by_source"],
        "by_type": ledger["by_type"],
        "events": [
            {
                "id": ev["id"],
                "source": ev["source"],
                "type": ev["type"],
                "module": ev["module"],
                "field": ev["field"],
                "assumed_value": _format_value(ev["assumed_value"]),
                "reason": ev["reason"],
                "penalty": ev["penalty"],
            }
            for ev in ledger["events"]
        ],
    }


def _format_value(val):
    """Format an assumed value for display (truncate long values)."""
    s = str(val)
    if len(s) > 60:
        return s[:57] + "..."
    return s


# --- Self-check ---
if __name__ == "__main__":
    print("=== Assumption Tracker Self-Check ===\n")

    # Test 1: Create and record
    print("Test 1: Create tracker and record assumptions")
    t = create_tracker()
    assert t["event_count"] == 0
    assert t["events"] == []

    record(t, "delivery_assessment", "module_resolution", "slack:PostMessage",
           "module", "slack:PostMessage", "Auto-resolved from app='slack' desc='Post notification'")
    record(t, "default_mapper", "mapper_default", "json:ParseJSON",
           "json", "{{1.body}}", "Required mapper field 'json' not provided; injected default")
    record(t, "default_parameters", "parameter_default", "google-sheets:addRow",
           "spreadsheetId", "__SPREADSHEET_ID__", "Required parameter not provided; injected placeholder")

    assert t["event_count"] == 3
    assert len(t["events"]) == 3
    assert t["events"][0]["id"] == 1
    assert t["events"][2]["id"] == 3
    assert t["events"][0]["source"] == "delivery_assessment"
    assert t["events"][1]["type"] == "mapper_default"
    print(f"  Recorded: {t['event_count']} events")
    print("  [OK]")

    # Test 2: Ledger structure
    print("\nTest 2: Get ledger")
    ledger = get_ledger(t)
    assert ledger["event_count"] == 3
    assert ledger["by_source"]["delivery_assessment"] == 1
    assert ledger["by_source"]["default_mapper"] == 1
    assert ledger["by_source"]["default_parameters"] == 1
    assert ledger["by_type"]["module_resolution"] == 1
    assert ledger["by_type"]["mapper_default"] == 1
    assert ledger["by_type"]["parameter_default"] == 1
    assert ledger["risk_level"] == "moderate"
    print(f"  By source: {ledger['by_source']}")
    print(f"  By type: {ledger['by_type']}")
    print(f"  Risk: {ledger['risk_level']}")
    print("  [OK]")

    # Test 3: Confidence penalty calculation
    print("\nTest 3: Confidence penalty")
    # module_resolution=0.02 + mapper_default=0.03 + parameter_default=0.03 = 0.08
    penalty = compute_confidence_penalty(t)
    assert penalty == 0.08, f"Expected 0.08, got {penalty}"
    print(f"  Penalty: {penalty}")

    # Test penalty cap
    t_heavy = create_tracker()
    for i in range(20):
        record(t_heavy, "self_heal", "repair", f"module_{i}", "field",
               "fixed", "Repair pass")
    heavy_penalty = compute_confidence_penalty(t_heavy)
    assert heavy_penalty == MAX_TOTAL_PENALTY, f"Expected cap at {MAX_TOTAL_PENALTY}, got {heavy_penalty}"
    print(f"  Heavy penalty (20 repairs): {heavy_penalty} (capped at {MAX_TOTAL_PENALTY})")
    print("  [OK]")

    # Test 4: Risk classification
    print("\nTest 4: Risk classification")

    t_low = create_tracker()
    assert classify_risk(t_low) == "low"

    t_mod = create_tracker()
    record(t_mod, "normalize", "field_inference", "m", "f", "v", "r")
    assert classify_risk(t_mod) == "moderate"

    t_elev = create_tracker()
    for i in range(5):
        record(t_elev, "normalize", "mapper_default", f"m{i}", "f", "v", "r")
    assert classify_risk(t_elev) == "elevated"

    t_high = create_tracker()
    for i in range(10):
        record(t_high, "normalize", "mapper_default", f"m{i}", "f", "v", "r")
    assert classify_risk(t_high) == "high"

    # Repair escalation: 3+ repairs → high even with low count
    t_repair = create_tracker()
    for i in range(3):
        record(t_repair, "self_heal", "repair", f"m{i}", "f", "v", "repair pass")
    assert classify_risk(t_repair) == "high", f"3 repairs should be high, got {classify_risk(t_repair)}"

    print("  low: 0 events ✓")
    print("  moderate: 1 event ✓")
    print("  elevated: 5 events ✓")
    print("  high: 10 events ✓")
    print("  high: 3 repairs ✓")
    print("  [OK]")

    # Test 5: Format for delivery report
    print("\nTest 5: Format for delivery report")
    dr = format_for_delivery_report(t)
    assert dr["assumption_count"] == 3
    assert dr["risk_level"] == "moderate"
    assert len(dr["items"]) == 3
    assert "module_resolution" in dr["items"][0]
    assert "recommendation" in dr
    assert len(dr["recommendation"]) > 0
    print(f"  Items: {len(dr['items'])}")
    print(f"  Risk: {dr['risk_level']}")
    print(f"  Recommendation: {dr['recommendation'][:60]}...")
    print("  [OK]")

    # Test 6: Format for build log (markdown)
    print("\nTest 6: Format for build log")
    md = format_for_build_log(t)
    assert "## Assumptions" in md
    assert "3" in md
    assert "moderate" in md
    assert "module_resolution" in md
    assert "|" in md  # table
    print(f"  Markdown: {len(md)} chars")
    print(f"  First line: {md.splitlines()[0]}")

    # Empty tracker
    md_empty = format_for_build_log(create_tracker())
    assert "No assumptions" in md_empty
    print(f"  Empty tracker: '{md_empty.splitlines()[2]}'")
    print("  [OK]")

    # Test 7: Format for delivery pack
    print("\nTest 7: Format for delivery pack")
    pack = format_for_delivery_pack(t)
    assert pack["assumption_count"] == 3
    assert pack["risk_level"] == "moderate"
    assert pack["confidence_penalty"] == 0.08
    assert len(pack["events"]) == 3
    assert pack["events"][0]["source"] == "delivery_assessment"
    assert "by_source" in pack
    assert "by_type" in pack

    import json
    json_str = json.dumps(pack, indent=2)
    roundtrip = json.loads(json_str)
    assert roundtrip == pack, "JSON roundtrip failed"
    print(f"  Pack keys: {list(pack.keys())}")
    print(f"  JSON size: {len(json_str)} chars")
    print("  [OK]")

    # Test 8: Determinism
    print("\nTest 8: Determinism")
    t_a = create_tracker()
    t_b = create_tracker()
    for tr in (t_a, t_b):
        record(tr, "delivery_assessment", "module_resolution", "slack:PostMessage",
               "module", "slack:PostMessage", "Auto-resolved")
        record(tr, "default_mapper", "mapper_default", "json:ParseJSON",
               "json", "{{1.body}}", "Injected default")
    assert get_ledger(t_a) == get_ledger(t_b)
    assert format_for_build_log(t_a) == format_for_build_log(t_b)
    assert format_for_delivery_pack(t_a) == format_for_delivery_pack(t_b)
    assert format_for_delivery_report(t_a) == format_for_delivery_report(t_b)
    print("  [OK]")

    print("\n=== All assumption_tracker checks passed ===")
