"""
Data Mapping Extractor

Extracts and validates all {{N.field}} data mapping references
from a canonical spec. Scans all string values in parameters,
mapper, and filter conditions.

Input: spec (dict) — full canonical spec
Output: dict with references, invalid_references, forward_references

Deterministic. No network calls. No conversation context.
"""

import re


# Pattern matches {{N.field}} or {{N.field.subfield}} where N is an integer
REFERENCE_PATTERN = re.compile(r"\{\{(\d+)\.([^}]+)\}\}")


def extract_data_mappings(spec):
    """Extract all data mapping references from a canonical spec.

    Scans all string values in:
    - trigger.parameters, trigger.webhook
    - modules[].parameters, modules[].mapper
    - connections[].filter.conditions

    Args:
        spec: Full canonical spec dict.

    Returns:
        dict with:
            - references: list of extracted reference dicts
            - reference_count: int
            - referenced_module_ids: sorted list of unique module IDs referenced
    """
    references = []

    # Collect all module IDs for context
    trigger_id = spec.get("trigger", {}).get("id", 1)
    module_ids = {trigger_id}
    for mod in spec.get("modules", []):
        module_ids.add(mod.get("id"))

    # Scan trigger
    _scan_object(
        obj=spec.get("trigger", {}).get("parameters", {}),
        context_module_id=trigger_id,
        context_location="trigger.parameters",
        references=references
    )

    # Scan modules
    for mod in spec.get("modules", []):
        mid = mod.get("id")
        _scan_object(
            obj=mod.get("parameters", {}),
            context_module_id=mid,
            context_location=f"module[{mid}].parameters",
            references=references
        )
        _scan_object(
            obj=mod.get("mapper", {}),
            context_module_id=mid,
            context_location=f"module[{mid}].mapper",
            references=references
        )

    # Scan connection filters
    for i, conn in enumerate(spec.get("connections", [])):
        filt = conn.get("filter")
        if filt and isinstance(filt, dict):
            conditions = filt.get("conditions", [])
            _scan_object(
                obj=conditions,
                context_module_id=conn.get("to"),
                context_location=f"connection[{i}].filter.conditions",
                references=references
            )

    # Collect unique referenced module IDs
    referenced_ids = sorted(set(r["source_module_id"] for r in references))

    return {
        "references": references,
        "reference_count": len(references),
        "referenced_module_ids": referenced_ids
    }


def validate_data_mappings(spec, graph_result):
    """Validate all data mapping references against the spec graph.

    Args:
        spec: Full canonical spec dict.
        graph_result: Output from graph_integrity_check().

    Returns:
        dict with:
            - valid: bool — True if all references are valid
            - errors: list of error dicts
            - warnings: list of warning dicts
    """
    extraction = extract_data_mappings(spec)
    trigger_id = spec.get("trigger", {}).get("id", 1)
    all_module_ids = {trigger_id}
    for mod in spec.get("modules", []):
        all_module_ids.add(mod.get("id"))

    errors = []
    warnings = []

    for ref in extraction["references"]:
        src_id = ref["source_module_id"]
        ctx_id = ref["context_module_id"]

        # DM-001: Reference points to valid module ID
        if src_id not in all_module_ids:
            errors.append({
                "rule_id": "DM-001",
                "severity": "error",
                "message": f"Reference {ref['raw']} points to non-existent module ID {src_id}",
                "module_id": ctx_id,
                "context": {"reference": ref["raw"], "location": ref["location"]}
            })
            continue

        # DM-002 / DM-003: Referenced module must precede the referencing module
        if ctx_id is not None and src_id != ctx_id:
            # Check that src_id can reach ctx_id (src precedes ctx in the graph)
            reachable = graph_result.get("reachability", {}).get(src_id, [])
            if ctx_id not in reachable:
                errors.append({
                    "rule_id": "DM-002",
                    "severity": "error",
                    "message": (
                        f"Reference {ref['raw']} in module {ctx_id} "
                        f"points to module {src_id} which does not precede it in the flow"
                    ),
                    "module_id": ctx_id,
                    "context": {"reference": ref["raw"], "location": ref["location"]}
                })

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "references_checked": len(extraction["references"])
    }


def _scan_object(obj, context_module_id, context_location, references):
    """Recursively scan an object for {{N.field}} references.

    Args:
        obj: The object to scan (dict, list, or string).
        context_module_id: The module ID where this object lives.
        context_location: String describing location for error reporting.
        references: List to append found references to (mutated in place).
    """
    if isinstance(obj, str):
        for match in REFERENCE_PATTERN.finditer(obj):
            references.append({
                "source_module_id": int(match.group(1)),
                "field": match.group(2),
                "context_module_id": context_module_id,
                "location": context_location,
                "raw": match.group(0)
            })
    elif isinstance(obj, dict):
        for key, val in sorted(obj.items()):
            _scan_object(val, context_module_id, f"{context_location}.{key}", references)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            _scan_object(item, context_module_id, f"{context_location}[{i}]", references)


# --- Self-check ---
if __name__ == "__main__":
    print("=== Data Mapping Extractor Self-Check ===\n")

    # Test 1: Extract from a simple spec
    print("Test 1: Basic extraction")
    spec = {
        "trigger": {"id": 1, "parameters": {"hook": "__WEBHOOK_ID__"}},
        "modules": [
            {
                "id": 2,
                "parameters": {},
                "mapper": {"json": "{{1.body}}"}
            },
            {
                "id": 3,
                "parameters": {},
                "mapper": {
                    "channel": "#general",
                    "text": "Hello {{2.name}}, your email is {{2.email}}"
                }
            }
        ],
        "connections": []
    }
    result = extract_data_mappings(spec)
    assert result["reference_count"] == 3, f"Expected 3 refs, got {result['reference_count']}"
    assert result["referenced_module_ids"] == [1, 2]
    print(f"  [OK] Found {result['reference_count']} references")
    for ref in result["references"]:
        print(f"       {ref['raw']} in {ref['location']}")

    # Test 2: Extract from filter conditions
    print("Test 2: Filter condition extraction")
    spec2 = {
        "trigger": {"id": 1, "parameters": {}},
        "modules": [
            {"id": 2, "parameters": {}, "mapper": {}},
            {"id": 3, "parameters": {}, "mapper": {}}
        ],
        "connections": [
            {
                "from": 2, "to": 3,
                "filter": {
                    "name": "Priority Check",
                    "conditions": [[{"a": "{{1.priority}}", "b": "high", "o": "text:equal"}]]
                }
            }
        ]
    }
    result2 = extract_data_mappings(spec2)
    assert result2["reference_count"] == 1
    assert result2["references"][0]["source_module_id"] == 1
    print(f"  [OK] Found filter reference: {result2['references'][0]['raw']}")

    # Test 3: No references in static values
    print("Test 3: No false positives")
    spec3 = {
        "trigger": {"id": 1, "parameters": {}},
        "modules": [
            {"id": 2, "parameters": {}, "mapper": {"url": "https://example.com", "method": "GET"}}
        ],
        "connections": []
    }
    result3 = extract_data_mappings(spec3)
    assert result3["reference_count"] == 0
    print("  [OK] No false positives in static values")

    # Test 4: Nested object scanning
    print("Test 4: Nested values")
    spec4 = {
        "trigger": {"id": 1, "parameters": {}},
        "modules": [
            {
                "id": 2,
                "parameters": {},
                "mapper": {
                    "values": ["{{1.name}}", "{{1.email}}", "static"]
                }
            }
        ],
        "connections": []
    }
    result4 = extract_data_mappings(spec4)
    assert result4["reference_count"] == 2
    print(f"  [OK] Found {result4['reference_count']} references in nested array")

    print("\n=== All data mapping extractor checks passed ===")
