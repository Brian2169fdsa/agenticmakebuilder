"""
Integration Test Generator

Given a canonical spec, generates a suite of test cases:
  1. Happy path — valid inputs, all steps should pass
  2. Edge cases — empty fields, unexpected types
  3. Error path — inputs that should trigger error handling

Test cases are synthetic — generated from the spec structure,
no real external data needed.
"""

import json
import os
import random
import string


SAMPLE_EMAILS = ["test@example.com", "user@acme.com", "noreply@domain.org"]
SAMPLE_NAMES = ["John Smith", "Jane Doe", "Alex Johnson"]
SAMPLE_COMPANIES = ["Acme Corp", "Beta Inc", "Gamma LLC"]
SAMPLE_PHONES = ["+1-555-0100", "+1-555-0200", "+1-555-0300"]


def _random_string(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=length))


def _infer_field_type(key: str, value=None) -> str:
    key_lower = key.lower()
    if any(x in key_lower for x in ["email", "mail"]):
        return "email"
    if any(x in key_lower for x in ["phone", "tel", "mobile"]):
        return "phone"
    if any(x in key_lower for x in ["name", "first", "last"]):
        return "name"
    if any(x in key_lower for x in ["company", "org", "business"]):
        return "company"
    if any(x in key_lower for x in ["amount", "price", "cost", "total"]):
        return "number"
    if any(x in key_lower for x in ["date", "time", "at", "when"]):
        return "datetime"
    if any(x in key_lower for x in ["url", "link", "href", "webhook"]):
        return "url"
    if any(x in key_lower for x in ["id", "_id", "identifier"]):
        return "id"
    return "text"


def _generate_value(field_type: str, variant: str = "happy") -> any:
    if variant == "empty":
        return ""
    if variant == "null":
        return None

    generators = {
        "email": lambda: random.choice(SAMPLE_EMAILS),
        "phone": lambda: random.choice(SAMPLE_PHONES),
        "name": lambda: random.choice(SAMPLE_NAMES),
        "company": lambda: random.choice(SAMPLE_COMPANIES),
        "number": lambda: random.randint(10, 9999),
        "datetime": lambda: "2026-02-17T10:00:00Z",
        "url": lambda: f"https://example.com/{_random_string()}",
        "id": lambda: _random_string(12),
        "text": lambda: f"Test {_random_string(6)} value",
    }
    return generators.get(field_type, generators["text"])()


def _extract_trigger_fields(trigger: dict) -> dict:
    """
    Infer expected webhook payload fields from trigger config.
    Falls back to generic form fields if nothing specific found.
    """
    params = trigger.get("parameters", {})
    fields = {}

    # Try to extract from explicit field config
    for key, val in params.items():
        if key in ("hook", "connection", "webhook"):
            continue
        ftype = _infer_field_type(key, val)
        fields[key] = ftype

    # Generic fallback — most forms have these
    if not fields:
        fields = {
            "email": "email",
            "name": "name",
            "message": "text",
            "company": "company",
        }

    return fields


def _extract_mapper_fields(steps: list) -> dict:
    """Extract all referenced fields from step mappers."""
    import re
    fields = {}
    for step in steps:
        mapper = step.get("mapper", {})
        for key, val in mapper.items():
            if isinstance(val, str):
                # Extract field names from {{N.field}} patterns
                matches = re.findall(r"\{\{[\d]+\.([^}]+)\}\}", val)
                for match in matches:
                    field = match.split(".")[0]
                    if field and not field.isdigit():
                        ftype = _infer_field_type(field)
                        fields[field] = ftype
    return fields


def generate_test_suite(canonical_spec: dict) -> dict:
    """
    Generate a test suite from a canonical spec.

    Returns:
        dict with:
            test_cases: list of test case dicts
            trigger_type: webhook / schedule / email / polling
            field_schema: inferred fields and types
    """
    trigger = canonical_spec.get("trigger", {})
    steps = canonical_spec.get("steps", [])
    trigger_type = trigger.get("type", "webhook")

    # Infer fields
    trigger_fields = _extract_trigger_fields(trigger)
    mapper_fields = _extract_mapper_fields(steps)
    all_fields = {**trigger_fields, **mapper_fields}

    # Remove internal/system fields
    skip_fields = {"connection", "hook", "webhookId", "__PLACEHOLDER__"}
    field_schema = {k: v for k, v in all_fields.items()
                    if k not in skip_fields and not k.startswith("__")}

    test_cases = []

    # ── Test 1: Happy path ──
    happy_payload = {k: _generate_value(v, "happy") for k, v in field_schema.items()}
    test_cases.append({
        "id": "TC-001",
        "name": "Happy Path",
        "description": "Valid inputs — all steps should pass",
        "variant": "happy",
        "payload": happy_payload,
        "expected": {
            "success": True,
            "all_steps_pass": True,
            "error_count": 0,
        },
    })

    # ── Test 2: Missing required field ──
    if field_schema:
        first_key = list(field_schema.keys())[0]
        missing_payload = dict(happy_payload)
        del missing_payload[first_key]
        test_cases.append({
            "id": "TC-002",
            "name": "Missing Required Field",
            "description": f"Payload missing '{first_key}' — scenario should handle gracefully",
            "variant": "missing_field",
            "payload": missing_payload,
            "expected": {
                "success": True,  # Should still run, not crash
                "notes": f"Field '{first_key}' missing — verify error handling",
            },
        })

    # ── Test 3: Empty string values ──
    empty_payload = {k: "" for k in field_schema}
    test_cases.append({
        "id": "TC-003",
        "name": "Empty String Values",
        "description": "All fields empty — tests null/empty handling in each step",
        "variant": "empty",
        "payload": empty_payload,
        "expected": {
            "success": True,
            "notes": "Verify downstream steps handle empty strings without crashing",
        },
    })

    # ── Test 4: Email-specific test (if email field present) ──
    email_fields = [k for k, v in field_schema.items() if v == "email"]
    if email_fields:
        bad_email_payload = dict(happy_payload)
        bad_email_payload[email_fields[0]] = "not-a-valid-email"
        test_cases.append({
            "id": "TC-004",
            "name": "Invalid Email Format",
            "description": "Invalid email value — tests input validation",
            "variant": "bad_email",
            "payload": bad_email_payload,
            "expected": {
                "success": True,
                "notes": "Scenario should not crash on bad email — verify behavior",
            },
        })

    # ── Test 5: Large payload ──
    large_payload = dict(happy_payload)
    for k in field_schema:
        if field_schema[k] == "text":
            large_payload[k] = "A" * 5000
    test_cases.append({
        "id": "TC-005",
        "name": "Large Payload",
        "description": "Oversized text fields — tests Make.com data limits",
        "variant": "large",
        "payload": large_payload,
        "expected": {
            "success": True,
            "notes": "Watch for Make.com payload size limits (10MB per execution)",
        },
    })

    return {
        "scenario_name": canonical_spec.get("scenario_name", "Scenario"),
        "trigger_type": trigger_type,
        "field_schema": field_schema,
        "test_case_count": len(test_cases),
        "test_cases": test_cases,
    }


if __name__ == "__main__":
    print("=== Test Generator Self-Check ===\n")

    sample_spec = {
        "scenario_name": "Acme Form to Slack",
        "trigger": {
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Form Submit",
            "parameters": {},
        },
        "steps": [
            {"module": "json:ParseJSON", "label": "Parse",
             "mapper": {"email": "{{1.email}}", "name": "{{1.name}}", "company": "{{1.company}}"},
             "parameters": {}},
            {"module": "slack:PostMessage", "label": "Notify",
             "mapper": {"text": "New lead: {{2.name}} from {{2.company}}"},
             "parameters": {}},
        ],
        "error_handling": {"default_strategy": "ignore"},
    }

    print("Test 1: Generate test suite")
    suite = generate_test_suite(sample_spec)
    assert suite["test_case_count"] >= 3
    assert len(suite["test_cases"]) == suite["test_case_count"]
    print(f"  Test cases: {suite['test_case_count']}")
    print(f"  Fields: {list(suite['field_schema'].keys())}")
    print("  [OK]")

    print("Test 2: Happy path payload has correct structure")
    happy = next(tc for tc in suite["test_cases"] if tc["id"] == "TC-001")
    assert happy["expected"]["success"] == True
    assert isinstance(happy["payload"], dict)
    print(f"  Payload keys: {list(happy['payload'].keys())}")
    print("  [OK]")

    print("Test 3: All test cases have required fields")
    for tc in suite["test_cases"]:
        assert "id" in tc
        assert "name" in tc
        assert "payload" in tc
        assert "expected" in tc
    print("  [OK]")

    print("Test 4: Field types inferred correctly")
    assert suite["field_schema"].get("email") == "email"
    assert suite["field_schema"].get("name") == "name"
    assert suite["field_schema"].get("company") == "company"
    print("  [OK]")

    print("\n=== All test generator checks passed ===")
