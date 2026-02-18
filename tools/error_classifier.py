"""
Error Classifier

Classifies Make.com execution errors into typed incidents with severity.
Used by the live monitor to distinguish credential failures from
rate limits from logic errors.

Pattern matching is intentional — no ML required.
Fast, deterministic, no external calls.
"""

import re
from typing import Optional

INCIDENT_TYPES = [
    "credential_expired",
    "credential_invalid",
    "rate_limit",
    "connection_timeout",
    "webhook_dead",
    "logic_error",
    "data_format_error",
    "quota_exceeded",
    "permission_denied",
    "not_found",
    "unknown",
]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

# Pattern → (incident_type, severity)
ERROR_PATTERNS = [
    # Credential failures
    (r"(401|unauthorized|invalid.*token|token.*invalid|api.*key.*invalid)", "credential_invalid", "high"),
    (r"(token.*expired|session.*expired|refresh.*token|re-authenticate)", "credential_expired", "high"),
    (r"(403|forbidden|permission.*denied|access.*denied)", "permission_denied", "high"),

    # Rate limits
    (r"(429|rate.*limit|too.*many.*requests|throttl)", "rate_limit", "medium"),
    (r"(quota.*exceeded|daily.*limit|monthly.*limit|api.*limit)", "quota_exceeded", "medium"),

    # Connection issues
    (r"(timeout|timed.*out|connection.*refused|ECONNREFUSED)", "connection_timeout", "medium"),
    (r"(ENOTFOUND|dns.*fail|could.*not.*resolve|host.*not.*found)", "connection_timeout", "high"),

    # Data issues
    (r"(invalid.*json|parse.*error|unexpected.*token|json.*syntax)", "data_format_error", "low"),
    (r"(missing.*field|required.*field|field.*required|cannot.*be.*empty)", "data_format_error", "low"),
    (r"(type.*error|cannot.*cast|invalid.*type)", "data_format_error", "low"),

    # Resource not found
    (r"(404|not.*found|does.*not.*exist|no.*such)", "not_found", "low"),

    # Webhook dead
    (r"(webhook.*invalid|hook.*not.*found|webhook.*deleted)", "webhook_dead", "critical"),
]


def classify_error(error_message: str, context: dict = None) -> dict:
    """
    Classify a single error message.

    Args:
        error_message: Raw error string from Make.com execution
        context: Optional dict with additional context (module, step, etc.)

    Returns:
        dict with incident_type, severity, confidence, suggestion
    """
    if not error_message:
        return {
            "incident_type": "unknown",
            "severity": "low",
            "confidence": 0.5,
            "suggestion": "No error message provided.",
        }

    msg = error_message.lower()

    for pattern, incident_type, severity in ERROR_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            suggestion = _get_suggestion(incident_type, context)
            return {
                "incident_type": incident_type,
                "severity": severity,
                "confidence": 0.85,
                "matched_pattern": pattern,
                "suggestion": suggestion,
            }

    return {
        "incident_type": "logic_error",
        "severity": "medium",
        "confidence": 0.4,
        "suggestion": "Review the scenario execution log for the specific step that failed.",
    }


def _get_suggestion(incident_type: str, context: dict = None) -> str:
    module = (context or {}).get("module", "the affected module")
    suggestions = {
        "credential_invalid": f"Re-authenticate the connection in {module}. The API key or OAuth token may have changed.",
        "credential_expired": f"Refresh the OAuth token or re-create the connection for {module}.",
        "rate_limit": "The API rate limit was hit. Consider adding a delay between executions or upgrading your API plan.",
        "quota_exceeded": "Monthly API quota exceeded. Upgrade your plan or wait for the quota to reset.",
        "connection_timeout": "Network connectivity issue. Check if the target service is online and reachable from Make.com.",
        "webhook_dead": "The webhook endpoint is invalid or was deleted. Recreate the webhook and update the trigger.",
        "data_format_error": "Invalid data format. Check the input data shape expected by this step and verify upstream mappings.",
        "permission_denied": f"Insufficient permissions. Ensure the connected account has the required access for {module}.",
        "not_found": "The target resource (record, file, channel, etc.) was not found. It may have been deleted.",
        "logic_error": "Unexpected execution error. Review the step configuration and test with a simple payload.",
        "unknown": "Unknown error. Check the full execution log in Make.com for details.",
    }
    return suggestions.get(incident_type, suggestions["unknown"])


def classify_execution_history(executions: list) -> dict:
    """
    Analyze a list of executions and detect patterns.

    Args:
        executions: List of Make.com execution dicts from API

    Returns:
        dict with:
            total, success_count, error_count, success_rate,
            dominant_error_type, incidents, needs_attention
    """
    total = len(executions)
    if total == 0:
        return {
            "total": 0, "success_count": 0, "error_count": 0,
            "success_rate": None, "dominant_error_type": None,
            "incidents": [], "needs_attention": False,
        }

    success_count = sum(1 for e in executions if e.get("status") == "success")
    error_count = total - success_count
    success_rate = round(success_count / total, 3)

    # Classify all errors
    incidents = []
    error_type_counts = {}
    for exe in executions:
        if exe.get("status") != "success":
            err_msg = exe.get("error", exe.get("message", ""))
            classification = classify_error(str(err_msg))
            incidents.append({
                "execution_id": exe.get("id", exe.get("executionId", "")),
                "error": err_msg,
                "classification": classification,
            })
            t = classification["incident_type"]
            error_type_counts[t] = error_type_counts.get(t, 0) + 1

    dominant_error_type = (
        max(error_type_counts, key=error_type_counts.get)
        if error_type_counts else None
    )

    # Determine if attention needed
    needs_attention = (
        error_count >= 3
        or success_rate < 0.5
        or dominant_error_type in ("credential_invalid", "credential_expired", "webhook_dead")
    )

    return {
        "total": total,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_rate,
        "error_type_counts": error_type_counts,
        "dominant_error_type": dominant_error_type,
        "incidents": incidents,
        "needs_attention": needs_attention,
    }


if __name__ == "__main__":
    print("=== Error Classifier Self-Check ===\n")

    print("Test 1: Credential invalid pattern")
    result = classify_error("401 Unauthorized: invalid_token")
    assert result["incident_type"] == "credential_invalid"
    assert result["severity"] == "high"
    print(f"  Type: {result['incident_type']}, Severity: {result['severity']}")
    print("  [OK]")

    print("Test 2: Rate limit pattern")
    result = classify_error("429 Too Many Requests: rate limit exceeded")
    assert result["incident_type"] == "rate_limit"
    assert result["severity"] == "medium"
    print("  [OK]")

    print("Test 3: Timeout pattern")
    result = classify_error("ECONNREFUSED connection refused to api.example.com:443")
    assert result["incident_type"] == "connection_timeout"
    print("  [OK]")

    print("Test 4: Data format error")
    result = classify_error("SyntaxError: Unexpected token in JSON")
    assert result["incident_type"] == "data_format_error"
    print("  [OK]")

    print("Test 5: Unknown error → logic_error fallback")
    result = classify_error("Something went very wrong in step 3")
    assert result["incident_type"] == "logic_error"
    print("  [OK]")

    print("Test 6: classify_execution_history")
    executions = [
        {"id": "1", "status": "success"},
        {"id": "2", "status": "failed", "error": "401 Unauthorized"},
        {"id": "3", "status": "failed", "error": "401 invalid_token"},
        {"id": "4", "status": "success"},
        {"id": "5", "status": "failed", "error": "token expired"},
    ]
    analysis = classify_execution_history(executions)
    assert analysis["success_count"] == 2
    assert analysis["error_count"] == 3
    assert analysis["dominant_error_type"] in ("credential_invalid", "credential_expired")
    assert analysis["needs_attention"] == True
    print(f"  Dominant error: {analysis['dominant_error_type']}")
    print(f"  Success rate: {analysis['success_rate']}")
    print("  [OK]")

    print("Test 7: Empty execution list")
    result = classify_execution_history([])
    assert result["total"] == 0
    assert result["success_rate"] is None
    print("  [OK]")

    print("\n=== All error classifier checks passed ===")
