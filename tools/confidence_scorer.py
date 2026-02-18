"""
Confidence Scorer

Computes a confidence score (0.0 to 1.0) for a generated canonical spec
based on validation results and generation context.

Input: validation_report (dict), agent_notes (list[str]), retry_count (int)
Output: dict with score, factors, grade, explanation

Scoring formula:
  Start at 1.0
  -0.02 per warning (max -0.10)
  -0.02 per assumption/agent_note (max -0.10)
  -0.10 per retry (max -0.20)
  -0.50 if any validation error exists
  Floor at 0.0

Deterministic. No network calls. No conversation context.
"""


def compute_confidence(validation_report, agent_notes=None, retry_count=0):
    """Compute a confidence score for a canonical spec build.

    Args:
        validation_report: Output from validate_canonical_spec().
        agent_notes: List of assumption/decision strings from spec generation.
        retry_count: Number of retries needed during build (0 = first attempt).

    Returns:
        dict with:
            - score: float (0.0 to 1.0)
            - factors: dict of individual factor contributions
            - grade: str (A, B, C, D, F)
            - explanation: str (human-readable summary)
    """
    if agent_notes is None:
        agent_notes = []

    score = 1.0
    factors = {}

    # Factor 1: Validation errors
    error_count = len(validation_report.get("errors", []))
    if error_count > 0:
        factors["validation_error_penalty"] = -0.50
        score -= 0.50
    else:
        factors["validation_error_penalty"] = 0.0

    # Factor 2: Warnings
    warning_count = len(validation_report.get("warnings", []))
    warning_penalty = min(warning_count * 0.02, 0.10)
    factors["warning_penalty"] = -warning_penalty
    score -= warning_penalty

    # Factor 3: Assumptions
    assumption_count = len(agent_notes)
    assumption_penalty = min(assumption_count * 0.02, 0.10)
    factors["assumption_penalty"] = -assumption_penalty
    score -= assumption_penalty

    # Factor 4: Retries
    retry_penalty = min(retry_count * 0.10, 0.20)
    factors["retry_penalty"] = -retry_penalty
    score -= retry_penalty

    # Factor 5: Validation pass rate
    checks_run = validation_report.get("checks_run", 0)
    checks_passed = validation_report.get("checks_passed", 0)
    if checks_run > 0:
        factors["validation_pass_rate"] = round(checks_passed / checks_run, 4)
    else:
        factors["validation_pass_rate"] = 0.0

    # Floor at 0.0
    score = max(round(score, 4), 0.0)

    # Grade
    if score >= 0.90:
        grade = "A"
    elif score >= 0.80:
        grade = "B"
    elif score >= 0.70:
        grade = "C"
    elif score >= 0.50:
        grade = "D"
    else:
        grade = "F"

    # Explanation
    parts = []
    if error_count > 0:
        parts.append(f"{error_count} validation error(s)")
    if warning_count > 0:
        parts.append(f"{warning_count} warning(s)")
    if assumption_count > 0:
        parts.append(f"{assumption_count} assumption(s)")
    if retry_count > 0:
        parts.append(f"{retry_count} retry(ies)")
    if not parts:
        explanation = "Spec passed all validation checks with no warnings, assumptions, or retries."
    else:
        explanation = "Score affected by: " + ", ".join(parts) + "."

    return {
        "score": score,
        "factors": factors,
        "grade": grade,
        "explanation": explanation
    }


# --- Self-check ---
if __name__ == "__main__":
    print("=== Confidence Scorer Self-Check ===\n")

    # Test 1: Perfect score
    print("Test 1: Perfect score (no errors, warnings, assumptions, retries)")
    result = compute_confidence(
        validation_report={"errors": [], "warnings": [], "checks_run": 40, "checks_passed": 40},
        agent_notes=[],
        retry_count=0
    )
    assert result["score"] == 1.0, f"Expected 1.0, got {result['score']}"
    assert result["grade"] == "A"
    print(f"  Score: {result['score']} ({result['grade']})")
    print(f"  {result['explanation']}")
    print("  [OK]")

    # Test 2: Warnings reduce score
    print("\nTest 2: 3 warnings")
    result = compute_confidence(
        validation_report={"errors": [], "warnings": [1, 2, 3], "checks_run": 40, "checks_passed": 37},
        agent_notes=[],
        retry_count=0
    )
    assert result["score"] == 0.94, f"Expected 0.94, got {result['score']}"
    assert result["grade"] == "A"
    print(f"  Score: {result['score']} ({result['grade']})")
    print("  [OK]")

    # Test 3: Assumptions reduce score
    print("\nTest 3: 2 assumptions + 1 warning")
    result = compute_confidence(
        validation_report={"errors": [], "warnings": [1], "checks_run": 40, "checks_passed": 39},
        agent_notes=["Assumed X", "Assumed Y"],
        retry_count=0
    )
    assert result["score"] == 0.94, f"Expected 0.94, got {result['score']}"
    print(f"  Score: {result['score']} ({result['grade']})")
    print("  [OK]")

    # Test 4: Retries reduce score
    print("\nTest 4: 1 retry")
    result = compute_confidence(
        validation_report={"errors": [], "warnings": [], "checks_run": 40, "checks_passed": 40},
        agent_notes=[],
        retry_count=1
    )
    assert result["score"] == 0.9, f"Expected 0.9, got {result['score']}"
    assert result["grade"] == "A"
    print(f"  Score: {result['score']} ({result['grade']})")
    print("  [OK]")

    # Test 5: Errors cause major penalty
    print("\nTest 5: Validation errors")
    result = compute_confidence(
        validation_report={"errors": [1, 2], "warnings": [1], "checks_run": 40, "checks_passed": 37},
        agent_notes=["Assumed something"],
        retry_count=1
    )
    assert result["score"] <= 0.5, f"Expected <= 0.5, got {result['score']}"
    assert result["grade"] in ("D", "F")
    print(f"  Score: {result['score']} ({result['grade']})")
    print("  [OK]")

    # Test 6: Maximum penalties (caps)
    print("\nTest 6: Maximum penalties capped")
    result = compute_confidence(
        validation_report={"errors": [1], "warnings": list(range(20)), "checks_run": 40, "checks_passed": 20},
        agent_notes=list(range(20)),
        retry_count=5
    )
    assert result["score"] >= 0.0, "Score must not go below 0.0"
    assert result["factors"]["warning_penalty"] == -0.10, "Warning penalty capped at -0.10"
    assert result["factors"]["assumption_penalty"] == -0.10, "Assumption penalty capped at -0.10"
    assert result["factors"]["retry_penalty"] == -0.20, "Retry penalty capped at -0.20"
    print(f"  Score: {result['score']} ({result['grade']})")
    print(f"  Factors: {result['factors']}")
    print("  [OK]")

    print("\n=== All confidence scorer checks passed ===")
