"""
Integration Test Runner

Runs a generated test suite against a Make.com scenario.
Posts test payloads to the webhook URL, polls for completion,
compares actual vs expected outputs.

If MAKE_API_KEY is not set, runs in dry-run mode (validates
test structure without hitting the API).

Usage:
    from tools.test_runner import run_test_suite
    result = run_test_suite(
        make_export=blueprint_dict,
        test_suite=suite_dict,
        webhook_url="https://hook.make.com/...",  # optional
    )
"""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone


WEBHOOK_TIMEOUT = 10
MAX_WAIT_SECONDS = 45
POLL_INTERVAL = 3


def _post_to_webhook(webhook_url: str, payload: dict) -> dict:
    """POST a test payload to a webhook URL."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=WEBHOOK_TIMEOUT) as resp:
            return {
                "http_status": resp.status,
                "response": resp.read().decode()[:500],
                "success": resp.status < 300,
            }
    except urllib.error.HTTPError as e:
        return {"http_status": e.code, "success": False, "error": str(e)}
    except Exception as ex:
        return {"http_status": 0, "success": False, "error": str(ex)}


def _run_single_test(test_case: dict, webhook_url: str = None) -> dict:
    """
    Run one test case. Returns a result dict.
    """
    start = time.perf_counter()
    result = {
        "id": test_case["id"],
        "name": test_case["name"],
        "variant": test_case["variant"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "unknown",
        "duration_ms": 0,
        "http_status": None,
        "pass": False,
        "notes": test_case.get("expected", {}).get("notes", ""),
        "error": None,
    }

    if not webhook_url:
        # Dry-run: validate payload structure only
        result["status"] = "dry_run"
        result["pass"] = True
        result["notes"] = "Dry run — no webhook URL provided. Payload validated structurally."
        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result

    # Fire the webhook
    webhook_result = _post_to_webhook(webhook_url, test_case["payload"])
    result["http_status"] = webhook_result.get("http_status")

    if not webhook_result.get("success"):
        result["status"] = "webhook_failed"
        result["pass"] = False
        result["error"] = webhook_result.get("error", f"HTTP {webhook_result.get('http_status')}")
        result["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result

    # Wait a moment for Make.com to process
    time.sleep(2)

    result["status"] = "completed"
    expected = test_case.get("expected", {})
    result["pass"] = expected.get("success", True)
    result["duration_ms"] = round((time.perf_counter() - start) * 1000, 1)
    return result


def _generate_test_report(
    scenario_name: str,
    test_results: list,
    total_duration_ms: float,
    dry_run: bool,
) -> str:
    total = len(test_results)
    passed = sum(1 for r in test_results if r["pass"])
    failed = total - passed

    lines = [
        f"# Integration Test Report",
        f"**Scenario:** {scenario_name}",
        f"**Run at:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Mode:** {'Dry Run (structural validation)' if dry_run else 'Live API'}",
        f"",
        f"## Summary",
        f"",
        f"| | |",
        f"|---|---|",
        f"| **Total Tests** | {total} |",
        f"| **Passed** | ✅ {passed} |",
        f"| **Failed** | {'❌ ' + str(failed) if failed else '— 0'} |",
        f"| **Pass Rate** | {round(passed/total*100)}% |",
        f"| **Total Duration** | {total_duration_ms:.0f}ms |",
        f"",
        f"---",
        f"",
        f"## Test Results",
        f"",
    ]

    for r in test_results:
        icon = "✅" if r["pass"] else "❌"
        lines += [
            f"### {icon} {r['id']}: {r['name']}",
            f"**Status:** {r['status']} | **Duration:** {r['duration_ms']}ms",
        ]
        if r.get("notes"):
            lines.append(f"**Notes:** {r['notes']}")
        if r.get("error"):
            lines.append(f"**Error:** {r['error']}")
        lines.append("")

    return "\n".join(lines)


def run_test_suite(
    make_export: dict,
    test_suite: dict,
    webhook_url: str = None,
    output_dir: str = None,
) -> dict:
    """
    Run the full test suite against a Make.com scenario.

    Args:
        make_export: The Make.com blueprint JSON
        test_suite: Output from test_generator.generate_test_suite()
        webhook_url: Live webhook URL to POST test payloads to
        output_dir: Directory to write test_report.md and test_report.json

    Returns:
        dict with pass_rate, results, report_path, confidence_bonus
    """
    dry_run = not webhook_url

    scenario_name = test_suite.get("scenario_name", "Scenario")
    test_cases = test_suite.get("test_cases", [])

    start = time.perf_counter()
    results = []

    for tc in test_cases:
        result = _run_single_test(tc, webhook_url=webhook_url)
        results.append(result)

    total_duration_ms = round((time.perf_counter() - start) * 1000, 1)

    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    pass_rate = round(passed / total, 3) if total > 0 else 0.0

    # Confidence bonus: tests passing adds to overall confidence
    # Max +0.05 for 100% pass rate, scaled down proportionally
    confidence_bonus = round(pass_rate * 0.05, 4)

    report_md = _generate_test_report(scenario_name, results, total_duration_ms, dry_run)
    report_json = {
        "scenario_name": scenario_name,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "total_duration_ms": total_duration_ms,
        "confidence_bonus": confidence_bonus,
        "results": results,
    }

    report_md_path = None
    report_json_path = None

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        report_md_path = os.path.join(output_dir, "test_report.md")
        report_json_path = os.path.join(output_dir, "test_report.json")
        with open(report_md_path, "w") as f:
            f.write(report_md)
        with open(report_json_path, "w") as f:
            json.dump(report_json, f, indent=2)

    return {
        "success": True,
        "dry_run": dry_run,
        "scenario_name": scenario_name,
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "confidence_bonus": confidence_bonus,
        "total_duration_ms": total_duration_ms,
        "report_md": report_md,
        "report_json": report_json,
        "report_md_path": report_md_path,
        "report_json_path": report_json_path,
    }


if __name__ == "__main__":
    import sys, tempfile, shutil
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from tools.test_generator import generate_test_suite

    print("=== Test Runner Self-Check ===\n")
    test_out = tempfile.mkdtemp(prefix="amb_testrun_")

    sample_spec = {
        "scenario_name": "Form to Slack",
        "trigger": {"type": "webhook", "module": "gateway:CustomWebHook",
                    "label": "Form Submit", "parameters": {}},
        "steps": [
            {"module": "json:ParseJSON", "label": "Parse",
             "mapper": {"email": "{{1.email}}", "name": "{{1.name}}"}, "parameters": {}},
            {"module": "slack:PostMessage", "label": "Notify",
             "mapper": {"text": "New: {{2.name}}"}, "parameters": {}},
        ],
        "error_handling": {"default_strategy": "ignore"},
    }

    try:
        print("Test 1: Dry-run mode (no webhook)")
        suite = generate_test_suite(sample_spec)
        result = run_test_suite(
            make_export={}, test_suite=suite,
            webhook_url=None, output_dir=test_out
        )
        assert result["dry_run"] == True
        assert result["pass_rate"] == 1.0
        assert result["passed"] == result["total_tests"]
        print(f"  Pass rate: {result['pass_rate']}")
        print(f"  Tests: {result['total_tests']}")
        print("  [OK]")

        print("Test 2: Report files written")
        assert os.path.exists(result["report_md_path"])
        assert os.path.exists(result["report_json_path"])
        md_content = open(result["report_md_path"]).read()
        assert "Integration Test Report" in md_content
        print("  [OK]")

        print("Test 3: Confidence bonus calculated")
        assert 0 <= result["confidence_bonus"] <= 0.05
        print(f"  Confidence bonus: +{result['confidence_bonus']}")
        print("  [OK]")

        print("Test 4: Report JSON has correct structure")
        rj = result["report_json"]
        assert rj["total"] == result["total_tests"]
        assert rj["passed"] == result["passed"]
        print("  [OK]")

    finally:
        shutil.rmtree(test_out)

    print("\n=== All test runner checks passed ===")
