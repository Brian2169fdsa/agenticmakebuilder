"""
Demo Build Runner

Runs the full build_scenario_pipeline with two sample plans:
  1. Linear scenario: webhook → parse JSON → post to Slack
  2. Router scenario: webhook → router → [Slack alert, Google Sheets log]

Outputs artifacts to /output/<slug>/vN/ and prints results.

Usage:
    python3 demo_build.py
"""

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from tools.module_registry_loader import load_module_registry
from tools.build_scenario_pipeline import build_scenario_pipeline


def main():
    print("=" * 60)
    print("  Agentic Make Builder — Demo Build")
    print("=" * 60)
    print()

    registry = load_module_registry()

    # === Demo 1: Linear Scenario ===
    print("-" * 60)
    print("  Demo 1: Linear Scenario")
    print("  webhook → parse JSON → post to Slack")
    print("-" * 60)
    print()

    linear_plan = {
        "scenario_name": "Form Submission to Slack",
        "scenario_description": (
            "Receives form submission data via webhook, "
            "parses the JSON payload, and posts a notification "
            "to a Slack channel."
        ),
        "trigger": {
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive Form Submission",
            "parameters": {"hook": "__WEBHOOK_ID__"}
        },
        "steps": [
            {
                "module": "json:ParseJSON",
                "label": "Parse Form Data",
                "mapper": {"json": "{{1.body}}"}
            },
            {
                "module": "slack:PostMessage",
                "label": "Notify Team on Slack",
                "mapper": {
                    "channel": "#form-submissions",
                    "text": "New submission from {{2.name}} ({{2.email}})"
                },
                "onerror": "ignore"
            }
        ],
        "error_handling": {
            "default_strategy": "ignore",
            "max_errors": 3
        },
        "agent_notes": [
            "Assumed webhook payload contains 'name' and 'email' fields",
            "Slack channel #form-submissions must exist"
        ]
    }

    r1 = build_scenario_pipeline(
        linear_plan, registry,
        "When someone submits a form, parse the data and notify the team on Slack"
    )
    _print_result(r1)

    # === Demo 2: Router Scenario ===
    print()
    print("-" * 60)
    print("  Demo 2: Router Scenario")
    print("  webhook → router → [Slack alert, Google Sheets log]")
    print("-" * 60)
    print()

    router_plan = {
        "scenario_name": "Priority Alert Router",
        "scenario_description": (
            "Receives events via webhook, routes high-priority events "
            "to a Slack alert channel, and logs all events to a Google Sheet."
        ),
        "trigger": {
            "type": "webhook",
            "module": "gateway:CustomWebHook",
            "label": "Receive Event",
            "parameters": {"hook": "__WEBHOOK_ID__"}
        },
        "steps": [
            {
                "module": "json:ParseJSON",
                "label": "Parse Event Payload",
                "mapper": {"json": "{{1.body}}"}
            },
            {
                "module": "builtin:BasicRouter",
                "label": "Route by Priority"
            },
            {
                "module": "slack:PostMessage",
                "label": "Send Urgent Alert",
                "mapper": {
                    "channel": "#urgent-alerts",
                    "text": "URGENT: {{2.title}} — {{2.description}}"
                },
                "onerror": "break"
            },
            {
                "module": "google-sheets:addRow",
                "label": "Log to Event Tracker",
                "parameters": {
                    "spreadsheetId": "__SPREADSHEET_ID__",
                    "sheetId": "__SHEET_ID__"
                },
                "mapper": {
                    "values": ["{{2.timestamp}}", "{{2.priority}}", "{{2.title}}"]
                }
            }
        ],
        "connections": [
            {"from": "trigger", "to": 0},
            {"from": 0, "to": 1},
            {
                "from": 1, "to": 2,
                "filter": {
                    "name": "High Priority",
                    "conditions": [[{
                        "a": "{{2.priority}}",
                        "b": "high",
                        "o": "text:equal"
                    }]]
                },
                "label": "High Priority"
            },
            {"from": 1, "to": 3, "label": "All Events"}
        ],
        "error_handling": {
            "default_strategy": "ignore",
            "max_errors": 5
        },
        "agent_notes": [
            "Router fallback route logs all events regardless of priority",
            "Slack alert only fires for priority=high"
        ]
    }

    r2 = build_scenario_pipeline(
        router_plan, registry,
        "Route incoming events: urgent ones go to Slack, all get logged to Google Sheets"
    )
    _print_result(r2)

    # === Summary ===
    print()
    print("=" * 60)
    print("  Demo Complete")
    print("=" * 60)
    print()

    results = [("Linear", r1), ("Router", r2)]
    for name, r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"  [{status}] {name}: {r['slug']} v{r.get('version', '?')} → {r.get('output_path', 'N/A')}")

    print()


def _print_result(result):
    """Print a pipeline result in a readable format."""
    if result["success"]:
        print(f"  Status:     SUCCESS")
        print(f"  Slug:       {result['slug']}")
        print(f"  Version:    v{result['version']}")
        print(f"  Output:     {result['output_path']}")
        print(f"  Confidence: {result['confidence']['score']} (Grade {result['confidence']['grade']})")

        cv = result["canonical_validation"]
        print(f"  Spec:       PASS ({cv['checks_passed']}/{cv['checks_run']})")

        ev = result["export_validation"]
        print(f"  Export:     PASS ({ev['checks_passed']}/{ev['checks_run']})")

        if result.get("heal_result"):
            h = result["heal_result"]
            print(f"  Self-Heal:  {h['repairs']} repair(s) in {h['retries_used']} pass(es)")

        # List artifacts
        if result["output_path"] and os.path.isdir(result["output_path"]):
            files = sorted(os.listdir(result["output_path"]))
            print(f"  Artifacts:  {len(files)} files")
            for f in files:
                size = os.path.getsize(os.path.join(result["output_path"], f))
                print(f"              - {f} ({size:,} bytes)")
    else:
        print(f"  Status:     FAILED")
        print(f"  Slug:       {result.get('slug', 'N/A')}")
        print(f"  Reason:     {result.get('failure_reason', 'Unknown')}")


if __name__ == "__main__":
    main()
