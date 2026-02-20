#!/usr/bin/env python3
"""
mab — ManageAI Agentic Builder CLI.

A command-line interface for interacting with the agenticmakebuilder API.

Usage:
    mab health              Show platform health
    mab plan                Run a plan interactively
    mab verify <id>         Verify a project
    mab deploy <id>         Deploy a project
    mab status <scenario>   Check deployment status
    mab monitor             Monitor all deployments
    mab jobs                List jobs
    mab jobs enqueue        Enqueue a job
    mab jobs cancel <id>    Cancel a job
    mab keys create         Create an API key
    mab keys list           List API keys
    mab keys revoke <id>    Revoke an API key
    mab tenants             List tenants
    mab events              List event subscriptions
    mab events subscribe    Subscribe to events
    mab learn outcome       Record a learning outcome
    mab learn insights      Get learning insights
    mab admin status        System status
    mab admin reindex       Reindex embeddings

Environment variables:
    MAB_BASE_URL    API base URL (default: http://localhost:8000)
    MAB_API_KEY     API key for authenticated requests
"""

import argparse
import json
import os
import sys

import httpx

BASE_URL = os.getenv("MAB_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("MAB_API_KEY", "")
OUTPUT_JSON = False


def api_get(path, params=None):
    """GET request to the API."""
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    try:
        resp = httpx.get(f"{BASE_URL}{path}", params=params, headers=headers, timeout=30)
        return resp
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {BASE_URL}")
        sys.exit(1)


def api_post(path, body=None):
    """POST request to the API."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    try:
        resp = httpx.post(f"{BASE_URL}{path}", json=body or {}, headers=headers, timeout=60)
        return resp
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {BASE_URL}")
        sys.exit(1)


def api_delete(path):
    """DELETE request to the API."""
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    try:
        resp = httpx.delete(f"{BASE_URL}{path}", headers=headers, timeout=30)
        return resp
    except httpx.ConnectError:
        print(f"Error: Cannot connect to {BASE_URL}")
        sys.exit(1)


def handle_error(resp):
    """Print error and exit if response is not 2xx."""
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        print(f"Error {resp.status_code}: {detail}")
        sys.exit(1)


def print_json(data):
    """Print formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def print_table(headers, rows):
    """Print a formatted ASCII table."""
    if not rows:
        print("(no data)")
        return
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    fmt = " | ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


# ── Commands ─────────────────────────────────────────────────────


def cmd_health(args):
    """Show platform health."""
    r = api_get("/health/full")
    handle_error(r)
    data = r.json()
    if OUTPUT_JSON:
        print_json(data)
        return
    print(f"Status: {data.get('status', 'unknown')}")
    checks = data.get("checks", {})
    if "database" in checks:
        print(f"Database: {checks['database'].get('status', '?')}")
    tables = checks.get("tables", {})
    if tables:
        rows = [[name, info.get("status", "?"), info.get("row_count", "?")] for name, info in tables.items()]
        print_table(["Table", "Status", "Rows"], rows)


def cmd_plan(args):
    """Run a plan."""
    request_text = input("Describe what to build: ").strip()
    customer = input("Customer name [Customer]: ").strip() or "Customer"
    r = api_post("/plan", {"original_request": request_text, "customer_name": customer})
    handle_error(r)
    data = r.json()
    if OUTPUT_JSON:
        print_json(data)
    else:
        print(f"Success: {data.get('success', '?')}")
        conf = data.get("confidence", {})
        if conf:
            print(f"Confidence: {conf.get('score', '?')} (Grade {conf.get('grade', '?')})")


def cmd_verify(args):
    """Verify a project."""
    r = api_post("/verify", {"project_id": args.id, "name": "CLI verify", "type": "project"})
    handle_error(r)
    print_json(r.json()) if OUTPUT_JSON else print(f"Score: {r.json().get('confidence_score', '?')}")


def cmd_deploy(args):
    """Deploy a project."""
    r = api_post("/deploy/makecom", {"project_id": args.id, "blueprint": {}, "scenario_name": f"CLI-{args.id[:8]}"})
    handle_error(r)
    print_json(r.json())


def cmd_status(args):
    """Check deployment status."""
    r = api_get("/deploy/status", {"project_id": args.scenario})
    handle_error(r)
    print_json(r.json())


def cmd_monitor(args):
    """Monitor all deployments."""
    r = api_get("/deploy/monitor")
    handle_error(r)
    print_json(r.json())


def cmd_jobs(args):
    """List jobs."""
    r = api_get("/jobs/list")
    handle_error(r)
    data = r.json()
    if OUTPUT_JSON:
        print_json(data)
        return
    rows = [[j["id"][:8], j["job_type"], j["status"], j.get("priority", ""), j.get("created_at", "")[:19]] for j in data.get("jobs", [])]
    print_table(["ID", "Type", "Status", "Priority", "Created"], rows)
    print(f"\nStats: {data.get('stats', {})}")


def cmd_jobs_enqueue(args):
    """Enqueue a job."""
    jtype = input("Job type (plan/verify/deploy/embed/reindex): ").strip()
    r = api_post("/jobs/enqueue", {"job_type": jtype, "payload": {}, "priority": 5})
    handle_error(r)
    print(f"Job enqueued: {r.json().get('job_id', '?')}")


def cmd_jobs_cancel(args):
    """Cancel a job."""
    r = api_delete(f"/jobs/{args.id}/cancel")
    handle_error(r)
    print(f"Cancelled: {r.json().get('cancelled', False)}")


def cmd_keys_create(args):
    """Create an API key."""
    name = input("Key name: ").strip() or "cli-key"
    r = api_post("/auth/keys", {"name": name})
    handle_error(r)
    data = r.json()
    print(f"Key ID: {data.get('key_id')}")
    print(f"Raw Key: {data.get('raw_key')}")
    print("WARNING: Save this key now. It cannot be retrieved again.")


def cmd_keys_list(args):
    """List API keys."""
    r = api_get("/auth/keys")
    handle_error(r)
    data = r.json()
    if OUTPUT_JSON:
        print_json(data)
        return
    rows = [[k["key_prefix"], k["name"], k["tenant_id"], k.get("revoked", False), (k.get("last_used_at") or "never")[:19]] for k in data.get("keys", [])]
    print_table(["Prefix", "Name", "Tenant", "Revoked", "Last Used"], rows)


def cmd_keys_revoke(args):
    """Revoke an API key."""
    r = api_delete(f"/auth/keys/{args.id}")
    handle_error(r)
    print(f"Revoked: {r.json().get('revoked', False)}")


def cmd_tenants(args):
    """List tenants."""
    r = api_get("/tenants")
    handle_error(r)
    data = r.json()
    if OUTPUT_JSON:
        print_json(data)
        return
    rows = [[t["id"], t["name"], t["plan"], t.get("active", True)] for t in data.get("tenants", [])]
    print_table(["ID", "Name", "Plan", "Active"], rows)


def cmd_events(args):
    """List event subscriptions."""
    r = api_get("/events/subscriptions")
    handle_error(r)
    data = r.json()
    if OUTPUT_JSON:
        print_json(data)
        return
    rows = [[s["id"][:8], s["event_type"], s["target_url"][:40], s["active"]] for s in data.get("subscriptions", [])]
    print_table(["ID", "Event", "URL", "Active"], rows)


def cmd_events_subscribe(args):
    """Subscribe to events."""
    etype = input("Event type: ").strip()
    url = input("Target URL: ").strip()
    r = api_post("/events/subscribe", {"event_type": etype, "target_url": url})
    handle_error(r)
    print(f"Subscribed: {r.json().get('subscription_id')}")


def cmd_learn_outcome(args):
    """Record a learning outcome."""
    pid = input("Project ID: ").strip()
    sid = input("Scenario ID: ").strip()
    outcome = input("Outcome (success/failure/partial): ").strip()
    r = api_post("/learn/outcome", {"project_id": pid, "scenario_id": sid, "outcome": outcome})
    handle_error(r)
    print(f"Learned: {r.json().get('learned', False)}")


def cmd_learn_insights(args):
    """Get learning insights."""
    desc = input("Description: ").strip()
    r = api_get("/learn/insights", {"description": desc})
    handle_error(r)
    print_json(r.json())


def cmd_admin_status(args):
    """System status."""
    r = api_get("/admin/system-status")
    handle_error(r)
    print_json(r.json())


def cmd_admin_reindex(args):
    """Reindex embeddings."""
    r = api_post("/admin/reindex")
    handle_error(r)
    print_json(r.json())


def main():
    global BASE_URL, API_KEY, OUTPUT_JSON

    parser = argparse.ArgumentParser(prog="mab", description="ManageAI Agentic Builder CLI")
    parser.add_argument("--url", default=os.getenv("MAB_BASE_URL", "http://localhost:8000"), help="API base URL")
    parser.add_argument("--key", default=os.getenv("MAB_API_KEY", ""), help="API key")
    parser.add_argument("--json", action="store_true", dest="json_output", help="Output raw JSON")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("health", help="Platform health")
    sub.add_parser("plan", help="Run a plan")
    p = sub.add_parser("verify", help="Verify a project")
    p.add_argument("id", help="Project ID")
    p = sub.add_parser("deploy", help="Deploy a project")
    p.add_argument("id", help="Project ID")
    p = sub.add_parser("status", help="Deployment status")
    p.add_argument("scenario", help="Scenario ID")
    sub.add_parser("monitor", help="Monitor deployments")

    jobs_p = sub.add_parser("jobs", help="Job queue commands")
    jobs_sub = jobs_p.add_subparsers(dest="jobs_cmd")
    jobs_sub.add_parser("list", help="List jobs")
    jobs_sub.add_parser("enqueue", help="Enqueue a job")
    jp = jobs_sub.add_parser("cancel", help="Cancel a job")
    jp.add_argument("id", help="Job ID")

    keys_p = sub.add_parser("keys", help="API key commands")
    keys_sub = keys_p.add_subparsers(dest="keys_cmd")
    keys_sub.add_parser("create", help="Create a key")
    keys_sub.add_parser("list", help="List keys")
    kp = keys_sub.add_parser("revoke", help="Revoke a key")
    kp.add_argument("id", help="Key ID")

    sub.add_parser("tenants", help="List tenants")

    events_p = sub.add_parser("events", help="Event bus commands")
    events_sub = events_p.add_subparsers(dest="events_cmd")
    events_sub.add_parser("list", help="List subscriptions")
    events_sub.add_parser("subscribe", help="Subscribe to events")

    learn_p = sub.add_parser("learn", help="Learning commands")
    learn_sub = learn_p.add_subparsers(dest="learn_cmd")
    learn_sub.add_parser("outcome", help="Record outcome")
    learn_sub.add_parser("insights", help="Get insights")

    admin_p = sub.add_parser("admin", help="Admin commands")
    admin_sub = admin_p.add_subparsers(dest="admin_cmd")
    admin_sub.add_parser("status", help="System status")
    admin_sub.add_parser("reindex", help="Reindex embeddings")

    args = parser.parse_args()
    BASE_URL = args.url
    API_KEY = args.key
    OUTPUT_JSON = args.json_output

    cmd_map = {
        "health": cmd_health,
        "plan": cmd_plan,
        "verify": cmd_verify,
        "deploy": cmd_deploy,
        "status": cmd_status,
        "monitor": cmd_monitor,
        "tenants": cmd_tenants,
    }

    if args.command in cmd_map:
        cmd_map[args.command](args)
    elif args.command == "jobs":
        if args.jobs_cmd == "enqueue":
            cmd_jobs_enqueue(args)
        elif args.jobs_cmd == "cancel":
            cmd_jobs_cancel(args)
        else:
            cmd_jobs(args)
    elif args.command == "keys":
        if args.keys_cmd == "create":
            cmd_keys_create(args)
        elif args.keys_cmd == "revoke":
            cmd_keys_revoke(args)
        else:
            cmd_keys_list(args)
    elif args.command == "events":
        if args.events_cmd == "subscribe":
            cmd_events_subscribe(args)
        else:
            cmd_events(args)
    elif args.command == "learn":
        if args.learn_cmd == "outcome":
            cmd_learn_outcome(args)
        elif args.learn_cmd == "insights":
            cmd_learn_insights(args)
        else:
            parser.print_help()
    elif args.command == "admin":
        if args.admin_cmd == "status":
            cmd_admin_status(args)
        elif args.admin_cmd == "reindex":
            cmd_admin_reindex(args)
        else:
            parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
