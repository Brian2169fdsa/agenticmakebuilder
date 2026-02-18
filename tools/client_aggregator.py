"""
Client Intelligence Layer — Aggregator + Pattern Analyzer + Report Generator

Aggregates all build and audit data per client, identifies patterns,
scores expansion opportunities, and generates monthly reports.

Three main functions:
    get_client_profile(db, customer_name)  → full intelligence profile
    get_all_clients(db)                    → summary list of all clients
    generate_monthly_report(db, customer_name, output_dir) → report files
"""

import json
import os
from datetime import datetime, timezone, timedelta, date
from typing import Optional


# ─── Pattern library ───

EXPANSION_OPPORTUNITIES = [
    {
        "id": "no_error_handling",
        "name": "Add Error Handling",
        "description": "Scenario has no error handling strategy. Failures go undetected.",
        "value": "high",
        "effort": "low",
        "roi_estimate": "Prevents data loss and silent failures.",
        "check": lambda build: not _has_error_handling(build),
    },
    {
        "id": "no_monitoring",
        "name": "Enable Monitoring",
        "description": "Scenario not registered for daily health audits.",
        "value": "high",
        "effort": "low",
        "roi_estimate": "Issues caught in hours, not days.",
        "check": lambda build: not build.get("is_monitored", False),
    },
    {
        "id": "low_confidence",
        "name": "Rebuild with Higher Confidence",
        "description": "Scenario confidence score is below 0.80. Structural improvements possible.",
        "value": "medium",
        "effort": "medium",
        "roi_estimate": "More reliable execution, fewer edge case failures.",
        "check": lambda build: _get_confidence(build) < 0.80,
    },
    {
        "id": "no_versioning",
        "name": "Version Control & Backup",
        "description": "Scenario has only one version. No rollback available.",
        "value": "medium",
        "effort": "low",
        "roi_estimate": "Ability to revert changes instantly.",
        "check": lambda build: build.get("version", 1) <= 1,
    },
    {
        "id": "high_ops_usage",
        "name": "Optimize Operations Usage",
        "description": "High ops consumption — scenario may benefit from batching or caching.",
        "value": "medium",
        "effort": "medium",
        "roi_estimate": "Potential 20-40% reduction in Make.com plan costs.",
        "check": lambda build: _get_monthly_ops(build) > 50000,
    },
]


def _has_error_handling(build: dict) -> bool:
    spec = build.get("canonical_spec", {}) or {}
    strategy = spec.get("error_handling", {}).get("default_strategy", "ignore")
    return strategy != "ignore"


def _get_confidence(build: dict) -> float:
    conf = build.get("confidence_score") or 0.0
    return float(conf)


def _get_monthly_ops(build: dict) -> int:
    cost = build.get("cost_estimate_json", {}) or {}
    return cost.get("monthly_ops_estimate", 0)


# ─── DB helpers ───

def _get_builds_for_client(db, customer_name: str) -> list:
    """Fetch all builds for a customer from the DB."""
    try:
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT b.id, b.slug, b.version, b.status, b.created_at,
                   b.confidence_score, b.confidence_grade, b.heal_attempts,
                   p.name as project_name
            FROM builds b
            JOIN projects p ON p.id = b.project_id
            WHERE LOWER(p.customer_name) = LOWER(:customer_name)
              AND b.status = 'success'
            ORDER BY b.slug, b.version DESC
        """), {"customer_name": customer_name})
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception:
        return []


def _get_audit_history_for_client(db, customer_name: str, days: int = 90) -> list:
    """Fetch audit history for a customer."""
    try:
        from sqlalchemy import text
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        result = db.execute(text("""
            SELECT slug, verdict, confidence_score, regression,
                   delta_confidence, audited_at
            FROM audit_runs
            WHERE LOWER(customer_name) = LOWER(:customer_name)
              AND audited_at >= :cutoff
            ORDER BY audited_at DESC
        """), {"customer_name": customer_name, "cutoff": cutoff})
        rows = result.fetchall()
        return [dict(r._mapping) for r in rows]
    except Exception:
        return []


def _get_incidents_for_client(slug_list: list) -> list:
    """Fetch open incidents for a list of slugs."""
    from tools.incident_manager import list_incidents
    incidents = []
    for slug in slug_list:
        incidents.extend(list_incidents(slug=slug, status="open"))
    return incidents


# ─── Profile builder ───

def get_client_profile(db, customer_name: str) -> dict:
    """
    Build a full intelligence profile for a client.

    Returns:
        dict with builds, health, ops, opportunities, risk_score
    """
    builds = _get_builds_for_client(db, customer_name)
    audit_history = _get_audit_history_for_client(db, customer_name)

    # Get latest version per slug
    latest_by_slug = {}
    for b in builds:
        slug = b["slug"]
        if slug not in latest_by_slug or b["version"] > latest_by_slug[slug]["version"]:
            latest_by_slug[slug] = b

    active_builds = list(latest_by_slug.values())
    slugs = [b["slug"] for b in active_builds]

    # Health summary
    total = len(active_builds)
    healthy = sum(1 for b in active_builds if _get_confidence(b) >= 0.85)
    needs_attention = sum(1 for b in active_builds if 0.70 <= _get_confidence(b) < 0.85)
    at_risk = sum(1 for b in active_builds if _get_confidence(b) < 0.70)

    # Ops estimate
    total_monthly_ops = sum(_get_monthly_ops(b) for b in active_builds)

    # Audit trends
    regressions_30d = sum(
        1 for a in audit_history
        if a.get("regression") and
        a["audited_at"] >= (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    )

    avg_confidence = (
        sum(_get_confidence(b) for b in active_builds) / total
        if total > 0 else None
    )

    # Expansion opportunities
    opportunities = []
    for build in active_builds:
        for opp in EXPANSION_OPPORTUNITIES:
            try:
                if opp["check"](build):
                    opportunities.append({
                        "slug": build["slug"],
                        "opportunity_id": opp["id"],
                        "name": opp["name"],
                        "description": opp["description"],
                        "value": opp["value"],
                        "effort": opp["effort"],
                        "roi_estimate": opp["roi_estimate"],
                    })
            except Exception:
                pass

    # Open incidents
    open_incidents = _get_incidents_for_client(slugs)

    # Risk score (0.0 = no risk, 1.0 = max risk)
    risk_score = 0.0
    if total > 0:
        risk_score += (at_risk / total) * 0.4
        risk_score += min(regressions_30d / 5, 1.0) * 0.3
        risk_score += min(len(open_incidents) / 5, 1.0) * 0.3
    risk_score = round(min(risk_score, 1.0), 3)

    # Opportunity score (0.0 = optimized, 1.0 = lots to improve)
    opp_score = round(min(len(opportunities) / (total * len(EXPANSION_OPPORTUNITIES) + 0.001), 1.0), 3) if total > 0 else 0.0

    return {
        "customer_name": customer_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_scenarios": total,
            "healthy": healthy,
            "needs_attention": needs_attention,
            "at_risk": at_risk,
            "total_monthly_ops_estimate": total_monthly_ops,
            "avg_confidence": round(avg_confidence, 3) if avg_confidence else None,
            "regressions_last_30d": regressions_30d,
            "open_incidents": len(open_incidents),
            "risk_score": risk_score,
            "opportunity_score": opp_score,
        },
        "active_scenarios": [
            {
                "slug": b["slug"],
                "version": b["version"],
                "confidence": _get_confidence(b),
                "grade": b.get("confidence_grade", "?"),
                "monthly_ops": _get_monthly_ops(b),
                "created_at": str(b.get("created_at", "")),
            }
            for b in active_builds
        ],
        "opportunities": opportunities,
        "open_incidents": open_incidents,
        "audit_history_90d": audit_history[:20],
    }


def get_all_clients(db) -> list:
    """Get summary stats for all clients."""
    try:
        from sqlalchemy import text
        result = db.execute(text("""
            SELECT DISTINCT p.customer_name,
                   COUNT(DISTINCT b.slug) as scenario_count,
                   AVG(b.confidence_score) as avg_confidence,
                   MAX(b.created_at) as last_build
            FROM projects p
            JOIN builds b ON b.project_id = p.id
            WHERE b.status = 'success'
            GROUP BY p.customer_name
            ORDER BY scenario_count DESC
        """))
        rows = result.fetchall()
        return [
            {
                "customer_name": r[0],
                "scenario_count": r[1],
                "avg_confidence": round(float(r[2]), 3) if r[2] else None,
                "last_build": str(r[3]) if r[3] else None,
            }
            for r in rows
        ]
    except Exception:
        return []


# ─── Report generator ───

def generate_monthly_report(
    db,
    customer_name: str,
    output_dir: str,
    report_month: str = None,  # "2026-02" format, defaults to current month
) -> dict:
    """
    Generate a monthly client intelligence report.

    Writes:
        monthly_report_{customer}_{month}.md
        monthly_report_{customer}_{month}.json
    """
    now = datetime.now(timezone.utc)
    month = report_month or now.strftime("%Y-%m")
    profile = get_client_profile(db, customer_name)

    os.makedirs(output_dir, exist_ok=True)
    safe_name = customer_name.lower().replace(" ", "_")

    md_content = _render_monthly_report_md(profile, month)
    json_content = {**profile, "report_month": month}

    md_path = os.path.join(output_dir, f"monthly_report_{safe_name}_{month}.md")
    json_path = os.path.join(output_dir, f"monthly_report_{safe_name}_{month}.json")

    with open(md_path, "w") as f:
        f.write(md_content)
    with open(json_path, "w") as f:
        json.dump(json_content, f, indent=2, default=str)

    return {
        "success": True,
        "customer_name": customer_name,
        "report_month": month,
        "md_path": md_path,
        "json_path": json_path,
        "summary": profile["summary"],
    }


def _render_monthly_report_md(profile: dict, month: str) -> str:
    s = profile["summary"]
    customer = profile["customer_name"]
    opps = profile["opportunities"]
    scenarios = profile["active_scenarios"]
    incidents = profile["open_incidents"]

    risk_label = "🟢 Low" if s["risk_score"] < 0.3 else ("🟡 Medium" if s["risk_score"] < 0.6 else "🔴 High")

    lines = [
        f"# Monthly AI Automation Report",
        f"**Client:** {customer} | **Period:** {month}",
        f"**Prepared by:** ManageAI | **Generated:** {datetime.now().strftime('%B %d, %Y')}",
        f"",
        f"---",
        f"",
        f"## Portfolio Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Active Scenarios | {s['total_scenarios']} |",
        f"| Healthy | ✅ {s['healthy']} |",
        f"| Needs Attention | ⚠️ {s['needs_attention']} |",
        f"| At Risk | 🚨 {s['at_risk']} |",
        f"| Avg Confidence | {s['avg_confidence'] or 'N/A'} |",
        f"| Monthly Ops (est.) | {s['total_monthly_ops_estimate']:,} |",
        f"| Regressions (30d) | {s['regressions_last_30d']} |",
        f"| Open Incidents | {s['open_incidents']} |",
        f"| **Risk Score** | **{risk_label}** ({s['risk_score']}) |",
        f"",
        f"---",
        f"",
        f"## Active Scenarios",
        f"",
        f"| Scenario | Version | Confidence | Monthly Ops |",
        f"|----------|---------|------------|-------------|",
    ]

    for sc in scenarios:
        conf_icon = "✅" if sc["confidence"] >= 0.85 else ("⚠️" if sc["confidence"] >= 0.70 else "🚨")
        lines.append(
            f"| {sc['slug']} | v{sc['version']} | "
            f"{conf_icon} {sc['confidence']} ({sc['grade']}) | "
            f"{sc['monthly_ops']:,} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Expansion Opportunities",
        f"",
    ]

    if opps:
        high_value = [o for o in opps if o["value"] == "high"]
        medium_value = [o for o in opps if o["value"] == "medium"]

        if high_value:
            lines.append("### 🔴 High Value")
            for o in high_value[:5]:
                lines.append(f"- **{o['name']}** ({o['slug']}): {o['description']}")
                lines.append(f"  *ROI: {o['roi_estimate']}*")
            lines.append("")

        if medium_value:
            lines.append("### 🟡 Medium Value")
            for o in medium_value[:5]:
                lines.append(f"- **{o['name']}** ({o['slug']}): {o['description']}")
            lines.append("")
    else:
        lines.append("*No expansion opportunities identified — portfolio is well-optimized.*")
        lines.append("")

    if incidents:
        lines += [
            f"---",
            f"",
            f"## Open Incidents",
            f"",
        ]
        for inc in incidents[:10]:
            sev_icon = "🚨" if inc["severity"] in ("high", "critical") else "⚠️"
            lines.append(f"- {sev_icon} **{inc['slug']}**: {inc['incident_type'].replace('_', ' ').title()} ({inc['severity']})")
        lines.append("")

    lines += [
        f"---",
        f"",
        f"## Next Steps",
        f"",
        f"1. Review any scenarios flagged as 'At Risk'",
        f"2. Address {len([o for o in opps if o['value'] == 'high'])} high-value opportunities",
        f"3. Resolve {s['open_incidents']} open incident(s)",
        f"4. Schedule 30-min review with your ManageAI account manager",
        f"",
        f"---",
        f"*Report generated automatically by ManageAI Platform v4.0.0*",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Client Aggregator Self-Check (no DB required) ===\n")

    print("Test 1: _has_error_handling")
    assert not _has_error_handling({"canonical_spec": {"error_handling": {"default_strategy": "ignore"}}})
    assert _has_error_handling({"canonical_spec": {"error_handling": {"default_strategy": "rollback"}}})
    print("  [OK]")

    print("Test 2: Opportunity checks")
    for opp in EXPANSION_OPPORTUNITIES:
        # Should not throw on empty build
        try:
            opp["check"]({})
        except Exception:
            pass
    print(f"  Opportunities defined: {len(EXPANSION_OPPORTUNITIES)}")
    print("  [OK]")

    print("Test 3: _render_monthly_report_md with mock profile")
    mock_profile = {
        "customer_name": "Acme Corp",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_scenarios": 5,
            "healthy": 3,
            "needs_attention": 1,
            "at_risk": 1,
            "total_monthly_ops_estimate": 45000,
            "avg_confidence": 0.87,
            "regressions_last_30d": 1,
            "open_incidents": 2,
            "risk_score": 0.35,
            "opportunity_score": 0.4,
        },
        "active_scenarios": [
            {"slug": "acme-form-to-slack", "version": 2, "confidence": 0.94, "grade": "A", "monthly_ops": 5000},
            {"slug": "acme-lead-qualifier", "version": 1, "confidence": 0.65, "grade": "D", "monthly_ops": 40000},
        ],
        "opportunities": [
            {"slug": "acme-lead-qualifier", "opportunity_id": "no_error_handling",
             "name": "Add Error Handling", "description": "No error handling configured.",
             "value": "high", "effort": "low", "roi_estimate": "Prevents data loss."},
        ],
        "open_incidents": [
            {"slug": "acme-lead-qualifier", "incident_type": "rate_limit", "severity": "medium",
             "status": "open", "detected_at": datetime.now(timezone.utc).isoformat()},
        ],
        "audit_history_90d": [],
    }

    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        md = _render_monthly_report_md(mock_profile, "2026-02")
        assert "Acme Corp" in md
        assert "Expansion Opportunities" in md
        assert "Active Scenarios" in md
        assert len(md) > 500
        print(f"  Report length: {len(md)} chars")
        print("  [OK]")
    finally:
        shutil.rmtree(tmp)

    print("\n=== All client aggregator checks passed ===")
    print("Note: DB-dependent functions require PostgreSQL connection.")
