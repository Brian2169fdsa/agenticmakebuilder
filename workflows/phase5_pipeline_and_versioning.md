# Workflow: Phase 5 — Pipeline Orchestration & Versioning

## Objective

Orchestrate the full scenario build lifecycle through a single deterministic entrypoint, produce versioned artifacts, and generate customer-facing delivery outputs including timeline, cost estimates, and a delivery summary.

---

## Required Inputs

1. **Plan** — Structured plan dict from agent reasoning
2. **Module registry** — Loaded via `module_registry_loader`
3. **Original request** — User's natural language description
4. **Base output dir** — Optional override (defaults to `/output/`)

---

## Steps

### Step 1: Run Full Pipeline (Tool)

Invoke: `build_scenario_pipeline(plan, registry, original_request)`

The pipeline executes Phases 1–4 in sequence:

| Stage | Tool | Action |
|-------|------|--------|
| 1 | `normalize_to_canonical_spec` | Plan → canonical spec |
| 2 | `validate_canonical_spec` | 35+ rule validation |
| 3 | `compute_confidence` | Score 0.0–1.0 |
| 4 | `generate_make_export` | Spec → Make.com blueprint |
| 5 | `validate_make_export` | 30+ structural checks |
| 6 | `self_heal_make_export` | Repair if needed (max 2 retries) |
| 7 | `save_versioned_spec` | Write core artifacts |
| 8 | `timeline_estimator` | Heuristic build timeline |
| 9 | `cost_estimator` | Config-driven cost estimate |
| 10 | `delivery_packager` | Customer delivery summary + pack |

### Step 2: Review Pipeline Result (Agent)

**If success=True:**
- All artifacts written to `/output/<slug>/vN/`
- Delivery summary available for customer presentation
- Build complete

**If success=False:**
- Check `failure_reason` for root cause
- If canonical validation failure: revise plan and retry
- If export validation failure: check for non-repairable errors
- Do NOT proceed with delivery if build failed

### Step 3: Version Management

The pipeline automatically handles versioning:
- **New scenario** → v1
- **Same slug rebuilt** → auto-increment (v2, v3, ...)
- **Global index** (`/output/index.json`) tracks all scenarios
- **Per-scenario index** (`/output/<slug>/index.json`) tracks version history

---

## Artifacts Written

```
/output/<slug>/vN/
  canonical_spec.json             # Phase 1: Canonical workflow specification
  make_export.json                # Phase 2: Make.com blueprint (importable)
  validation_report.json          # Phase 1: Canonical spec validation
  export_validation_report.json   # Phase 3: Blueprint validation
  confidence.json                 # Confidence score + factors
  build_log.md                    # Human-readable build log
  timeline.json                   # Heuristic build timeline estimate
  cost_estimate.json              # Config-driven cost estimate
  customer_delivery_summary.md    # Customer-facing delivery document
  delivery_pack.json              # Machine-readable delivery package
```

---

## Tools Used

| Tool | File | Purpose |
|------|------|---------|
| build_scenario_pipeline | `/tools/build_scenario_pipeline.py` | Full lifecycle orchestrator |
| normalize_to_canonical_spec | `/tools/normalize_to_canonical_spec.py` | Plan → canonical spec |
| validate_canonical_spec | `/tools/validate_canonical_spec.py` | Canonical spec validation |
| confidence_scorer | `/tools/confidence_scorer.py` | Build confidence scoring |
| generate_make_export | `/tools/generate_make_export.py` | Spec → Make.com blueprint |
| validate_make_export | `/tools/validate_make_export.py` | Blueprint validation |
| self_heal_make_export | `/tools/self_heal_make_export.py` | Deterministic repair |
| spec_version_manager | `/tools/spec_version_manager.py` | Versioned artifact storage |
| timeline_estimator | `/tools/timeline_estimator.py` | Heuristic timeline estimate |
| cost_estimator | `/tools/cost_estimator.py` | Config-driven cost estimate |
| delivery_packager | `/tools/delivery_packager.py` | Customer delivery outputs |

---

## Success Criteria

Phase 5 is complete when:
- [ ] Pipeline runs end-to-end without errors
- [ ] All 10 artifacts written to `/output/<slug>/vN/`
- [ ] Version auto-increments for repeated builds
- [ ] Global and per-scenario indexes updated
- [ ] Delivery summary is customer-presentable
- [ ] Timeline and cost estimates are reasonable heuristics
- [ ] Failed builds produce clear failure summaries

---

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Canonical validation failure | Pipeline stops; no artifacts written; failure summary returned |
| Non-healable export errors | Pipeline stops after Phase 4; failure summary returned |
| Same slug rebuilt | Version auto-increments; previous versions preserved |
| Exception during pipeline | Caught and reported in failure_reason; no partial artifacts |
| Empty steps (trigger only) | Builds successfully with minimal scenario |

---

## Stopping Conditions

- **STOP** if canonical spec validation fails and confidence < 0.70
- **STOP** if Make export has non-healable errors after 2 repair passes
- **STOP** on any unhandled exception

---

## What This Workflow Does NOT Do

- Does NOT make API calls to Make.com
- Does NOT import the blueprint into Make.com
- Does NOT handle real credentials or API keys
- Does NOT bypass any validation step
