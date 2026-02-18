# Workflow: Phase 2 — Make.com Export Generation

## Objective

Convert a validated canonical workflow specification into a Make.com blueprint JSON that can be imported via Make.com's blueprint import feature.

---

## Required Inputs

1. **Canonical spec** — Validated spec from Phase 1 (zero validation errors)
2. **Module registry** — Loaded via `module_registry_loader`

---

## Steps

### Step 1: Receive Validated Spec (Agent)

The agent confirms:
- Phase 1 canonical spec passed validation (zero errors)
- Confidence score >= 0.70
- All modules in the spec exist in the curated registry

### Step 2: Generate Blueprint (Tool)

Invoke: `generate_make_export(spec)`

Key conversions performed:
- **Explicit connections** → implicit flow ordering (Make.com's native format)
- **Router connections** → nested `routes[].flow[]` arrays
- **Connection filters** → filters placed on the receiving module
- **Credential placeholders** → `parameters.__IMTCONN__`
- **onerror string** → `onerror` array `[{"directive": str}]`
- **Designer positions** → deterministic x,y coordinates (X_SPACING=300, Y_SPACING=200)

### Step 3: Proceed to Validation (Agent)

Pass the generated blueprint to Phase 3 for structural validation.

---

## Tools Used

| Tool | File | Purpose |
|------|------|---------|
| generate_make_export | `/tools/generate_make_export.py` | Canonical spec → Make.com blueprint |
| graph_integrity_check | `/tools/graph_integrity_check.py` | Used internally for flow ordering |

---

## Success Criteria

Phase 2 is complete when:
- [ ] Blueprint JSON generated with `name`, `flow[]`, and `metadata`
- [ ] Trigger is first item in `flow[]` with `id: 1`
- [ ] Router modules contain nested `routes` with sub-flows
- [ ] Filters placed on receiving modules (not connections)
- [ ] Credentials mapped to `__IMTCONN__` in parameters
- [ ] Blueprint is valid JSON and serializable

---

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Trigger-only scenario (no modules) | Generates flow with single trigger module |
| Router with filtered and unfiltered routes | Filter placed on first module of filtered route; unfiltered route has no filter |
| Nested routers | Recursively builds routes within routes |
| Module with onerror "resume" + resume_value | Includes `resume` key in onerror entry |
| Module without credential | No `__IMTCONN__` key in parameters |

---

## Stopping Conditions

- **STOP** if canonical spec is not provided or is malformed
- **STOP** if canonical spec has not passed Phase 1 validation

---

## What This Workflow Does NOT Do

- Does NOT validate the generated blueprint (that is Phase 3)
- Does NOT repair structural issues (that is Phase 4)
- Does NOT make API calls to Make.com
- Does NOT handle real credentials
