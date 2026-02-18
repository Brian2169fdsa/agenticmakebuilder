# Workflow: Phase 4 — Self-Healing

## Objective

Deterministically repair structural errors in a Make.com blueprint JSON based on the validation report from Phase 3. Apply up to 2 repair passes without AI reasoning or network calls.

---

## Required Inputs

1. **Blueprint** — Make.com blueprint JSON with validation errors
2. **Module registry** — For credential and version defaults
3. **Max retries** — Default 2

---

## Steps

### Step 1: Initial Validation (Tool)

Invoke: `validate_make_export(blueprint, registry)`

Identify errors and classify as repairable or non-repairable.

### Step 2: Repair Pass (Tool)

Invoke: `self_heal_make_export(blueprint, registry, max_retries=2)`

For each repair pass:
1. Collect all errors from validation report
2. Filter to repairable rule IDs
3. Apply deterministic fixes (sorted by rule_id for consistency)
4. Re-validate
5. If still errors and retries remain, repeat

### Step 3: Evaluate Result (Agent)

**If healed (success=True):**
- Proceed to Phase 5 with repaired blueprint

**If not healed (success=False):**
- Review remaining errors
- If only non-repairable errors: STOP and report
- Non-repairable errors require upstream fixes (Phase 1 re-specification)

---

## Supported Repairs

| Rule ID | Error | Repair Action |
|---------|-------|---------------|
| MR-002 | Missing/invalid name | Set "Untitled Scenario" |
| MR-003 | Missing/invalid flow | Initialize empty array |
| MR-004 | Missing/invalid metadata | Create default metadata object |
| MF-001 | Non-integer module ID | Attempt int conversion |
| MF-002 | Duplicate module IDs | Reassign unique IDs (increment from max) |
| MF-003 | Trigger not first in flow | Move id=1 module to index 0 |
| MF-006 | Missing version field | Set from registry (default 1) |
| MM-001 | Missing metadata.designer | Add deterministic x,y positions |
| MM-002 | Missing parameters | Add empty dict |
| MM-004 | onerror not array | Wrap string/dict in array |
| MM-005 | Invalid onerror directive | Remove invalid entries |
| MT-001 | Router missing routes | Add empty routes array |
| MT-003 | Route missing flow | Add empty flow array |
| MC-003 | Missing __IMTCONN__ | Add credential from registry |
| MD-001 | Missing metadata.scenario | Add default scenario object |
| MD-002 | Missing maxErrors | Set default (3) |
| MD-003 | Missing designer.orphans | Add empty array |

### Non-Repairable Errors

| Rule ID | Error | Why Not Repairable |
|---------|-------|--------------------|
| MF-005 | Missing module type identifier | Requires context from canonical spec |
| MF-007 | Module not in registry | Unknown module cannot be fixed locally |
| MC-001 | Numeric __IMTCONN__ | Possible credential leak; needs investigation |
| MC-002 | Invalid __IMTCONN__ pattern | Correct value unknown |
| MT-002 | Router < 2 routes | Requires canonical spec to add routes |

---

## Tools Used

| Tool | File | Purpose |
|------|------|---------|
| self_heal_make_export | `/tools/self_heal_make_export.py` | Deterministic blueprint repair |
| validate_make_export | `/tools/validate_make_export.py` | Re-validation after each pass |

---

## Success Criteria

Phase 4 is complete when:
- [ ] All repairable errors fixed
- [ ] Repair log documents every action taken
- [ ] Final validation report has zero errors — OR only non-repairable errors remain
- [ ] Original blueprint not mutated (deep-copy safety)
- [ ] Max 2 repair passes respected

---

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Blueprint already valid | Returns immediately with 0 retries, empty repair log |
| Only non-repairable errors | Returns immediately with 0 retries, success=False |
| Repair introduces new errors | Second pass catches and fixes cascade errors |
| All errors fixed in pass 1 | Stops after 1 retry, does not use pass 2 |
| Empty blueprint ({}) | Repairs root structure (MR-002/003/004), may still fail on MR-005 |

---

## Stopping Conditions

- **STOP** after 2 repair passes regardless of remaining errors
- **STOP** immediately if no repairable errors exist
- **STOP** if blueprint is beyond repair (non-dict, etc.)

---

## What This Workflow Does NOT Do

- Does NOT use AI reasoning to determine repairs
- Does NOT modify the canonical spec (upstream changes require Phase 1 re-run)
- Does NOT make API calls to Make.com
- Does NOT add new modules or connections
