# Workflow: Phase 1 — Canonical Workflow Spec

## Objective

Transform a user's natural language request into a validated, versioned canonical workflow specification that captures everything needed to generate a Make.com scenario.

---

## Required Inputs

1. **User request** — Natural language description of the desired Make.com scenario
2. **Clarifications** — Answers to agent questions about trigger, integrations, branching, error handling, and expected outputs

---

## Steps

### Step 1: Requirement Gathering (Agent)

The agent must clarify:
- **Trigger type** — webhook, schedule, or app event?
- **Integrations** — which apps/services are involved?
- **Branching** — any conditional routing needed?
- **Error handling** — what should happen when a module fails?
- **Expected outputs** — what does "done" look like?

### Step 2: Tool Selection (Agent)

1. Load module registry via `module_registry_loader.load_module_registry()`
2. Verify all required modules exist in the curated registry
3. If a required module is NOT in the registry, STOP and inform the user

### Step 3: Spec Generation (Tool)

Invoke: `normalize_to_canonical_spec(plan, registry, original_request)`

The plan dict must include:
- Trigger definition (type, module, label, parameters)
- Module list (in execution order)
- Connection map (which modules connect to which)
- Error handling strategy
- Any data mapping requirements

### Step 4: Validation (Tool)

Invoke: `validate_canonical_spec(spec, registry)`

**Success criteria:**
- Zero errors in validation report
- All 7 rule categories pass (SC, MI, CI, RR, CR, DM, SS)

**If validation fails:**
- Identify failed rules from the report
- Determine if the failure is in spec structure (fixable) or user requirements (needs clarification)
- If fixable: repair and re-validate (max 2 retries)
- If needs clarification: return to Step 1

### Step 5: Confidence Scoring (Tool)

Invoke: `compute_confidence(validation_report, agent_notes, retry_count)`

**Minimum acceptable score:** 0.70 (Grade C or higher)

If score < 0.70:
- Review agent_notes for excessive assumptions
- Consider asking user for more clarity
- Do NOT proceed to Phase 2

### Step 6: Version + Store (Tool)

Invoke: `save_versioned_spec(slug, spec, report, confidence)`

This creates:
```
/output/<slug>/vN/
  canonical_spec.json
  validation_report.json
  confidence.json
  build_log.md
```

---

## Tools Used

| Tool | File | Purpose |
|------|------|---------|
| module_registry_loader | `/tools/module_registry_loader.py` | Load curated module registry |
| normalize_to_canonical_spec | `/tools/normalize_to_canonical_spec.py` | Generate canonical spec from plan |
| validate_canonical_spec | `/tools/validate_canonical_spec.py` | Run 35+ validation rules |
| graph_integrity_check | `/tools/graph_integrity_check.py` | DAG/cycle/orphan analysis |
| data_mapping_extractor | `/tools/data_mapping_extractor.py` | Validate data references |
| confidence_scorer | `/tools/confidence_scorer.py` | Compute build confidence |
| spec_version_manager | `/tools/spec_version_manager.py` | Versioned artifact storage |

---

## Success Criteria

Phase 1 is complete when:
- [ ] Canonical spec passes all validation rules (zero errors)
- [ ] Confidence score >= 0.70
- [ ] Artifacts versioned in `/output/<slug>/vN/`
- [ ] Build log generated

---

## Edge Cases

| Edge Case | Handling |
|-----------|----------|
| User requests unsupported module | STOP. Inform user. Do not hallucinate module types. |
| Ambiguous trigger type | Ask user to clarify before proceeding. |
| Circular dependency in flow | Validation catches via CI-005. Repair by removing the back-edge. |
| Orphan module (not connected) | Validation catches via CI-003. Repair by adding missing connection. |
| Missing credential placeholder | Validation catches via MI-008. Repair by adding placeholder from registry. |
| Data mapping references nonexistent module | Validation catches via DM-001. Repair by fixing the reference. |
| Forward reference (module references a downstream module) | Validation catches via DM-002. This requires restructuring the flow. |

---

## Stopping Conditions

- **STOP** if validation fails after 2 repair attempts
- **STOP** if confidence score < 0.70 after all retries
- **STOP** if user request requires modules not in the curated registry
- **STOP** if user cannot clarify ambiguous requirements after being asked

---

## What This Workflow Does NOT Do

- Does NOT generate Make.com export JSON (that is Phase 2)
- Does NOT make API calls to Make.com
- Does NOT handle real credentials
- Does NOT bypass validation under any circumstances
