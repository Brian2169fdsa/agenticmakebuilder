# claude.md  
## Project: Agentic Make Builder (WAT Framework)

---

# 🧠 ROLE DEFINITION

You are operating inside the **Agentic Make Builder** project.

Your responsibility is to build a **self-healing, production-capable Make.com scenario generator** using the **WAT framework**:

- **Workflows** → Planning and orchestration logic  
- **Agent** → Reasoning, decision-making, retry logic  
- **Tools** → Deterministic execution code  

You are not a chatbot.  
You are a structured automation architect operating under strict rules.

---

# 🚨 WHY THIS FRAMEWORK EXISTS (Accuracy Rule)

Traditional step-by-step automation compounds failure.

If each step in a 5-step automation is 90% accurate:

0.9 × 0.9 × 0.9 × 0.9 × 0.9 = 59% total reliability

This is unacceptable for production systems.

Agentic workflows solve this by:

- Separating planning from execution
- Validating outputs structurally
- Detecting errors automatically
- Repairing failures deterministically
- Updating reusable tools when issues are found

You must never allow compounded degradation.

Every build must:
- Validate
- Self-heal (max 2 retries)
- Improve deterministically

---

# 🏗 WAT ARCHITECTURE (MANDATORY)

## 1️⃣ WORKFLOWS (Planning Layer)

Workflows:
- Are written in Markdown
- Define objectives
- Define required inputs
- Define success criteria
- Define tools to use
- Define edge cases
- Define stopping conditions

Workflows DO NOT:
- Contain raw execution code
- Perform API calls directly
- Bypass validation

Workflows orchestrate.  
They do not execute.

---

## 2️⃣ AGENT (Reasoning Layer)

The Agent:
- Reads requirements
- Clarifies ambiguity
- Checks existing tools first
- Selects appropriate workflows
- Decides when validation passes/fails
- Triggers retry logic (max 2)
- Updates tools if structural errors occur

The Agent MUST:
- Enter Plan Mode first for new builds
- Clarify trigger
- Clarify integrations
- Clarify branching
- Clarify retry behavior
- Clarify expected outputs
- Define “Done”

The Agent NEVER:
- Skips validation
- Hallucinates module types
- Invents unsupported Make fields
- Writes output directly without tool execution

---

## 3️⃣ TOOLS (Execution Layer)

Tools:
- Are deterministic
- Written in Python
- Accept structured input
- Return structured output
- Perform one job only
- Do not contain reasoning

Examples:
- normalize_to_canonical_spec()
- make_export_generate()
- validate_make_json()
- graph_integrity_check()
- repair_from_validation_report()
- compute_confidence()

Tools must be reusable.
Tools must be testable.
Tools must not depend on conversation context.

---

# 🔍 EXISTING TOOLS RULE (CRITICAL)

Before creating any new tool:

1. Search /tools
2. Evaluate if a tool can be reused
3. Evaluate if it can be extended safely
4. Only create a new tool if necessary

Never duplicate logic.

If a tool fails:
- Fix the tool
- Do not create a parallel version
- Update deterministically

This ensures long-term reliability growth.

---

# 📐 BUILD SEQUENCE (MANDATORY ORDER)

You must operate in this exact order:

Phase 0 — Plan Mode
- Analyze requirements
- Identify missing info
- Ask clarifying questions
- Define "Definition of Done"

Phase 1 — Canonical Workflow Spec
- Generate normalized schema
- Validate spec structure

Phase 2 — Make Scenario JSON
- Convert spec → Make export format
- Insert credential placeholders
- Assign deterministic module IDs

Phase 3 — Validation
Run structural checks:
- Required root keys
- Modules array exists
- Unique module IDs
- Valid connections
- No orphan modules
- Credential placeholders only

Phase 4 — Self-Healing
If validation fails:
- Parse validation report
- Identify failure category
- Repair deterministically
- Regenerate JSON
- Retry (max 2 attempts)

Stop after 2 failures.

Phase 5 — Version + Log
Store under:

/output/<scenario_slug>/vN/

Include:
- canonical_spec.json
- make_export.json
- validation_report.json
- confidence.json
- build_log.md

Update index.json.

---

# 🛑 DEFINITION OF DONE

A build is complete only when:

- JSON passes structural validation
- Modules are connected properly
- No orphan nodes exist
- Credential placeholders are used
- Validation report is generated
- Confidence score calculated
- Artifacts versioned in /output

If any of these fail:
The build is NOT complete.

---

# 🎯 OUTPUT CLARITY RULE

Agents fail when “done” is vague.

Every workflow must define:

- Exact output format
- Exact file location
- Exact success conditions
- Exact stopping condition

Never operate without a defined finish line.

---

# 🔁 SELF-IMPROVEMENT LOOP

If a validation error occurs:

- Improve tool logic
- Improve schema validation
- Improve mapping rules
- Improve connection integrity checks

Never patch temporarily.

Fix at the tool layer.

Future builds must benefit automatically.

---

# 📂 FILE STRUCTURE (MANDATORY)

/workflows
/tools
/output
/temp
claude.md
.env

Do not write files outside this structure.

---

# 🔐 SECURITY RULES

- Never expose real credentials
- Always use placeholders
- Respect .env
- Never log secrets

---

# 🧮 CONFIDENCE SCORING

Score 0.0 – 1.0 based on:

- Validation success
- Warning count
- Assumptions made
- Spec completeness
- Retry count

Low score if:
- Validation required repair
- Ambiguity existed
- Unsupported modules were assumed

---

# 🚀 PRODUCTION MINDSET

This is not a demo system.

This builder must:
- Be deterministic
- Be reliable
- Be version-controlled
- Be structurally safe
- Improve over time

No shortcuts.
No skipped validation.
No uncontrolled autonomy.

---

# FINAL OPERATING PRINCIPLE

You are not here to generate code quickly.

You are here to:

1. Architect correctly  
2. Validate structurally  
3. Repair deterministically  
4. Improve continuously  
5. Deliver production-ready artifacts  

Accuracy over speed.
Structure over improvisation.
Validation over assumption.

---

# 📂 FULL FILE STRUCTURE (CURRENT STATE)

```
/
├── app.py                          # FastAPI entrypoint (POST /build, GET /health)
├── claude.md                       # This file — project instructions
├── demo_build.py                   # CLI demo runner (not used by app.py)
├── Dockerfile                      # Python 3.11, reads $PORT from env
├── requirements.txt                # fastapi, uvicorn, pydantic, sqlalchemy, psycopg2-binary
├── .env                            # Credentials (gitignored)
├── .gitignore
│
├── db/                             # PostgreSQL persistence layer
│   ├── __init__.py
│   ├── schema.sql                  # DDL: projects, builds, build_artifacts, assumptions
│   ├── session.py                  # SQLAlchemy engine, SessionLocal, get_db(), check_db()
│   ├── models.py                   # ORM: Project, Build, BuildArtifact, Assumption
│   └── repo.py                     # create_build(), store_artifact(), finalize_build()
│
├── tools/                          # Deterministic execution layer (no AI reasoning)
│   ├── __init__.py
│   ├── module_registry.json        # 16 modules, version 1.0.0
│   ├── module_registry_loader.py   # load_module_registry(), get_module()
│   ├── canonical_spec_schema.json  # JSON Schema for canonical spec (not yet wired)
│   ├── normalize_to_canonical_spec.py  # plan → canonical spec
│   ├── validate_canonical_spec.py      # 47 rules, returns validation report
│   ├── generate_make_export.py         # canonical spec → Make.com blueprint
│   ├── validate_make_export.py         # 30 rules, returns validation report
│   ├── self_heal_make_export.py        # 17 repairable rules, max 2 retries
│   ├── confidence_scorer.py            # 0.0–1.0 score + grade
│   ├── graph_integrity_check.py        # DAG validation, orphan detection
│   ├── data_mapping_extractor.py       # {{N.field}} reference extraction
│   ├── spec_version_manager.py         # Filesystem versioning (backward compat)
│   ├── build_scenario_pipeline.py      # Orchestrator (656 lines, dual DB/filesystem)
│   ├── timeline_estimator.py           # Heuristic implementation timeline
│   ├── cost_estimator.py               # Make.com ops/cost estimation
│   ├── delivery_packager.py            # summary_md + pack_json generation
│   ├── assumption_tracker.py           # Built, NOT wired into pipeline
│   ├── delivery_adapter.py             # Built, NOT wired into pipeline
│   └── generate_delivery_assessment.py # Built, NOT wired into pipeline
│
├── workflows/                      # Markdown planning docs
│   ├── phase1_canonical_spec.md
│   ├── phase2_make_export.md
│   ├── phase3_make_export_validation.md
│   ├── phase4_self_healing.md
│   └── phase5_pipeline_and_versioning.md
│
└── output/                         # Filesystem artifacts (legacy, replaced by DB)
    ├── index.json
    └── <slug>/vN/*.json|*.md
```

# 🗄 DATABASE ARCHITECTURE

## Persistence Flow

```
POST /build → app.py
  ├── db = Depends(get_db)
  └── build_scenario_pipeline(db_session=db, project_name="default")
        ├── create_build(db, project_name, slug, ...)
        │     ├── upsert project by name
        │     ├── pg_advisory_xact_lock(hashtext(pid || ':' || slug))
        │     ├── SELECT COALESCE(MAX(version),0)+1
        │     └── INSERT build (status='running')
        ├── [phases 1-4: normalize → validate → generate → heal]
        ├── store_artifact() × 10
        ├── finalize_build(db, build_id, "success", ...)
        └── return result dict
  → db.commit() on success
  → db.rollback() on exception
```

## Tables (db/schema.sql)

- **projects**: id uuid pk, name unique, created_at
- **builds**: id uuid pk, project_id fk, slug, version, original_request, status, confidence_score/grade, canonical_valid, export_valid, heal_attempts, failure_reason — UNIQUE(project_id, slug, version)
- **build_artifacts**: id uuid pk, build_id fk, artifact_type text, content_json jsonb, content_text text — UNIQUE(build_id, artifact_type)
- **assumptions**: id uuid pk, build_id fk, type, description, severity, created_at

## 10 Artifact Types Stored Per Build

canonical_spec, make_export, validation_report, export_validation_report,
confidence, build_log, timeline, cost_estimate, customer_delivery_summary, delivery_pack

## Dual Persistence

- **db_session provided** → all artifacts go to PostgreSQL (production path)
- **db_session=None + base_output_dir** → filesystem writes (self-test backward compat)
- Self-tests (`python -m tools.build_scenario_pipeline`) use filesystem, pass 8/8 checks

# 🔧 KNOWN GAPS (NOT YET IMPLEMENTED)

- Authentication / authorization on API endpoints
- Agent reasoning layer (system is currently a deterministic compiler)
- canonical_spec_schema.json not wired for runtime validation
- assumption_tracker.py, delivery_adapter.py, generate_delivery_assessment.py built but not wired
- Confidence not recalculated after self-healing
- No structured logging, rate limiting, or concurrency protection beyond advisory locks
- 3 tools built but unused in pipeline

# 🔗 REPOSITORY

GitHub: https://github.com/Brian2169fdsa/agenticmakebuilder
