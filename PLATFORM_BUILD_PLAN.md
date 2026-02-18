# ManageAI — Agentic Delivery OS
## Complete Platform Build Plan
**Version: 1.0 | Date: 2026-02-17**

---

## What This Document Is

This is the master build plan for turning the Agentic Make Builder into
a full-stack AI delivery operating system. Every feature, every repo,
every migration path. Written to execute against, not just read.

---

## Platform Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    ManageAI Platform                            │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  agenticmakebuilder  (this repo — core compiler)          │   │
│  │                                                           │   │
│  │  Intent → Canonical Spec → Any Automation Platform        │   │
│  │                                                           │   │
│  │  + SOW Generator          (add to this repo)             │   │
│  │  + n8n Compiler           (add to this repo)             │   │
│  │  + Zapier Compiler        (add to this repo)             │   │
│  │  + Integration Tester     (add to this repo)             │   │
│  │  + Live Monitor           (add to this repo)             │   │
│  │  + Client Intelligence    (add to this repo)             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────┐   ┌──────────────────┐                    │
│  │  persona-builder │   │  client-portal   │                    │
│  │  (own repo)      │   │  (own repo)      │                    │
│  └──────────────────┘   └──────────────────┘                    │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Infrastructure Migration Path                           │   │
│  │  Make.com → n8n (workflows) → Temporal (orchestration)   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

# SECTION 1 — ADDITIONS TO THIS REPO
*agenticmakebuilder — add in this order*

---

## BUILD 1 — SOW & Proposal Generator
**Fastest business win. All data already exists in the pipeline.**
**Estimated build time: 1 day**

### What It Does
After every successful `/plan` call, generates a complete business document
package ready to send to the client. No manual writing.

### Outputs
1. **Statement of Work** (DOCX) — scope, timeline, cost, deliverables
2. **Executive Summary** (1-page PDF) — for non-technical stakeholders
3. **Technical Spec** (MD) — for your internal delivery team
4. **Implementation Checklist** (MD) — step-by-step deployment guide

### Data Sources (all already in pipeline)
- `timeline.json` → timeline section, milestones
- `cost_estimate.json` → pricing section, Make.com plan recommendation
- `canonical_spec.json` → technical spec, module list
- `delivery_pack.json` → scope, customer name, scenario description
- `confidence.json` → risk section (low confidence = flag for review)
- `build_blueprint.md` → technical appendix

### Files to Build
```
tools/
  sow_generator.py          ← main generator, reads artifacts, renders templates
  sow_templates/
    sow_template.md         ← Statement of Work markdown template
    exec_summary.md         ← Executive summary template
    tech_spec.md            ← Technical spec template
    checklist.md            ← Implementation checklist template
```

### New Endpoints
```
POST /sow/{slug}/{version}  ← generate SOW package from existing build
GET  /sow/{slug}/{version}  ← download SOW ZIP (same as /download but filtered)
```

### Auto-trigger
Wire into `/plan` so every successful build automatically generates the SOW
and includes `sow_path` in the response. Sales team gets it immediately.

### Schema Addition
```sql
CREATE TABLE sow_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID REFERENCES builds(id),
    generated_at TIMESTAMPTZ DEFAULT now(),
    sow_text TEXT,
    exec_summary_text TEXT,
    tech_spec_text TEXT,
    checklist_text TEXT
);
```

---

## BUILD 2 — n8n Compiler
**Adds n8n as a second compile target. Zero changes to canonical spec.**
**Estimated build time: 2 days**

### What It Does
Adds `generate_n8n_export.py` which converts the same canonical spec
into a valid n8n workflow JSON. The `/plan` endpoint gets a `target`
parameter: `"make"` (default) | `"n8n"`.

### n8n vs Make.com Differences
| Concept | Make.com | n8n |
|---------|----------|-----|
| Trigger | gateway:CustomWebHook | n8n-nodes-base.webhook |
| Flow array | `flow: [{id, module, ...}]` | `nodes: [{type, name, ...}]` |
| Connections | implicit by `routes` | explicit `connections` dict |
| Credentials | connection objects | credential references |
| Parameters | `parameters` + `mapper` | `parameters` flat |

### Files to Build
```
tools/
  generate_n8n_export.py        ← n8n workflow generator
  validate_n8n_export.py        ← n8n-specific validation rules
  n8n_module_registry.json      ← n8n node type registry
```

### New Endpoints
```
POST /plan?target=n8n           ← full pipeline → n8n export
POST /build?target=n8n          ← direct build → n8n
GET  /download/{slug}/{v}?format=n8n  ← download n8n workflow JSON
```

### Module Registry Strategy
Keep separate registries: `module_registry.json` (Make.com) and
`n8n_module_registry.json`. The canonical spec normalizer uses whichever
registry matches the target. The same intent maps to different nodes.

### Validation
Same 77-rule framework applied to n8n structure. Adapt the most critical
rules: credential format, connection integrity, node ID uniqueness.

### Migration Path from Make.com → n8n
```
tools/make_to_n8n_migrator.py   ← reads make_export.json, produces n8n_export.json
```
This lets you take any existing client Make.com scenario and generate
the equivalent n8n workflow. Critical for client migrations.

---

## BUILD 3 — Integration Testing Agent
**Proves what you built actually works before client delivery.**
**Estimated build time: 3 days**

### What It Does
After a build completes, generates and executes a test suite against the
scenario using the Make.com API sandbox. Returns pass/fail per module with
actual execution results.

### Test Lifecycle
```
1. Generate synthetic test payload from canonical spec
2. POST to Make.com API → create test execution
3. Poll for completion (max 30 seconds)
4. Capture output at each module
5. Compare actual vs expected outputs
6. Report pass/fail with diff
```

### Test Generation
The AI reads the canonical spec and generates:
- Happy path test (normal inputs, expected outputs)
- Edge case tests (empty payload, missing fields)
- Error path test (triggers error handling branch)

### Files to Build
```
tools/
  test_generator.py         ← generates synthetic test payloads from canonical spec
  make_api_client.py        ← Make.com API wrapper (create/run/poll scenarios)
  test_runner.py            ← orchestrates test execution
  test_reporter.py          ← formats test results into readable report
```

### New Endpoints
```
POST /test/{slug}/{version}     ← run test suite against a build
GET  /test/{slug}/{version}     ← get latest test results
```

### New Artifact
Add `test_report.json` and `test_report.md` to the 11-artifact output,
making it 13 artifacts.

### Confidence Impact
Wire test results into confidence scorer. A build that passes all tests
gets a +0.05 confidence bonus. A build that fails tests gets flagged
regardless of structural validation score.

### Configuration
```env
MAKE_API_KEY=your-make-api-key
MAKE_TEAM_ID=your-team-id
MAKE_TEST_SANDBOX=true
```

---

## BUILD 4 — Live Scenario Monitor
**Real-time error detection, not just daily audits.**
**Estimated build time: 2 days**

### What It Does
Polls Make.com API every 15 minutes for execution history across all
registered client scenarios. Detects error spikes, credential failures,
rate limit hits. Auto-creates repair tickets.

### Detection Patterns
- **Error spike**: >3 failures in last hour on a previously healthy scenario
- **Credential failure**: 401/403 errors → flag specific connection
- **Rate limit**: 429 errors → flag plan upgrade needed
- **Consecutive failures**: 5+ failures with no successes → alert critical
- **Silence**: Scenario expected to run but hasn't → dead trigger alert

### Files to Build
```
tools/
  make_execution_poller.py     ← polls Make.com API for execution history
  error_classifier.py          ← classifies error types and severity
  incident_manager.py          ← creates and tracks repair incidents
```

### New DB Tables
```sql
CREATE TABLE incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL,
    customer_name TEXT,
    incident_type TEXT NOT NULL,     -- error_spike, credential_failure, etc.
    severity TEXT NOT NULL,          -- low, medium, high, critical
    detected_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    status TEXT DEFAULT 'open',
    details JSONB
);

CREATE TABLE execution_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL,
    snapped_at TIMESTAMPTZ DEFAULT now(),
    executions_last_hour INTEGER,
    errors_last_hour INTEGER,
    success_rate FLOAT,
    last_error TEXT
);
```

### New Endpoints
```
GET  /monitor/status              ← health status of all monitored scenarios
GET  /monitor/{slug}              ← execution history + incidents for one scenario
POST /monitor/poll                ← trigger manual poll cycle
GET  /incidents                   ← list open incidents
POST /incidents/{id}/resolve      ← mark incident resolved
```

### Alerting Integration
Wires into existing `tools/alerting.py`. Incidents above `high` severity
fire Slack alert immediately (not waiting for daily digest).

---

## BUILD 5 — Client Intelligence Layer
**Turns every build into institutional knowledge.**
**Estimated build time: 3 days**

### What It Does
Aggregates all build and audit data per client. Surfaces patterns,
expansion opportunities, and risk. Generates automated monthly reports.

### Intelligence Generated
- Total automation footprint per client (all scenarios, all platforms)
- Combined Make.com ops consumption and cost projection
- Module usage patterns ("Acme always uses Airtable, never uses error handling")
- Scenario health trends over 30/60/90 days
- Expansion opportunity scoring ("3 scenarios could be merged, saving 40 ops/day")
- Risk scoring ("2 scenarios have no error handling, 1 has credential expiry in 14 days")

### Files to Build
```
tools/
  client_aggregator.py         ← aggregates all builds by customer_name
  pattern_analyzer.py          ← identifies usage patterns and opportunities
  report_generator.py          ← generates monthly client health report
  opportunity_scorer.py        ← scores expansion opportunities by value/effort
```

### New DB Tables
```sql
CREATE TABLE client_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_name TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    total_scenarios INTEGER,
    healthy_scenarios INTEGER,
    total_ops_per_month INTEGER,
    estimated_monthly_cost FLOAT,
    risk_score FLOAT,
    opportunity_score FLOAT,
    details JSONB,
    UNIQUE(customer_name, snapshot_date)
);
```

### New Endpoints
```
GET  /clients                      ← list all clients with summary stats
GET  /clients/{name}               ← full intelligence profile for one client
GET  /clients/{name}/report        ← download monthly report (PDF/DOCX)
POST /clients/{name}/snapshot      ← trigger manual snapshot
GET  /opportunities                ← expansion opportunities ranked by value
```

### Monthly Report Auto-Generation
Schedule via cron/Make.com on the 1st of each month:
- Client name + account summary
- All scenarios with health status
- Ops usage and cost trend
- Risk items requiring attention
- Expansion recommendations with ROI estimates
- Next month preview

---

## BUILD 6 — Zapier Compiler
**Third compile target. Lower priority than n8n.**
**Estimated build time: 1.5 days (after n8n is done)**

### What It Does
Same pattern as n8n compiler. Adds `generate_zapier_export.py`.
Lower priority because Zapier has less flexibility and smaller clients.
Build after n8n is proven.

### Key Difference from n8n
Zapier Zaps are strictly linear (trigger → action → action).
No branching, no routing. The canonical spec router logic must be
flattened or the build fails with a clear error.

---

# SECTION 2 — SEPARATE REPOS

---

## REPO 2 — Persona Builder
**Your actual product line made into repeatable infrastructure.**
**Repo name: `persona-builder`**
**Estimated build time: 1 week**

### What It Is
A separate service that builds, versions, and deploys AI assistant personas.
Follows the exact same architecture as agenticmakebuilder but for a
completely different artifact type.

### Inputs
```json
{
  "persona_name": "Rebecka",
  "role": "Executive Assistant",
  "client": "Acme Corp",
  "tone": "professional",
  "capabilities": ["scheduling", "email_drafting", "research", "summarization"],
  "knowledge_base": [
    {"type": "pdf", "source": "company_overview.pdf"},
    {"type": "url", "source": "https://acme.com/faq"},
    {"type": "text", "source": "Custom instructions..."}
  ],
  "escalation_rules": {
    "always_escalate": ["legal questions", "pricing above $10k"],
    "escalate_after_turns": 5,
    "escalation_contact": "brian@acme.com"
  },
  "platforms": ["claude", "openai_assistant", "custom_api"],
  "memory_mode": "per_session" | "persistent" | "none"
}
```

### Outputs (one package per persona)
1. **System prompt** (versioned, templated, tested)
2. **OpenAI Assistant config JSON** (ready to POST to Assistants API)
3. **Claude Project instructions** (ready to paste)
4. **Knowledge base upload manifest** (files + metadata)
5. **Test conversation suite** (10 scenarios: happy path, edge cases, escalation)
6. **Deployment checklist** (step-by-step for delivery team)
7. **Client-facing persona card** (what the persona can/can't do)

### Pipeline
```
intake → normalize_persona_spec → validate_persona_spec →
generate_system_prompt → validate_system_prompt →
generate_platform_configs → generate_test_suite →
version + persist → deliver
```

### Key Tools
```
tools/
  normalize_persona_spec.py       ← intake → canonical persona spec
  validate_persona_spec.py        ← 40+ persona validation rules
  system_prompt_generator.py      ← canonical spec → system prompt
  openai_config_generator.py      ← → OpenAI Assistant JSON
  claude_config_generator.py      ← → Claude Project instructions
  test_suite_generator.py         ← generates test conversations via Claude API
  persona_tester.py               ← runs test suite against live persona
  knowledge_base_packager.py      ← packages files for upload
```

### Integration with agenticmakebuilder
Personas and automation scenarios are linked. A client's Rebecka persona
should know about their Make.com scenarios. Add a cross-reference:

```json
{
  "linked_scenarios": ["acme-form-to-slack", "acme-lead-qualifier"],
  "can_trigger": ["acme-form-to-slack"],
  "receives_output_from": ["acme-lead-qualifier"]
}
```

### Endpoints
```
POST /persona/plan           ← full pipeline
POST /persona/assess         ← assessment only
POST /persona/test           ← run test suite
GET  /persona/{name}/v{n}    ← get specific version
GET  /personas               ← list all personas
GET  /download/persona/{name}/{v}  ← download deployment package
```

### Versioning Model
Same slug+version model as agenticmakebuilder. When the client changes
their company name, you rebuild v2 of the persona with a full diff.

---

## REPO 3 — Client Portal
**The Lovable-facing product your clients interact with.**
**Repo name: `manageai-portal`**
**Estimated build time: 2 weeks**

### What It Is
A web application (React + Supabase) that gives clients a dashboard to:
- View all their active scenarios and personas
- See health status in real-time
- Download their delivery artifacts
- Submit new automation requests
- Track build progress

### Pages
```
/dashboard          ← overview: all scenarios, health summary
/scenarios          ← list of all builds with status
/scenarios/{slug}   ← single scenario: versions, health, download
/personas           ← list of all personas
/personas/{name}    ← single persona: versions, test results
/requests           ← submit new automation request (feeds /intake)
/reports            ← monthly client intelligence reports
/settings           ← API key, notification preferences
```

### Integration Points
- Calls `agenticmakebuilder` API for all build operations
- Calls `persona-builder` API for persona operations
- Subscribes to Supabase realtime for live status updates
- Uses `/job/{id}` polling for async build status

### Auth
Supabase Auth. Each client gets their own org. Multi-tenant by design.
Delivery team has admin view across all clients.

---

# SECTION 3 — INFRASTRUCTURE MIGRATION

---

## MIGRATION 1 — Make.com → n8n
**Self-hosted, more powerful, no per-operation pricing.**

### Why n8n
- Self-hosted → no Make.com pricing surprises for high-volume clients
- Code nodes → arbitrary JavaScript/Python in the workflow
- Better error handling primitives
- Native queue support
- REST/GraphQL API coverage is better
- You control the infrastructure

### Migration Strategy (3 phases)

**Phase 1 — Parallel compiler (2 days)**
Add n8n as a compile target (Build 2 above). Generate both Make.com and
n8n exports from every build. No client impact. Internal validation only.

**Phase 2 — Pilot migration (2 weeks)**
Pick 2 low-risk client scenarios. Use `make_to_n8n_migrator.py` to convert.
Run both in parallel. Compare execution results. Fix edge cases.

**Phase 3 — Full migration (1-2 months)**
For each client, schedule migration window. Run migrator, test, cut over.
Keep Make.com as fallback for 30 days. Decommission after confirmation.

### n8n Infrastructure Setup
```yaml
# docker-compose.yml addition
n8n:
  image: n8nio/n8n
  ports:
    - "5678:5678"
  environment:
    - N8N_BASIC_AUTH_ACTIVE=true
    - N8N_BASIC_AUTH_USER=${N8N_USER}
    - N8N_BASIC_AUTH_PASSWORD=${N8N_PASSWORD}
    - DB_TYPE=postgresdb
    - DB_POSTGRESDB_HOST=postgres
    - DB_POSTGRESDB_DATABASE=n8n
  volumes:
    - n8n_data:/home/node/.n8n
```

### What Changes in the Compiler
- `generate_n8n_export.py` replaces `generate_make_export.py` as default
- `validate_n8n_export.py` runs instead of `validate_make_export.py`
- `n8n_module_registry.json` replaces `module_registry.json` as default
- `/audit` endpoint posts to n8n API instead of validating Make.com export
- `make_api_client.py` replaced with `n8n_api_client.py`

---

## MIGRATION 2 — Background Tasks → Temporal
**Production-grade workflow orchestration.**

### Why Temporal
- FastAPI `BackgroundTasks` is not durable. If the server crashes mid-build,
  the job is lost. Temporal persists workflow state and auto-resumes.
- Built-in retry logic with backoff, timeouts, and compensation
- Full workflow history — every step, every retry, every result
- Scales horizontally — multiple workers pick up jobs from the queue
- Visibility UI — see exactly where any workflow is at any point

### What Temporal Replaces
- `tools/job_queue.py` (in-memory job dict)
- FastAPI `BackgroundTasks` in `/lovable-ticket`
- Manual retry logic in `auto_iterator.py`
- Cron scheduling for `/audit-daily`

### What Temporal Workflows Look Like
```python
# workflows/build_workflow.py
@workflow.defn
class BuildWorkflow:
    @workflow.run
    async def run(self, input: BuildInput) -> BuildOutput:
        # Each activity is durable, auto-retried on failure
        assessment = await workflow.execute_activity(
            assess_intake, input, start_to_close_timeout=timedelta(seconds=30)
        )
        build = await workflow.execute_activity(
            run_pipeline, assessment, start_to_close_timeout=timedelta(minutes=5)
        )
        if build.confidence < 0.80:
            build = await workflow.execute_activity(
                auto_iterate, build, start_to_close_timeout=timedelta(minutes=10)
            )
        await workflow.execute_activity(
            fire_callback, build, start_to_close_timeout=timedelta(seconds=10)
        )
        return build
```

### Migration Phases

**Phase 1 — Temporal alongside FastAPI (1 week)**
Install Temporal server via Docker. Convert `/lovable-ticket` to use
Temporal workflow instead of BackgroundTasks. Everything else unchanged.

**Phase 2 — Migrate audit scheduler (3 days)**
Replace cron + `/audit-daily` with a Temporal scheduled workflow.
Gets retry logic, history, and visibility for free.

**Phase 3 — Migrate auto-iterator (3 days)**
Auto-iterator becomes a Temporal workflow with proper retry policies
and timeout handling. No more manual loop counting.

**Phase 4 — Full orchestration (1 week)**
The entire `/plan` pipeline becomes a Temporal workflow.
Every phase is an activity. Full visibility into every step.
Server can crash mid-build and resume exactly where it left off.

### Infrastructure Addition
```yaml
# docker-compose.yml
temporal:
  image: temporalio/auto-setup:latest
  ports:
    - "7233:7233"
  environment:
    - DB=postgresql
    - DB_PORT=5432
    - POSTGRES_USER=${POSTGRES_USER}
    - POSTGRES_PWD=${POSTGRES_PWD}

temporal-ui:
  image: temporalio/ui:latest
  ports:
    - "8080:8080"
  environment:
    - TEMPORAL_ADDRESS=temporal:7233
```

---

# SECTION 4 — FULL EXECUTION ROADMAP

---

## Phase 1 — Core Platform Complete (Now → Week 2)
*All additions to agenticmakebuilder*

| Build | Time | Business Value |
|-------|------|----------------|
| SOW Generator | 1 day | Closes sales loop, immediate demo |
| n8n Compiler | 2 days | Platform independence, bigger client pool |
| Integration Tester | 3 days | Proof of delivery, client trust |
| Live Monitor | 2 days | Production safety net |
| Client Intelligence | 3 days | Monthly reports, expansion revenue |

**End state:** agenticmakebuilder is a complete automation delivery platform.
Every build produces 13+ artifacts, is tested before delivery, monitored
in production, and reported to the client monthly.

---

## Phase 2 — Second Product Line (Week 3-4)
*persona-builder repo*

| Build | Time | Business Value |
|-------|------|----------------|
| Persona spec + validator | 2 days | Foundation |
| System prompt generator | 1 day | Core output |
| Platform config generators | 2 days | OpenAI + Claude configs |
| Test suite generator | 2 days | Quality assurance |
| Delivery package + API | 1 day | Client delivery |

**End state:** Building a new AI persona takes 30 minutes instead of
a full day of manual prompt engineering. Consistent quality across all
4 persona products.

---

## Phase 3 — Client-Facing Portal (Week 5-6)
*manageai-portal repo*

| Build | Time | Business Value |
|-------|------|----------------|
| Dashboard + scenario list | 2 days | Client self-service |
| Build request intake | 1 day | Sales pipeline automation |
| Download + reporting | 2 days | Client transparency |
| Real-time status updates | 2 days | Delivery confidence |
| Admin view for team | 1 day | Internal operations |

**End state:** Clients log in and see their entire automation portfolio.
They submit new requests. They download their artifacts. They see health
status. Your team sees everything across all clients.

---

## Phase 4 — Infrastructure Upgrade (Month 2)
*n8n migration + Temporal*

| Migration | Time | Business Value |
|-----------|------|----------------|
| n8n compiler + pilot | 1 week | Cost reduction, more power |
| Temporal for Lovable builds | 1 week | Durability, no lost jobs |
| Temporal for audit scheduler | 3 days | Reliable scheduled workflows |
| Full Temporal orchestration | 1 week | Production-grade resilience |

**End state:** The platform runs on self-hosted n8n instead of Make.com.
Every workflow is durable. Server crashes don't lose builds. Full
visibility into every step of every workflow.

---

## Phase 5 — Zapier + Scale (Month 3)
*Zapier compiler + horizontal scaling*

| Build | Time | Business Value |
|-------|------|----------------|
| Zapier compiler | 1.5 days | SMB client coverage |
| Multi-worker Temporal | 3 days | Concurrent builds |
| Rate limiting + quotas | 2 days | Multi-tenant fairness |
| Alembic migrations | 2 days | Safe DB schema changes |

---

# SECTION 5 — ENVIRONMENT VARIABLES (COMPLETE)

Add all of these to your `.env` as you build:

```env
# Core
DATABASE_URL=postgresql://...
API_KEYS=key1,key2
MANAGEAI_INTERNAL_KEY=internal-key
ANTHROPIC_API_KEY=sk-ant-...

# Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
ALERT_EMAIL=alerts@manageai.io
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=alerts@manageai.io
SMTP_PASSWORD=...

# Make.com API (for integration tester + live monitor)
MAKE_API_KEY=...
MAKE_TEAM_ID=...

# n8n (after migration)
N8N_API_URL=http://localhost:5678
N8N_API_KEY=...

# Temporal (after migration)
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=manageai

# Auto-iterator
AUTO_ITER_MAX=2
AUTO_ITER_MIN_CONF=0.80
AUTO_ITER_MODEL=claude-sonnet-4-6
```

---

# SECTION 6 — FINAL ARCHITECTURE DIAGRAM

```
Client Request
     │
     ▼
┌──────────────┐     ┌──────────────────────────────────────────┐
│ Client Portal│────▶│           agenticmakebuilder              │
│ (React/Next) │     │                                           │
└──────────────┘     │  /intake → /plan → /build → /audit       │
                     │  /sow    → /test  → /monitor → /metrics  │
                     │  /diff   → /download → /clients          │
                     └──────────────────────────────────────────┘
                                       │
              ┌────────────────────────┼──────────────────────┐
              │                        │                       │
              ▼                        ▼                       ▼
     ┌──────────────┐       ┌──────────────────┐    ┌─────────────────┐
     │  PostgreSQL  │       │  Temporal Server  │    │  n8n / Make.com │
     │  (Supabase)  │       │  (Workflow State) │    │  (Execution)    │
     └──────────────┘       └──────────────────┘    └─────────────────┘
              │
              ▼
     ┌──────────────┐
     │  persona-    │
     │  builder     │
     │  (own API)   │
     └──────────────┘
```

---

# DEFINITION OF DONE — FULL PLATFORM

- [ ] SOW generated on every successful build
- [ ] n8n compiler producing valid workflows
- [ ] Integration tests run before every delivery
- [ ] Live monitor polling every 15 minutes
- [ ] Client intelligence snapshots run monthly
- [ ] Persona builder API live and building personas
- [ ] Client portal accessible to clients
- [ ] n8n migration complete for pilot clients
- [ ] Temporal handling all async workflows
- [ ] Every endpoint behind API key auth
- [ ] Structured logging on every operation
- [ ] Slack alerts on regressions + incidents
- [ ] Monthly reports auto-generated on the 1st

---

**This is the full platform. Start with SOW Generator tomorrow.**
**Every build after that adds a layer.**
**In 90 days this is a product nobody in the boutique AI agency space has.**
