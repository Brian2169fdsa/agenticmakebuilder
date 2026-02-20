# agenticmakebuilder v2.2.0

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-87%20passing-brightgreen?logo=pytest&logoColor=white)
![Endpoints](https://img.shields.io/badge/Endpoints-51-orange)

## What It Does

Agenticmakebuilder is ManageAI's agentic delivery platform. It takes a natural language project brief, runs it through a multi-agent pipeline (assess, build, verify, deploy), and produces validated Make.com scenario blueprints ready for client handoff. The platform tracks costs, learns from past projects, and self-heals when things go wrong.

The system orchestrates four specialized agents — Assessor, Builder, Validator, and Deployer — through a state-machine pipeline. Each project flows from intake to deployment with confidence scoring at every stage, automatic pipeline advancement, TF-IDF similarity search against past projects, and real-time cost/margin tracking per client.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/Brian2169fdsa/agenticmakebuilder.git
cd agenticmakebuilder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure database
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"

# Run
uvicorn app:app --reload

# Verify
curl http://localhost:8000/health
# → {"ok": true}
```

## Full Endpoint Reference

### Core Pipeline (6 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| GET | `/health` | Liveness probe | — |
| POST | `/intake` | Natural language front door | `message`, `customer_name` |
| POST | `/assess` | Structured intake → delivery report + plan_dict | `original_request`, `customer_name`, `trigger_type` |
| POST | `/plan` | Full pipeline: assess + build + 11 artifacts | `original_request`, `customer_name`, `processing_steps` |
| POST | `/build` | Compiler direct (requires plan_dict) | `plan`, `original_request` |
| POST | `/audit` | Audit existing Make.com blueprint | `blueprint`, `scenario_name` |

### Multi-Agent Orchestration (6 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/orchestrate` | Advance project through pipeline stages | `project_id`, `current_stage` |
| POST | `/agent/complete` | Agent completion + auto-advance | `project_id`, `agent_name`, `outcome` |
| GET | `/pipeline/status` | Full pipeline state view | `?project_id=` |
| POST | `/pipeline/advance` | Advance project to next/specific stage | `project_id`, `force_stage` |
| GET | `/pipeline/dashboard` | All projects grouped by pipeline stage | — |
| POST | `/briefing` | Daily supervisor briefing report | — |

### Intelligence Layer (4 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/clients/health` | Client health assessment (healthy/at_risk/unhealthy) | — |
| GET | `/clients/list` | Client directory with project stats | — |
| POST | `/briefing/daily` | Comprehensive daily briefing with alerts + recommendations | — |
| POST | `/handoff` | Multi-agent handoff bridge | `from_agent`, `to_agent`, `project_id` |

### Agent Memory & Learning (4 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/memory` | Store client context (decisions, tech stack, patterns) | `client_id`, `project_id`, `key_decisions` |
| GET | `/memory` | Retrieve client context | `?client_id=&project_id=` |
| POST | `/memory/embed` | Embed project brief for similarity search | `project_id`, `brief`, `outcome` |
| GET | `/similar` | TF-IDF cosine similarity search | `?description=&top_n=` |

### Confidence & Verification (5 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/verify` | 77-rule blueprint validation with fix instructions | `blueprint`, `project_id` (optional) |
| POST | `/verify/loop` | Iterative verify-fix-verify cycle (max 5 iterations) | `project_id`, `blueprint`, `max_iterations` |
| POST | `/verify/auto` | Auto-verify + pipeline advance if confidence >= 85 | `project_id`, `blueprint` |
| GET | `/confidence/history` | Verification run history with score trends | `?project_id=` |
| GET | `/confidence/trend` | Confidence score trend (improving/declining/stable) | `?project_id=` |

### Deployment Agent (3 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/deploy/makecom` | Deploy blueprint to Make.com via API | `project_id`, `blueprint`, `api_key` |
| POST | `/deploy/n8n` | Deploy workflow to n8n via REST API | `project_id`, `workflow`, `n8n_url`, `api_key` |
| GET | `/deploy/status` | Deployment status + health checks | `?project_id=` |

### Cost & Margin Intelligence (4 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/costs/track` | Track token costs with auto-margin + alerts | `project_id`, `model`, `input_tokens`, `output_tokens` |
| GET | `/costs/summary` | Cost/revenue/margin per client | `?client_id=` |
| GET | `/costs/report` | Weekly cost report in markdown | `?client_id=&weeks=` |
| POST | `/costs/estimate` | Pre-build cost estimation from historical data | `description`, `category` |

### Persona Engine (7 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/persona/memory` | Link persona to client with tone/style prefs | `persona`, `client_id`, `tone_preferences` |
| GET | `/persona/context` | Persona's full context for a client | `?persona=&client_id=` |
| POST | `/persona/feedback` | Store interaction feedback | `persona`, `client_id`, `rating` |
| GET | `/persona/performance` | Persona performance stats | `?persona=` |
| POST | `/persona/deploy` | Generate client-specific persona artifact (v2.0) | `persona`, `client_id` |
| POST | `/persona/test` | Test persona via Claude API | `persona`, `message`, `client_id` |
| GET | `/personas/list` | All personas with live stats | — |

### Admin Control Plane (5 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/admin/reset-project` | Reset project pipeline to target stage | `project_id`, `target_stage` |
| POST | `/admin/bulk-verify` | Batch verify up to 10 projects | `project_ids`, `blueprint` |
| GET | `/admin/system-status` | Full platform snapshot (DB, registry, costs, personas) | — |
| POST | `/admin/reindex` | Re-embed all client context into TF-IDF store | — |
| GET | `/admin/audit-log` | Agent handoff audit trail | `?limit=` |

### Platform Health (3 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| GET | `/health/full` | Comprehensive check: DB, tables, embeddings, pipeline | — |
| GET | `/health/memory` | Embedding store stats + vocabulary analysis | — |
| POST | `/health/repair` | Self-healing: stalled pipelines, orphans, stale deploys | — |

### Supervisor (1 endpoint)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| GET | `/supervisor/stalled` | Detect stalled projects (>48h no update) | — |

### Natural Language Command (1 endpoint)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/command` | Route free-text commands to endpoints | `command`, `customer_name` |

## Architecture

```
                         agenticmakebuilder v2.2.0
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   Client                                                        │
  │     │                                                           │
  │     ▼                                                           │
  │   /intake ──or── /plan                                          │
  │     │              │                                            │
  │     ▼              ▼                                            │
  │   ┌──────────────────────────────────────────────┐              │
  │   │          ORCHESTRATION PIPELINE              │              │
  │   │                                              │              │
  │   │  intake ──► build ──► verify ──► deploy      │              │
  │   │    │          │          │          │         │              │
  │   │  Assessor  Builder  Validator  Deployer      │              │
  │   │                                              │              │
  │   │  /orchestrate     /agent/complete            │              │
  │   │  /pipeline/status /pipeline/advance          │              │
  │   │  /pipeline/dashboard                         │              │
  │   └──────────────────────────────────────────────┘              │
  │          │              │              │              │          │
  │          ▼              ▼              ▼              ▼          │
  │   ┌────────────┐ ┌───────────┐ ┌────────────┐ ┌──────────┐    │
  │   │   MEMORY   │ │   COSTS   │ │   HEALTH   │ │  ADMIN   │    │
  │   │            │ │           │ │            │ │          │    │
  │   │ /memory    │ │ /costs/   │ │ /health/   │ │ /admin/  │    │
  │   │ /similar   │ │  track    │ │  full      │ │  reset   │    │
  │   │ /memory/   │ │  summary  │ │  memory    │ │  bulk-   │    │
  │   │  embed     │ │  report   │ │  repair    │ │  verify  │    │
  │   │            │ │  estimate │ │            │ │  system  │    │
  │   │ TF-IDF     │ │           │ │ Self-heal  │ │  reindex │    │
  │   │ Vectors    │ │ Margin    │ │ Auto-fix   │ │  audit   │    │
  │   └────────────┘ │ Alerts    │ └────────────┘ └──────────┘    │
  │                  └───────────┘                                  │
  │                       │                                         │
  │   ┌──────────────────────────────────────────────┐              │
  │   │   SUPABASE SYNC LAYER                        │              │
  │   │                                              │              │
  │   │  supabase_client.py  → REST API (httpx)      │              │
  │   │  pipeline_sync.py    → State + Verification  │              │
  │   │  notification_sender → Stall/Cost Alerts     │              │
  │   └──────────────────────────────────────────────┘              │
  │                                                                 │
  │   ┌──────────────────────────────────────────────┐              │
  │   │   /command — Natural Language Router          │              │
  │   │   "check health" → /health/full              │              │
  │   │   "show costs for Acme" → /costs/summary     │              │
  │   │   "find similar to CRM sync" → /similar      │              │
  │   └──────────────────────────────────────────────┘              │
  │                                                                 │
  │   PostgreSQL (Supabase/Railway)    embeddings.json (TF-IDF)     │
  └─────────────────────────────────────────────────────────────────┘
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `ANTHROPIC_API_KEY` | No | Anthropic API key for persona test endpoint |
| `MAKE_API_KEY` | No | Make.com API token for deployment + monitoring |
| `MAKE_TEAM_ID` | No | Make.com team ID |
| `MAKE_API_BASE` | No | Make.com API base URL (default: `https://us1.make.com/api/v2`) |
| `SLACK_WEBHOOK_URL` | No | Slack webhook for incident notifications |
| `MANAGEAI_HOURLY_RATE` | No | Hourly rate for SOW estimates (default: `150`) |
| `MONITOR_INTERVAL` | No | Execution poller interval in seconds (default: `900`) |
| `MONITOR_LOOKBACK` | No | Number of recent executions to check (default: `20`) |
| `SUPABASE_URL` | No | connect-hub Supabase project URL for real-time sync |
| `SUPABASE_SERVICE_KEY` | No | Service role key for server-side writes to Supabase |

## Running Tests

```bash
# Activate venv
source .venv/bin/activate

# Install test dependencies
pip install pytest httpx

# Run full suite
pytest tests/ -v

# Expected: 87 passed
```

Tests mock the database layer so no Postgres connection is needed to run them.

## Deployment (Railway)

1. Connect your GitHub repo to [Railway](https://railway.app)
2. Add a PostgreSQL plugin
3. Set `DATABASE_URL` to the Railway Postgres connection string
4. Railway auto-detects Python and runs via nixpacks
5. Initialize the database:
   ```bash
   psql $DATABASE_URL < db/schema.sql
   ```

The app binds to `$PORT` automatically when deployed.
