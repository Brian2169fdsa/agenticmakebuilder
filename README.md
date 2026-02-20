# agenticmakebuilder v2.0.0

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-39%20passing-brightgreen?logo=pytest&logoColor=white)
![Endpoints](https://img.shields.io/badge/Endpoints-35-orange)

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

### Multi-Agent Orchestration (4 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/orchestrate` | Advance project through pipeline stages | `project_id`, `current_stage` |
| POST | `/agent/complete` | Agent completion + auto-advance | `project_id`, `agent_name`, `outcome` |
| GET | `/pipeline/status` | Full pipeline state view | `?project_id=` |
| POST | `/briefing` | Daily supervisor briefing report | — |

### Agent Memory & Learning (4 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/memory` | Store client context (decisions, tech stack, patterns) | `client_id`, `project_id`, `key_decisions` |
| GET | `/memory` | Retrieve client context | `?client_id=&project_id=` |
| POST | `/memory/embed` | Embed project brief for similarity search | `project_id`, `brief`, `outcome` |
| GET | `/similar` | TF-IDF cosine similarity search | `?description=&top_n=` |

### Confidence & Verification (3 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/verify` | 77-rule blueprint validation with fix instructions | `blueprint`, `project_id` (optional) |
| POST | `/verify/loop` | Iterative verify-fix-verify cycle (max 5 iterations) | `project_id`, `blueprint`, `max_iterations` |
| GET | `/confidence/history` | Verification run history with score trends | `?project_id=` |

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

### Persona Engine (5 endpoints)

| Method | Endpoint | Description | Key Params |
|--------|----------|-------------|------------|
| POST | `/persona/memory` | Link persona to client with tone/style prefs | `persona`, `client_id`, `tone_preferences` |
| GET | `/persona/context` | Persona's full context for a client | `?persona=&client_id=` |
| POST | `/persona/feedback` | Store interaction feedback | `persona`, `client_id`, `rating` |
| GET | `/persona/performance` | Persona performance stats | `?persona=` |
| POST | `/persona/deploy` | Generate client-specific persona artifact | `persona`, `client_id` |

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

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `MAKE_API_KEY` | No | Make.com API token for deployment + monitoring |
| `MAKE_TEAM_ID` | No | Make.com team ID |
| `MAKE_API_BASE` | No | Make.com API base URL (default: `https://us1.make.com/api/v2`) |
| `SLACK_WEBHOOK_URL` | No | Slack webhook for incident notifications |
| `MANAGEAI_HOURLY_RATE` | No | Hourly rate for SOW estimates (default: `150`) |
| `MONITOR_INTERVAL` | No | Execution poller interval in seconds (default: `900`) |
| `MONITOR_LOOKBACK` | No | Number of recent executions to check (default: `20`) |

## Architecture

```
                         agenticmakebuilder v2.0.0
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
  │   │  /orchestrate    /agent/complete             │              │
  │   │  /pipeline/status  /briefing                 │              │
  │   └──────────────────────────────────────────────┘              │
  │          │              │              │                         │
  │          ▼              ▼              ▼                         │
  │   ┌────────────┐ ┌───────────┐ ┌────────────┐                  │
  │   │   MEMORY   │ │   COSTS   │ │   HEALTH   │                  │
  │   │            │ │           │ │            │                  │
  │   │ /memory    │ │ /costs/   │ │ /health/   │                  │
  │   │ /similar   │ │  track    │ │  full      │                  │
  │   │ /memory/   │ │  summary  │ │  memory    │                  │
  │   │  embed     │ │  report   │ │  repair    │                  │
  │   │            │ │  estimate │ │            │                  │
  │   │ TF-IDF     │ │           │ │ Self-heal  │                  │
  │   │ Vectors    │ │ Margin    │ │ Auto-fix   │                  │
  │   └────────────┘ │ Alerts    │ └────────────┘                  │
  │                  └───────────┘                                  │
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

## Running Tests

```bash
# Activate venv
source .venv/bin/activate

# Install test dependencies
pip install pytest httpx

# Run full suite
pytest tests/ -v

# Expected: 39 passed
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
