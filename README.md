# agenticmakebuilder v3.0.0

![Python 3.13](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![Tests](https://img.shields.io/badge/Tests-206%20passing-brightgreen?logo=pytest&logoColor=white)
![Endpoints](https://img.shields.io/badge/Endpoints-79-orange)
![Make.com](https://img.shields.io/badge/Make.com-Live%20Deploy-7B68EE)
![Railway](https://img.shields.io/badge/Railway-Deployed-000?logo=railway)

## What It Does

Agenticmakebuilder is ManageAI's agentic delivery platform. It takes a natural language project brief, runs it through a multi-agent pipeline (assess, build, verify, deploy), and produces validated Make.com scenario blueprints ready for client handoff. The platform tracks costs, learns from past projects, and self-heals when things go wrong.

v3.0.0 adds API key authentication, a background job queue with worker threads, a webhook event bus, multi-tenant isolation, a full CLI tool, and 206+ tests with end-to-end coverage.

## Architecture

```
Client → [X-API-Key] → FastAPI (79 endpoints)
                            │
              ┌─────────────┼─────────────────┐
              │             │                 │
         Job Queue     Make.com API      Supabase Sync
         (worker)      (deploy/monitor)  (state/notify)
              │             │                 │
              ▼             ▼                 ▼
         Background     Scenarios         Dashboard
         Processing     Webhooks          Alerts
              │             │
              ▼             ▼
         Event Bus → Webhook Subscribers
              │
              ▼
         Learning Loop → TF-IDF Embeddings
```

## Quick Start

```bash
git clone https://github.com/Brian2169fdsa/agenticmakebuilder.git
cd agenticmakebuilder
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"
uvicorn app:app --reload
curl http://localhost:8000/health  # → {"ok": true}
```

## Full Endpoint Reference (79 endpoints)

### Core Pipeline (6)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| POST | `/intake` | Natural language front door |
| POST | `/assess` | Structured intake → delivery report |
| POST | `/plan` | Full pipeline (supports `?async_mode=true`) |
| POST | `/build` | Compiler direct (requires plan_dict) |
| POST | `/audit` | Audit existing blueprint |

### Verification & Confidence (5)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/verify` | 77-rule blueprint validation |
| POST | `/verify/loop` | Iterative verify-fix cycle |
| POST | `/verify/auto` | Auto-verify + pipeline advance |
| GET | `/confidence/history` | Verification run history |
| GET | `/confidence/trend` | Score trend analysis |

### Multi-Agent Orchestration (8)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/orchestrate` | Advance through pipeline stages |
| POST | `/agent/complete` | Agent completion + auto-advance |
| GET | `/pipeline/status` | Full pipeline state view |
| POST | `/pipeline/advance` | Advance to next stage |
| GET | `/pipeline/dashboard` | Projects grouped by stage |
| POST | `/briefing` | Supervisor briefing |
| POST | `/briefing/daily` | Comprehensive daily briefing |
| POST | `/handoff` | Multi-agent handoff bridge |

### Intelligence Layer (2)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/clients/health` | Client health assessment |
| GET | `/clients/list` | Client directory with stats |

### Agent Memory & Learning (4)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/memory` | Store client context |
| GET | `/memory` | Retrieve client context |
| POST | `/memory/embed` | Embed for similarity search |
| GET | `/similar` | TF-IDF cosine similarity |

### Deployment (7)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/deploy/makecom` | Deploy to Make.com (live or simulation) |
| POST | `/deploy/n8n` | Deploy to n8n |
| GET | `/deploy/status` | Deployment status |
| POST | `/deploy/run` | Trigger scenario execution |
| GET | `/deploy/list` | List all deployments |
| POST | `/deploy/teardown` | Deactivate + delete scenario |
| GET | `/deploy/monitor` | Monitor active deployments |

### Webhook Listener (1)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/webhook/makecom` | Receive Make.com execution events |

### Learning Feedback Loop (3)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/learn/outcome` | Record deployment outcome |
| GET | `/learn/insights` | Risk assessment from history |
| GET | `/learn/summary` | Aggregate learning stats |

### Cost & Margin Intelligence (4)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/costs/track` | Track token costs |
| GET | `/costs/summary` | Cost/revenue/margin per client |
| GET | `/costs/report` | Weekly cost report |
| POST | `/costs/estimate` | Pre-build cost estimation |

### Persona Engine (7)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/persona/memory` | Link persona to client |
| GET | `/persona/context` | Persona's client context |
| POST | `/persona/feedback` | Store interaction feedback |
| GET | `/persona/performance` | Persona performance stats |
| POST | `/persona/deploy` | Generate persona artifact |
| POST | `/persona/test` | Test persona via Claude API |
| GET | `/personas/list` | All personas with live stats |

### Authentication (4)
| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/auth/keys` | Create API key (shown once) | Admin |
| GET | `/auth/keys` | List all API keys | Admin |
| DELETE | `/auth/keys/{id}` | Revoke an API key | Admin |
| POST | `/auth/keys/rotate` | Rotate key | Admin |

### Job Queue (6)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/jobs/enqueue` | Enqueue background job |
| GET | `/jobs/list` | List jobs with filters |
| GET | `/jobs/stats` | Queue statistics |
| GET | `/jobs/{job_id}` | Get job status/result |
| DELETE | `/jobs/{id}/cancel` | Cancel pending job |
| POST | `/jobs/cleanup` | Remove old jobs |

### Event Bus (6)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/events/subscribe` | Subscribe to events |
| DELETE | `/events/subscriptions/{id}` | Deactivate subscription |
| GET | `/events/subscriptions` | List subscriptions |
| GET | `/events/deliveries` | List deliveries |
| POST | `/events/test` | Fire test event |
| GET | `/events/types` | List event types |

### Multi-Tenant (7)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tenants` | List all tenants |
| GET | `/tenants/{id}` | Tenant details |
| POST | `/tenants` | Create tenant |
| PATCH | `/tenants/{id}` | Update tenant |
| DELETE | `/tenants/{id}` | Soft-delete tenant |
| GET | `/tenants/{id}/usage` | Tenant usage stats |

### Admin Control Plane (5)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/admin/reset-project` | Reset pipeline stage |
| POST | `/admin/bulk-verify` | Batch verify projects |
| GET | `/admin/system-status` | Platform snapshot |
| POST | `/admin/reindex` | Re-embed all context |
| GET | `/admin/audit-log` | Handoff audit trail |

### Platform Health (3)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health/full` | Comprehensive health check |
| GET | `/health/memory` | Embedding store stats |
| POST | `/health/repair` | Self-healing repair |

### Supervisor (1)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/supervisor/stalled` | Detect stalled projects |

### Natural Language (1)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/command` | Route free-text commands |

## Authentication

Authentication is disabled by default. To enable:

```bash
export AUTH_ENABLED=true
export MASTER_API_KEY=your-secret-master-key
```

Create your first API key:
```bash
curl -X POST http://localhost:8000/auth/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "my-first-key", "tenant_id": "default"}'
```

Use in requests:
```bash
curl -H "X-API-Key: mab_..." http://localhost:8000/health
```

## Job Queue

Long-running operations can be offloaded to background workers:

```bash
# Async plan generation
curl -X POST "http://localhost:8000/plan?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"original_request": "Build webhook", "customer_name": "Acme"}'
# Returns: {"job_id": "...", "status": "pending", "poll_url": "/jobs/..."}

# Poll for result
curl http://localhost:8000/jobs/{job_id}
```

## Event Bus

Subscribe to platform events:
```bash
curl -X POST http://localhost:8000/events/subscribe \
  -H "Content-Type: application/json" \
  -d '{"event_type": "project.deployed", "target_url": "https://your-app.com/webhook"}'
```

Event types: `project.created`, `project.plan_generated`, `project.verified`, `project.deployed`, `project.deploy_failed`, `project.stalled`, `pipeline.advanced`, `execution.success`, `execution.failure`, `cost.threshold_exceeded`, `persona.feedback_received`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | — | PostgreSQL connection string (required) |
| `ANTHROPIC_API_KEY` | — | Claude API key for persona testing |
| `MAKECOM_API_KEY` | — | Make.com API token for live deployment |
| `MAKECOM_TEAM_ID` | — | Make.com team ID |
| `MAKECOM_ORG_ID` | — | Make.com organization ID |
| `MAKECOM_API_BASE` | `https://us1.make.com/api/v2` | Make.com API base |
| `MAKECOM_WEBHOOK_SECRET` | — | Secret for webhook verification |
| `AUTH_ENABLED` | `false` | Enable API key authentication |
| `MASTER_API_KEY` | — | Master key that bypasses all auth |
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | — | Supabase service role key |
| `SLACK_WEBHOOK_URL` | — | Slack webhook for notifications |
| `MANAGEAI_HOURLY_RATE` | `150` | Hourly rate for SOW estimates |
| `MONITOR_INTERVAL` | `900` | Execution poller interval (seconds) |
| `MONITOR_LOOKBACK` | `20` | Recent executions to check |

## Testing

```bash
source .venv/bin/activate
pip install pytest httpx
pytest tests/ -v
# Expected: 206 passed
```

Test files:
- `test_endpoints.py` — Core pipeline (39 tests)
- `test_supabase_sync.py` — Supabase integration (17 tests)
- `test_sprint4.py` — Intelligence layer (16 tests)
- `test_admin.py` — Admin control plane (13 tests)
- `test_makecom_client.py` — Make.com client (12 tests)
- `test_blueprint_builder.py` — Blueprint builder (12 tests)
- `test_deploy_endpoints.py` — Deploy endpoints (15 tests)
- `test_auth.py` — Authentication (13 tests)
- `test_job_queue.py` — Job queue (14 tests)
- `test_event_bus.py` — Event bus (12 tests)
- `test_learning.py` — Learning loop (10 tests)
- `test_tenant.py` — Multi-tenant (11 tests)
- `test_e2e.py` — End-to-end integration (8 tests)

## CLI

```bash
python cli/mab.py --help
python cli/mab.py health
python cli/mab.py keys list
python cli/mab.py jobs
python cli/mab.py tenants
python cli/mab.py admin status
```

Set `MAB_BASE_URL` and `MAB_API_KEY` environment variables for remote access.

## Deployment (Railway)

1. Connect your GitHub repo to [Railway](https://railway.app)
2. Add a PostgreSQL plugin
3. Set `DATABASE_URL` to the Railway Postgres connection string
4. Railway auto-detects Python and runs via nixpacks
5. The app auto-migrates tables on startup
