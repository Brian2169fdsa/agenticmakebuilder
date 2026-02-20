"""
Database Session Management

SQLAlchemy engine + sessionmaker. Reads DATABASE_URL from environment.
Provides get_db() generator for FastAPI Depends injection.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://localhost:5432/agenticmakebuilder",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"connect_timeout": 5},
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db():
    """FastAPI dependency — yields a session, closes on teardown."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_db():
    """Verify database connectivity. Logs warning on failure instead of crashing."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("✓ Database connected")
        _run_migrations()
    except Exception as e:
        print(f"⚠ Database not reachable (app will still start): {e}")


def _run_migrations():
    """Apply lightweight schema migrations (idempotent)."""
    migrations = [
        # projects table
        "ALTER TABLE projects ADD COLUMN IF NOT EXISTS customer_name TEXT DEFAULT 'Unknown'",
        "ALTER TABLE projects ADD COLUMN IF NOT EXISTS tenant_id TEXT DEFAULT 'default'",
        # project_agent_state table
        "ALTER TABLE project_agent_state ADD COLUMN IF NOT EXISTS current_stage TEXT NOT NULL DEFAULT 'intake'",
        "ALTER TABLE project_agent_state ADD COLUMN IF NOT EXISTS current_agent TEXT",
        "ALTER TABLE project_agent_state ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ DEFAULT now()",
        "ALTER TABLE project_agent_state ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now()",
        "ALTER TABLE project_agent_state ADD COLUMN IF NOT EXISTS pipeline_health TEXT NOT NULL DEFAULT 'on_track'",
        "ALTER TABLE project_agent_state ADD COLUMN IF NOT EXISTS stage_history JSONB DEFAULT '[]'::jsonb",
        # verification_runs table
        "ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS confidence_score FLOAT",
        "ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS issues_found INTEGER DEFAULT 0",
        "ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS passed BOOLEAN DEFAULT false",
        "ALTER TABLE verification_runs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()",
        # agent_handoffs table
        "ALTER TABLE agent_handoffs ADD COLUMN IF NOT EXISTS from_agent TEXT",
        "ALTER TABLE agent_handoffs ADD COLUMN IF NOT EXISTS to_agent TEXT",
        "ALTER TABLE agent_handoffs ADD COLUMN IF NOT EXISTS context_bundle JSONB DEFAULT '{}'::jsonb",
        "ALTER TABLE agent_handoffs ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT now()",
        # deployments table
        "ALTER TABLE deployments ADD COLUMN IF NOT EXISTS scenario_name TEXT",
        "ALTER TABLE deployments ADD COLUMN IF NOT EXISTS scenario_id TEXT",
    ]

    # DDL for new v3 tables (CREATE TABLE IF NOT EXISTS is idempotent)
    v3_tables = [
        """CREATE TABLE IF NOT EXISTS api_keys (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key_hash TEXT NOT NULL UNIQUE,
            key_prefix TEXT NOT NULL,
            name TEXT NOT NULL,
            tenant_id TEXT NOT NULL DEFAULT 'default',
            permissions JSONB DEFAULT '["read","write"]'::jsonb,
            rate_limit_per_minute INTEGER DEFAULT 60,
            created_at TIMESTAMPTZ DEFAULT now(),
            last_used_at TIMESTAMPTZ,
            expires_at TIMESTAMPTZ,
            revoked BOOLEAN DEFAULT false,
            created_by TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)",
        "CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id)",
        """CREATE TABLE IF NOT EXISTS job_queue (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            result JSONB,
            error_message TEXT,
            tenant_id TEXT DEFAULT 'default',
            project_id TEXT,
            priority INTEGER DEFAULT 5,
            created_at TIMESTAMPTZ DEFAULT now(),
            started_at TIMESTAMPTZ,
            completed_at TIMESTAMPTZ,
            retry_count INTEGER DEFAULT 0,
            max_retries INTEGER DEFAULT 3
        )""",
        "CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status)",
        "CREATE INDEX IF NOT EXISTS idx_job_queue_type ON job_queue(job_type)",
        "CREATE INDEX IF NOT EXISTS idx_job_queue_created ON job_queue(created_at DESC)",
        """CREATE TABLE IF NOT EXISTS learning_outcomes (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            project_id TEXT NOT NULL,
            scenario_id TEXT,
            outcome TEXT NOT NULL,
            execution_count INTEGER DEFAULT 0,
            avg_duration_ms INTEGER,
            error_patterns TEXT[] DEFAULT '{}',
            learning_summary TEXT,
            embedding_id TEXT,
            created_at TIMESTAMPTZ DEFAULT now()
        )""",
        "CREATE INDEX IF NOT EXISTS idx_learning_project ON learning_outcomes(project_id)",
        "CREATE INDEX IF NOT EXISTS idx_learning_outcome ON learning_outcomes(outcome)",
        """CREATE TABLE IF NOT EXISTS webhook_subscriptions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            event_type TEXT NOT NULL,
            target_url TEXT NOT NULL,
            secret TEXT,
            tenant_id TEXT DEFAULT 'default',
            active BOOLEAN DEFAULT true,
            created_at TIMESTAMPTZ DEFAULT now(),
            last_triggered_at TIMESTAMPTZ,
            failure_count INTEGER DEFAULT 0
        )""",
        """CREATE TABLE IF NOT EXISTS webhook_deliveries (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            subscription_id UUID,
            event_type TEXT NOT NULL,
            payload JSONB NOT NULL,
            response_status INTEGER,
            response_body TEXT,
            delivered_at TIMESTAMPTZ DEFAULT now(),
            success BOOLEAN DEFAULT false,
            duration_ms INTEGER
        )""",
        "CREATE INDEX IF NOT EXISTS idx_webhook_subs_event ON webhook_subscriptions(event_type)",
        "CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_sub ON webhook_deliveries(subscription_id)",
        """CREATE TABLE IF NOT EXISTS tenants (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            plan TEXT NOT NULL DEFAULT 'standard',
            rate_limit_per_minute INTEGER DEFAULT 60,
            max_projects INTEGER DEFAULT 50,
            max_api_keys INTEGER DEFAULT 5,
            features JSONB DEFAULT '["plan","verify","memory","persona"]'::jsonb,
            created_at TIMESTAMPTZ DEFAULT now(),
            active BOOLEAN DEFAULT true
        )""",
        "INSERT INTO tenants (id, name, plan) VALUES ('default', 'Default Tenant', 'enterprise') ON CONFLICT DO NOTHING",
    ]
    migrations.extend(v3_tables)
    try:
        with engine.begin() as conn:
            for sql in migrations:
                conn.execute(text(sql))
        print("✓ Migrations applied")
    except Exception as e:
        print(f"⚠ Migration warning (non-fatal): {e}")
