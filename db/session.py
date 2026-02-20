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
    ]
    try:
        with engine.begin() as conn:
            for sql in migrations:
                conn.execute(text(sql))
        print("✓ Migrations applied")
    except Exception as e:
        print(f"⚠ Migration warning (non-fatal): {e}")
