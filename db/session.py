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
    except Exception as e:
        print(f"⚠ Database not reachable (app will still start): {e}")
