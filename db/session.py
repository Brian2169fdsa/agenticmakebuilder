"""
Database Session Management

SQLAlchemy engine + sessionmaker. Reads DATABASE_URL from environment.
Provides get_db() generator for FastAPI Depends injection.
"""

import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://localhost:5432/agenticmakebuilder",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
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
    """Verify database connectivity. Raises on failure."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
