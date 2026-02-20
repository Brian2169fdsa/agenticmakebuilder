"""
Authentication — API key management and validation.

Provides key generation, hashing, validation, and FastAPI dependencies.
Auth is opt-in: set AUTH_ENABLED=true to enforce API key checks.
"""

import hashlib
import os
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from sqlalchemy import text

from db.session import engine

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
MASTER_KEY = os.getenv("MASTER_API_KEY", "")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"


def hash_key(raw_key: str) -> str:
    """SHA-256 hash of the raw API key."""
    return hashlib.sha256(raw_key.encode()).hexdigest()


def generate_api_key(prefix: str = "mab") -> tuple:
    """Generate a new API key. Returns (raw_key, key_hash)."""
    raw = f"{prefix}_{secrets.token_urlsafe(32)}"
    return raw, hash_key(raw)


async def get_api_key(
    x_api_key: Optional[str] = Security(API_KEY_HEADER),
) -> dict:
    """FastAPI dependency — validates API key and returns key metadata.

    When AUTH_ENABLED is false, returns a default permissive record.
    When AUTH_ENABLED is true, validates the key against the api_keys table.
    MASTER_API_KEY bypasses all checks.
    """
    if not AUTH_ENABLED:
        return {
            "tenant_id": "default",
            "permissions": ["read", "write", "admin"],
            "key_prefix": "dev",
        }

    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Pass X-API-Key header.",
        )

    if MASTER_KEY and x_api_key == MASTER_KEY:
        return {
            "tenant_id": "master",
            "permissions": ["read", "write", "admin"],
            "key_prefix": "master",
        }

    kh = hash_key(x_api_key)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT tenant_id, permissions, key_prefix, rate_limit_per_minute "
                    "FROM api_keys "
                    "WHERE key_hash = :hash AND revoked = false "
                    "AND (expires_at IS NULL OR expires_at > now())"
                ),
                {"hash": kh},
            ).fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Invalid or expired API key.")
            conn.execute(
                text("UPDATE api_keys SET last_used_at = now() WHERE key_hash = :hash"),
                {"hash": kh},
            )
            conn.commit()
            return {
                "tenant_id": row.tenant_id,
                "permissions": list(row.permissions) if row.permissions else ["read", "write"],
                "key_prefix": row.key_prefix,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auth check failed: {e}")


def require_permission(permission: str):
    """Returns a FastAPI dependency that checks for a specific permission."""

    async def checker(key_data: dict = Security(get_api_key)):
        if permission not in key_data.get("permissions", []):
            raise HTTPException(
                status_code=403,
                detail=f"Permission '{permission}' required.",
            )
        return key_data

    return checker
