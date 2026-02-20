"""
Supabase REST Client — Pure httpx, no supabase-py dependency.

Connects agenticmakebuilder to the connect-hub Supabase instance for
real-time sync of pipeline state, costs, and agent actions.

Reads SUPABASE_URL and SUPABASE_SERVICE_KEY from environment.
Falls back gracefully (returns None/empty) if not configured.
"""

import os
from typing import Optional

import httpx


class SupabaseError(Exception):
    """Raised when a Supabase REST API call fails."""

    def __init__(self, message: str, status_code: int = 0, body: str = ""):
        self.status_code = status_code
        self.body = body
        super().__init__(message)


class SupabaseClient:
    """Lightweight Supabase REST client using httpx."""

    def __init__(self, base_url: str, service_key: str):
        self.base_url = base_url.rstrip("/")
        self.rest_url = f"{self.base_url}/rest/v1"
        self.headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def select(
        self,
        table: str,
        filters: Optional[dict] = None,
        columns: str = "*",
        limit: Optional[int] = None,
    ) -> list[dict]:
        """SELECT rows from a table with optional eq filters.

        Args:
            table: Table name.
            filters: Dict of {column: value} for eq filters.
            columns: Comma-separated column list (default "*").
            limit: Max rows to return.

        Returns:
            List of row dicts.
        """
        params = {"select": columns}
        if filters:
            for col, val in filters.items():
                params[col] = f"eq.{val}"
        if limit:
            params["limit"] = str(limit)

        resp = httpx.get(
            f"{self.rest_url}/{table}",
            headers=self.headers,
            params=params,
            timeout=10,
        )
        if resp.status_code >= 400:
            raise SupabaseError(
                f"SELECT from {table} failed: {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text,
            )
        return resp.json()

    def insert(self, table: str, data: dict | list[dict]) -> list[dict]:
        """INSERT one or more rows into a table.

        Args:
            table: Table name.
            data: Single dict or list of dicts to insert.

        Returns:
            List of inserted row dicts.
        """
        payload = data if isinstance(data, list) else [data]

        resp = httpx.post(
            f"{self.rest_url}/{table}",
            headers=self.headers,
            json=payload,
            timeout=10,
        )
        if resp.status_code >= 400:
            raise SupabaseError(
                f"INSERT into {table} failed: {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text,
            )
        return resp.json()

    def update(self, table: str, filters: dict, data: dict) -> list[dict]:
        """UPDATE rows matching eq filters.

        Args:
            table: Table name.
            filters: Dict of {column: value} for eq filters.
            data: Dict of columns to update.

        Returns:
            List of updated row dicts.
        """
        params = {}
        for col, val in filters.items():
            params[col] = f"eq.{val}"

        resp = httpx.patch(
            f"{self.rest_url}/{table}",
            headers=self.headers,
            params=params,
            json=data,
            timeout=10,
        )
        if resp.status_code >= 400:
            raise SupabaseError(
                f"UPDATE {table} failed: {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text,
            )
        return resp.json()

    def upsert(
        self, table: str, data: dict | list[dict], on_conflict: str = "id"
    ) -> list[dict]:
        """UPSERT (insert or update on conflict) rows.

        Args:
            table: Table name.
            data: Single dict or list of dicts.
            on_conflict: Comma-separated conflict columns.

        Returns:
            List of upserted row dicts.
        """
        payload = data if isinstance(data, list) else [data]
        headers = {
            **self.headers,
            "Prefer": "return=representation,resolution=merge-duplicates",
        }

        resp = httpx.post(
            f"{self.rest_url}/{table}",
            headers=headers,
            params={"on_conflict": on_conflict},
            json=payload,
            timeout=10,
        )
        if resp.status_code >= 400:
            raise SupabaseError(
                f"UPSERT into {table} failed: {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text,
            )
        return resp.json()


# ── Singleton ─────────────────────────────────────────────────

_client: Optional[SupabaseClient] = None


def get_supabase_client() -> Optional[SupabaseClient]:
    """Return shared SupabaseClient instance, or None if not configured."""
    global _client
    if _client is not None:
        return _client

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()

    if not url or not key:
        return None

    _client = SupabaseClient(base_url=url, service_key=key)
    return _client
