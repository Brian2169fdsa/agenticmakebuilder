-- Agentic Make Builder — PostgreSQL Schema
-- Version: 2.0.0
-- Phase 1: Persistent storage replacing filesystem artifacts
--
-- Tables: projects, builds, build_artifacts, assumptions
-- All UUIDs generated server-side via gen_random_uuid (pgcrypto).
-- Version increments are atomic per (project_id, slug) via advisory lock
-- executed in db/repo.py — no server-side function required.

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- PROJECTS
-- ============================================================

CREATE TABLE projects (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ============================================================
-- BUILDS
-- ============================================================

CREATE TABLE builds (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id          UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    slug                TEXT NOT NULL,
    version             INTEGER NOT NULL,
    original_request    TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    confidence_score    DOUBLE PRECISION,
    confidence_grade    TEXT,
    canonical_valid     BOOLEAN,
    export_valid        BOOLEAN,
    heal_attempts       INTEGER NOT NULL DEFAULT 0,
    status              TEXT NOT NULL DEFAULT 'running',
    failure_reason      TEXT,

    UNIQUE (project_id, slug, version)
);

CREATE INDEX idx_builds_project_id ON builds (project_id);
CREATE INDEX idx_builds_slug_version ON builds (slug, version);
CREATE INDEX idx_builds_status ON builds (status);

-- ============================================================
-- BUILD ARTIFACTS
-- ============================================================

CREATE TABLE build_artifacts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id        UUID NOT NULL REFERENCES builds(id) ON DELETE CASCADE,
    artifact_type   TEXT NOT NULL,
    content_json    JSONB,
    content_text    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (build_id, artifact_type)
);

CREATE INDEX idx_build_artifacts_build_id ON build_artifacts (build_id);

-- ============================================================
-- ASSUMPTIONS
-- ============================================================

CREATE TABLE assumptions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id        UUID NOT NULL REFERENCES builds(id) ON DELETE CASCADE,
    type            TEXT NOT NULL,
    description     TEXT NOT NULL,
    severity        TEXT NOT NULL DEFAULT 'low',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_assumptions_build_id ON assumptions (build_id);
