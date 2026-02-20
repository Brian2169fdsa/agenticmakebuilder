-- ManageAI Agentic Builder — Schema v4.0.0
-- Run: psql $DATABASE_URL < db/schema.sql

-- ── Core tables ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS builds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending',
    confidence_score FLOAT,
    confidence_grade TEXT,
    heal_attempts INTEGER DEFAULT 0,
    output_path TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(slug, version)
);

CREATE TABLE IF NOT EXISTS build_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID REFERENCES builds(id) ON DELETE CASCADE,
    artifact_type TEXT NOT NULL,
    content_json JSONB,
    content_text TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS assumptions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID REFERENCES builds(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'low',
    note TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ── Audit tables ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS audit_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL,
    customer_name TEXT,
    verdict TEXT NOT NULL,
    confidence_score FLOAT,
    error_count INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    delta_confidence FLOAT,
    regression BOOLEAN DEFAULT false,
    audited_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_runs_slug ON audit_runs(slug);
CREATE INDEX IF NOT EXISTS idx_audit_runs_audited_at ON audit_runs(audited_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_runs_regression ON audit_runs(regression) WHERE regression = true;

-- ── SOW documents ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS sow_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID REFERENCES builds(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    customer_name TEXT,
    generated_at TIMESTAMPTZ DEFAULT now(),
    sow_text TEXT,
    exec_summary_text TEXT,
    tech_spec_text TEXT,
    checklist_text TEXT,
    total_investment FLOAT,
    recommended_plan TEXT
);

CREATE INDEX IF NOT EXISTS idx_sow_slug ON sow_documents(slug, version);

-- ── Integration tests ──────────────────────────────────────────

CREATE TABLE IF NOT EXISTS test_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID REFERENCES builds(id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    version INTEGER NOT NULL,
    ran_at TIMESTAMPTZ DEFAULT now(),
    dry_run BOOLEAN DEFAULT true,
    total_tests INTEGER DEFAULT 0,
    passed INTEGER DEFAULT 0,
    failed INTEGER DEFAULT 0,
    pass_rate FLOAT,
    confidence_bonus FLOAT DEFAULT 0,
    report_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_test_runs_slug ON test_runs(slug, version);

-- ── Incidents ──────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL,
    customer_name TEXT,
    incident_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    detected_at TIMESTAMPTZ DEFAULT now(),
    acknowledged_at TIMESTAMPTZ,
    resolved_at TIMESTAMPTZ,
    status TEXT DEFAULT 'open',
    details JSONB,
    resolution_note TEXT
);

CREATE INDEX IF NOT EXISTS idx_incidents_slug ON incidents(slug);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity) WHERE status = 'open';

-- ── Execution snapshots (live monitor) ────────────────────────

CREATE TABLE IF NOT EXISTS execution_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT NOT NULL,
    customer_name TEXT,
    snapped_at TIMESTAMPTZ DEFAULT now(),
    executions_checked INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    success_rate FLOAT,
    dominant_error_type TEXT,
    needs_attention BOOLEAN DEFAULT false
);

CREATE INDEX IF NOT EXISTS idx_snapshots_slug ON execution_snapshots(slug, snapped_at DESC);

-- ── Client snapshots (intelligence layer) ────────────────────

CREATE TABLE IF NOT EXISTS client_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_name TEXT NOT NULL,
    snapshot_date DATE NOT NULL,
    total_scenarios INTEGER DEFAULT 0,
    healthy_scenarios INTEGER DEFAULT 0,
    total_monthly_ops INTEGER DEFAULT 0,
    avg_confidence FLOAT,
    risk_score FLOAT DEFAULT 0,
    opportunity_score FLOAT DEFAULT 0,
    open_incidents INTEGER DEFAULT 0,
    details JSONB,
    UNIQUE(customer_name, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_client_snapshots_name ON client_snapshots(customer_name, snapshot_date DESC);

-- ── Agent handoffs (multi-agent orchestration) ──────────────

CREATE TABLE IF NOT EXISTS agent_handoffs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_agent TEXT NOT NULL,
    to_agent TEXT NOT NULL,
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    context_bundle JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_handoffs_project ON agent_handoffs(project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_handoffs_to_agent ON agent_handoffs(to_agent, created_at DESC);

-- ── Project financials (token cost tracking) ────────────────

CREATE TABLE IF NOT EXISTS project_financials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    operation_type TEXT NOT NULL,
    cost_usd FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_financials_project ON project_financials(project_id, created_at DESC);
