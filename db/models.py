"""
SQLAlchemy ORM Models
Maps to the PostgreSQL schema in db/schema.sql.
Tables: projects, builds, build_artifacts, assumptions,
        persona_client_context, persona_feedback.
All enum-like columns use plain TEXT — no PostgreSQL enum types.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, Integer, Float, Boolean, Text,
    DateTime, ForeignKey, UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False, unique=True)
    customer_name = Column(Text, default="Unknown")
    status = Column(Text, nullable=False, default="active")
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    revenue = Column(Float, default=0.0)

    builds = relationship(
        "Build", back_populates="project", cascade="all, delete-orphan",
    )


class Build(Base):
    __tablename__ = "builds"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    slug = Column(Text, nullable=False)
    version = Column(Integer, nullable=False)
    original_request = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    confidence_score = Column(Float)
    confidence_grade = Column(Text)
    canonical_valid = Column(Boolean)
    export_valid = Column(Boolean)
    heal_attempts = Column(Integer, nullable=False, default=0)
    status = Column(Text, nullable=False, default="running")
    failure_reason = Column(Text)

    project = relationship("Project", back_populates="builds")
    artifacts = relationship(
        "BuildArtifact", back_populates="build", cascade="all, delete-orphan",
    )
    assumptions_list = relationship(
        "Assumption", back_populates="build", cascade="all, delete-orphan",
    )

    __table_args__ = (
        UniqueConstraint(
            "project_id", "slug", "version",
            name="uq_builds_project_slug_version",
        ),
    )


class BuildArtifact(Base):
    __tablename__ = "build_artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    build_id = Column(
        UUID(as_uuid=True),
        ForeignKey("builds.id", ondelete="CASCADE"),
        nullable=False,
    )
    artifact_type = Column(Text, nullable=False)
    content_json = Column(JSONB)
    content_text = Column(Text)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    build = relationship("Build", back_populates="artifacts")

    __table_args__ = (
        UniqueConstraint(
            "build_id", "artifact_type",
            name="uq_build_artifacts_build_type",
        ),
    )


class Assumption(Base):
    __tablename__ = "assumptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    build_id = Column(
        UUID(as_uuid=True),
        ForeignKey("builds.id", ondelete="CASCADE"),
        nullable=False,
    )
    type = Column(Text, nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(Text, nullable=False, default="low")
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    build = relationship("Build", back_populates="assumptions_list")


class AgentHandoff(Base):
    __tablename__ = "agent_handoffs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    from_agent = Column(Text, nullable=False)
    to_agent = Column(Text, nullable=False)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    context_bundle = Column(JSONB)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    project = relationship("Project")


class ProjectFinancial(Base):
    __tablename__ = "project_financials"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    model = Column(Text, nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    operation_type = Column(Text, nullable=False)
    cost_usd = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class ProjectAgentState(Base):
    __tablename__ = "project_agent_state"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    current_stage = Column(Text, nullable=False, default="intake")
    current_agent = Column(Text)
    started_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    pipeline_health = Column(Text, nullable=False, default="on_track")
    stage_history = Column(JSONB, default=list)

    project = relationship("Project")


class ClientContext(Base):
    __tablename__ = "client_context"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(Text, nullable=False)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    key_decisions = Column(JSONB)
    tech_stack = Column(JSONB)
    failure_patterns = Column(JSONB)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    project = relationship("Project")

    __table_args__ = (
        UniqueConstraint(
            "client_id", "project_id",
            name="uq_client_context_client_project",
        ),
    )


class VerificationRun(Base):
    __tablename__ = "verification_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    confidence_score = Column(Float, nullable=False)
    passed = Column(Boolean, nullable=False)
    error_count = Column(Integer, nullable=False, default=0)
    warning_count = Column(Integer, nullable=False, default=0)
    fix_instructions = Column(JSONB)
    iteration = Column(Integer, nullable=False, default=1)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    project = relationship("Project")


class Deployment(Base):
    __tablename__ = "deployments"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    target = Column(Text, nullable=False)
    external_id = Column(Text)
    external_url = Column(Text)
    status = Column(Text, nullable=False, default="pending")
    last_health_check = Column(JSONB)
    deployed_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    project = relationship("Project")


class PersonaClientContext(Base):
    __tablename__ = "persona_client_context"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    persona = Column(Text, nullable=False)
    client_id = Column(Text, nullable=False)
    tone_preferences = Column(JSONB)
    past_interactions_summary = Column(Text)
    communication_style = Column(Text)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "persona", "client_id",
            name="uq_persona_client_context_persona_client",
        ),
    )


class PersonaFeedback(Base):
    __tablename__ = "persona_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    persona = Column(Text, nullable=False)
    client_id = Column(Text, nullable=False)
    interaction_id = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)
    notes = Column(Text)
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )