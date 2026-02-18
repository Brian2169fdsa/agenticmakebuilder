"""
SQLAlchemy ORM Models
Maps to the PostgreSQL schema in db/schema.sql.
Tables: projects, builds, build_artifacts, assumptions.
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
    created_at = Column(
        DateTime(timezone=True), nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

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