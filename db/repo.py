"""
Database Repository — Build Persistence Layer

Replaces tools/spec_version_manager.py for production storage.
Three public functions:
    create_build     — upsert project + insert build with atomic version
    store_artifact   — insert a single build artifact row
    finalize_build   — update build row with final status and scores

Advisory lock strategy:
    pg_advisory_xact_lock(hashtext(project_id::text || ':' || slug))
    prevents concurrent version collisions within a single transaction.

Deterministic. No AI reasoning. No conversation context.
"""

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session

from db.models import Project, Build, BuildArtifact


def create_build(db: Session, project_name: str, slug: str,
                 original_request: str, created_at=None) -> Build:
    """Get-or-create project by name, then insert a build row with the
    next atomic version for (project_id, slug).

    Args:
        db: Active SQLAlchemy session (caller manages commit/rollback).
        project_name: Human-readable project name (unique key).
        slug: Scenario slug (kebab-case).
        original_request: User's original natural-language request.
        created_at: Optional datetime; defaults to utcnow.

    Returns:
        The newly created Build ORM instance (id, version populated).
    """
    # Upsert project by name
    project = db.query(Project).filter(Project.name == project_name).first()
    if not project:
        project = Project(name=project_name)
        db.add(project)
        db.flush()

    # Advisory lock scoped to (project_id, slug)
    db.execute(
        text(
            "SELECT pg_advisory_xact_lock("
            "hashtext(:pid || ':' || :slug))"
        ),
        {"pid": str(project.id), "slug": slug},
    )

    # Atomic next version
    result = db.execute(
        text(
            "SELECT COALESCE(MAX(version), 0) + 1 "
            "FROM builds "
            "WHERE project_id = :pid AND slug = :slug"
        ),
        {"pid": str(project.id), "slug": slug},
    )
    version = result.scalar()

    ts = created_at if isinstance(created_at, datetime) else datetime.now(timezone.utc)

    build = Build(
        project_id=project.id,
        slug=slug,
        version=version,
        original_request=original_request,
        created_at=ts,
    )
    db.add(build)
    db.flush()
    return build


def store_artifact(db: Session, build_id, artifact_type: str,
                   content_json=None, content_text=None):
    """Insert a single build artifact.

    Exactly one of content_json / content_text should be provided.

    Args:
        db: Active SQLAlchemy session.
        build_id: UUID of the parent build.
        artifact_type: Artifact kind string (e.g. 'canonical_spec').
        content_json: Dict/list stored as JSONB.
        content_text: Plain text stored as TEXT.
    """
    artifact = BuildArtifact(
        build_id=build_id,
        artifact_type=artifact_type,
        content_json=content_json,
        content_text=content_text,
    )
    db.add(artifact)
    db.flush()


def finalize_build(db: Session, build_id, status: str, *,
                   confidence_score=None, confidence_grade=None,
                   canonical_valid=None, export_valid=None,
                   heal_attempts=0, failure_reason=None):
    """Update a build row with its final status and scores.

    Args:
        db: Active SQLAlchemy session.
        build_id: UUID of the build to finalize.
        status: One of 'success', 'failed'.
        confidence_score: Float 0.0–1.0.
        confidence_grade: Letter grade (A/B/C/D/F).
        canonical_valid: Whether canonical spec passed validation.
        export_valid: Whether Make export passed validation.
        heal_attempts: Number of self-heal retries used.
        failure_reason: Human-readable failure description.
    """
    build = db.query(Build).filter(Build.id == build_id).one()
    build.status = status
    build.confidence_score = confidence_score
    build.confidence_grade = confidence_grade
    build.canonical_valid = canonical_valid
    build.export_valid = export_valid
    build.heal_attempts = heal_attempts
    build.failure_reason = failure_reason
    db.flush()
