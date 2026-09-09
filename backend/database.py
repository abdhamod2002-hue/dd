"""Database configuration for the AI Littering Detection backend.

Provides a lazily-created SQLAlchemy 2.0 engine, a session factory, a declarative
``Base`` class, a FastAPI ``get_db`` dependency, and a ``create_all`` helper that
imports the models and creates every table.

The engine is created lazily (on first attribute access) so that simply importing
this module — and therefore ``backend.main`` — never touches the database. This
keeps the package importable in environments without a running PostgreSQL server.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql://litter:litter@localhost:5432/littering"

DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


class Base(DeclarativeBase):
    """Declarative base shared by all ORM models."""


# The engine is created lazily so importing the package never connects.
# ``connect_args`` / ``pool_pre_ping`` keep things friendly for dev Postgres.
_engine = None
_SessionLocal: "sessionmaker | None" = None


def _build_engine():
    """Construct and cache the engine + session factory on first use."""
    global _engine, _SessionLocal
    if _engine is None:
        # SQLite (used in some tests) needs check_same_thread=False.
        connect_args = (
            {"check_same_thread": False}
            if DATABASE_URL.startswith("sqlite")
            else {}
        )
        _engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args=connect_args,
            future=True,
        )
        _SessionLocal = sessionmaker(
            bind=_engine,
            autocommit=False,
            autoflush=False,
            class_=Session,
            future=True,
        )
    return _engine


@property
def _lazy_engine(self):  # pragma: no cover - unused, placeholder for clarity
    raise RuntimeError("Use get_engine() instead")


def get_engine():
    """Return the cached engine, creating it on first call."""
    return _build_engine()


def get_session_factory() -> sessionmaker:
    """Return the cached session factory, creating the engine if needed."""
    _build_engine()
    assert _SessionLocal is not None  # noqa: S101 - set by _build_engine
    return _SessionLocal


# Backwards-compatible module-level attributes that trigger lazy creation.
def __getattr__(name: str):  # PEP 562 module-level __getattr__
    if name == "engine":
        return get_engine()
    if name == "SessionLocal":
        return get_session_factory()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and closes it."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def create_all() -> None:
    """Create all tables registered on ``Base``.

    Imports the models package so that every table is mapped on ``Base.metadata``
    before issuing ``CREATE TABLE``. Safe to call at application startup.
    """
    # Import here to avoid a circular import at module load time.
    from backend import models  # noqa: F401 - registers tables on Base

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_report_json_text_column(engine)
    _ensure_analysis_job_columns(engine)
    _ensure_event_and_evidence_columns(engine)


def _ensure_report_json_text_column(engine) -> None:
    """Best-effort migration for older varchar(4096) analysis reports."""
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(engine)
        if "video_analysis_jobs" not in inspector.get_table_names():
            return
        dialect = engine.dialect.name
        if dialect == "postgresql":
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE video_analysis_jobs ALTER COLUMN report_json TYPE TEXT"))
        # SQLite does not enforce VARCHAR length, so no migration is needed.
    except Exception:
        # Startup must not fail because an optional convenience migration is
        # unsupported by the current DB permissions/version.
        return


def _ensure_event_and_evidence_columns(engine) -> None:
    """Best-effort migration for event-review columns added after first release, plus new event‑centric identifiers (Phase D)."""

    """Best-effort migration for event-review columns added after first release."""
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(engine)
        dialect = engine.dialect.name
        table_names = set(inspector.get_table_names())

        if "events" in table_names:
            existing = {col["name"] for col in inspector.get_columns("events")}
            # Ensure analysis_job_id exists (legacy column)
            if "analysis_job_id" not in existing:
                col_type = "INTEGER" if dialect == "postgresql" else "INTEGER"
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE events ADD COLUMN analysis_job_id {col_type}"))
            # New event‑centric identifiers (Phase D) - must include stable person uid
            for col_name in ("event_actor_person_track_id", "event_actor_person_uid", "event_object_track_id", "event_object_uid"):
                if col_name not in existing:
                    col_type = "INTEGER" if dialect == "postgresql" else "INTEGER"
                    with engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE events ADD COLUMN {col_name} {col_type}"))

        if "evidence" in table_names:
            existing = {col["name"] for col in inspector.get_columns("evidence")}
            for name in ("person_image_path", "waste_image_path", "clip_path", "face_image_path",
                         # P1-2: temporal sequence stills
                         "carry_image_path", "release_image_path", "ground_image_path"):
                if name not in existing:
                    col_type = "VARCHAR(512)" if dialect == "postgresql" else "TEXT"
                    with engine.begin() as conn:
                        conn.execute(text(f"ALTER TABLE evidence ADD COLUMN {name} {col_type}"))
    except Exception:
        return
def _ensure_analysis_job_columns(engine) -> None:
    """Best-effort migration for analysis-job columns added after first release."""
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(engine)
        if "video_analysis_jobs" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("video_analysis_jobs")}
        dialect = engine.dialect.name
        if "analyzed_video_path" not in existing:
            col_type = "VARCHAR(512)" if dialect == "postgresql" else "TEXT"
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE video_analysis_jobs ADD COLUMN analyzed_video_path {col_type}"))
        # Persistent-analysis-archive columns (additive, idempotent).
        # - started_at           : TIMESTAMP with timezone when processing began
        # - original_video_path  : repo-root-relative original upload path
        # - manifest_json        : full per-analysis artifact manifest (Text)
        if dialect == "postgresql":
            col_type_map = {
                "started_at": "TIMESTAMP",
                "original_video_path": "VARCHAR(512)",
                "manifest_json": "TEXT",
            }
        else:
            col_type_map = {
                "started_at": "TIMESTAMP",
                "original_video_path": "TEXT",
                "manifest_json": "TEXT",
            }
        for col, col_type in col_type_map.items():
            if col not in existing:
                with engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE video_analysis_jobs ADD COLUMN {col} {col_type}"))
    except Exception:
        return
