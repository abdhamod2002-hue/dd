"""SQLAlchemy ORM models for the AI Littering Detection system.

Tables
------
Camera  : a CCTV camera source.
Event   : a confirmed littering event reported by the inference engine.
Evidence: snapshot + short video clip backing an event.
User    : an operator / reviewer account.

Relationships: Camera 1->* Events, Event 1->* Evidence.

Uses SQLAlchemy 2.0 ``Mapped`` / ``mapped_column`` typing, compatible with
Python 3.9 (annotations are evaluated lazily via ``from __future__``).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    location: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    events: Mapped[List["Event"]] = relationship(
        "Event", back_populates="camera", cascade="all, delete-orphan"
    )


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False, index=True
    )
    person_track_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    object_track_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    object_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="confirmed")
    analysis_job_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    # Phase‑C stable identifiers (event‑centric)
    event_actor_person_track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    event_actor_person_uid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    event_object_track_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    event_object_uid: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    camera: Mapped[Optional["Camera"]] = relationship("Camera", back_populates="events")
    evidence: Mapped[List["Evidence"]] = relationship(
        "Evidence", back_populates="event", cascade="all, delete-orphan"
    )


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    event_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("events.id", ondelete="CASCADE"), nullable=False, index=True
    )
    image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    person_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    waste_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    clip_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    face_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # P1-2: temporal sequence stills (carry -> release -> ground) cut from the
    # original video at the FSM's recorded frames.
    carry_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    release_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    ground_image_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    event: Mapped[Optional["Event"]] = relationship("Event", back_populates="evidence")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    role: Mapped[str] = mapped_column(String(64), nullable=False, default="operator")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class VideoAnalysisJob(Base):
    """Represents an uploaded video analysis job processed by the production AI pipeline."""
    __tablename__ = "video_analysis_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")  # queued, processing, completed, failed
    duration_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processed_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processing_fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    events_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    persons_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    objects_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    report_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON diagnostic report
    analyzed_video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    # --- Persistent analysis archive fields (additive, nullable) ---
    # analysis_id is simply `id`; exposed as a property for API/JSON clarity.
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    # Portable (repo-root-relative) path to the uploaded original video, so the
    # job remains viewable inside the container after a restart — absolute
    # host paths do not resolve under /app.
    original_video_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # Per-analysis artifact manifest: original/analyzed/frames/events + file
    # sizes, written at completion time and keyed to this analysis_id.
    manifest_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    @property
    def analysis_id(self) -> int:
        # Alias of the primary key; clients may key archive requests/results by
        # analysis_id, which is stable for the lifetime of this record.
        return self.id
