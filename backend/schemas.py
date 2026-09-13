"""Pydantic v2 schemas for request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Camera
# --------------------------------------------------------------------------- #
class CameraBase(BaseModel):
    name: str = Field(..., max_length=255)
    location: Optional[str] = Field(None, max_length=255)
    status: str = Field("active", max_length=64)


class CameraCreate(CameraBase):
    pass


class CameraOut(CameraBase):
    id: int
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Event
# --------------------------------------------------------------------------- #
class EventBase(BaseModel):
    camera_id: int
    person_track_id: Optional[str] = Field(None, max_length=64)
    object_track_id: Optional[str] = Field(None, max_length=64)
    object_type: str = Field(..., max_length=64)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    status: str = Field("confirmed", max_length=64)
    timestamp: Optional[datetime] = None
    # Phase C — stable actor/object identity frozen at carry time by the event
    # detector (MASTER_REPAIR_PLAN P0-1 root cause A). Without these fields the
    # live-camera path structurally cannot populate the DB's stable-identity
    # columns, so they stayed NULL for 100% of events reported via
    # POST /api/events.
    event_actor_person_track_id: Optional[int] = None
    event_actor_person_uid: Optional[int] = None
    event_object_track_id: Optional[int] = None
    event_object_uid: Optional[int] = None


class EventCreate(EventBase):
    """Payload used by the inference engine to report a confirmed littering event."""

    pass


class EventOut(EventBase):
    id: int
    analysis_job_id: Optional[int] = None
    created_at: Optional[datetime] = None
    # New event‑centric identifiers (Phase D) — authoritative actor/object
    event_actor_person_track_id: Optional[int] = None
    event_actor_person_uid: Optional[int] = None
    event_object_track_id: Optional[int] = None
    event_object_uid: Optional[int] = None

    model_config = ConfigDict(from_attributes=True, extra="ignore")


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #
class EvidenceOut(BaseModel):
    id: int
    event_id: int
    image_path: Optional[str] = None
    video_path: Optional[str] = None
    person_image_path: Optional[str] = None
    waste_image_path: Optional[str] = None
    clip_path: Optional[str] = None
    face_image_path: Optional[str] = None
    # P1-2: carry/release/ground sequence stills
    carry_image_path: Optional[str] = None
    release_image_path: Optional[str] = None
    ground_image_path: Optional[str] = None
    duration_sec: Optional[float] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class EvidenceUploadResponse(BaseModel):
    """Returned after a snapshot + video upload."""

    evidence: EvidenceOut
    snapshot_filename: Optional[str] = None
    video_filename: Optional[str] = None


# --------------------------------------------------------------------------- #
# User
# --------------------------------------------------------------------------- #
class UserOut(BaseModel):
    id: int
    name: str
    email: str
    role: str
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
class StatisticsOut(BaseModel):
    total_events: int = 0
    events_today: int = 0
    per_object_type: Dict[str, int] = Field(default_factory=dict)
    avg_confidence: float = 0.0


# --------------------------------------------------------------------------- #
# Generic paginated response for events
# --------------------------------------------------------------------------- #
class EventListOut(BaseModel):
    items: List[EventOut]
    total: int
    limit: int
    offset: int


class EventReviewOut(BaseModel):
    event: EventOut
    evidence: List[EvidenceOut] = Field(default_factory=list)
    job: Optional[VideoAnalysisJobOut] = None
    report: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------- #
# Video Analysis Job Schemas
# --------------------------------------------------------------------------- #
class VideoAnalysisJobBase(BaseModel):
    filename: str
    original_filename: str
    status: str = "queued"
    duration_sec: Optional[float] = None
    total_frames: Optional[int] = None
    processed_frames: int = 0
    fps: Optional[float] = None
    processing_fps: Optional[float] = None
    events_count: int = 0
    persons_detected: int = 0
    objects_detected: int = 0
    report_json: Optional[str] = None
    analyzed_video_path: Optional[str] = None
    error_message: Optional[str] = None
    # --- Persistent analysis archive fields ---
    started_at: Optional[datetime] = None
    original_video_path: Optional[str] = None
    manifest_json: Optional[str] = None
    analysis_id: Optional[int] = None
    # --- Live pipeline stage monitoring & telemetry ---
    current_stage: Optional[str] = None
    current_stage_status: Optional[str] = None
    stage_step: Optional[int] = 1
    total_stages: Optional[int] = 12
    stage_name_display: Optional[str] = None
    last_successful_stage: Optional[str] = None
    stages: Optional[Any] = None
    active_persons_count: Optional[int] = 0
    unique_persons_count: Optional[int] = 0
    total_person_track_ids: Optional[int] = 0
    candidates_count: Optional[int] = 0
    rejected_count: Optional[int] = 0
    last_processed_frame: Optional[int] = None
    last_update_time: Optional[str] = None


class VideoAnalysisJobOut(VideoAnalysisJobBase):
    id: int
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class VideoAnalysisJobListOut(BaseModel):
    items: List[VideoAnalysisJobOut]
    total: int
    limit: int
    offset: int


# --------------------------------------------------------------------------- #
# Analysis manifest (P2-13) — explicit response model for
# GET /api/analysis/jobs/{id}/manifest so the dashboard contract cannot drift
# silently. All nested containers use extra="allow": fields added later by the
# pipeline are passed through to the frontend instead of being stripped, while
# the documented top-level keys stay validated.
# --------------------------------------------------------------------------- #
class AnalysisManifestClip(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_id: Optional[Any] = None
    evidence_dir: Optional[str] = None
    snapshot: Optional[str] = None
    person: Optional[str] = None
    waste: Optional[str] = None
    carry: Optional[str] = None
    release: Optional[str] = None
    ground: Optional[str] = None
    clip: Optional[str] = None
    face: Optional[str] = None


class AnalysisManifestEvent(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_id: Optional[Any] = None
    person_track_id: Optional[Any] = None
    bag_track_id: Optional[Any] = None
    confidence: Optional[float] = None
    state: Optional[str] = None
    reason: Optional[str] = None
    frames: Optional[Dict[str, Any]] = None
    timestamps: Optional[Dict[str, Any]] = None


class AnalysisManifestMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    duration_sec: Optional[float] = None
    source_fps: Optional[float] = None
    resolution: Optional[List[Optional[int]]] = None
    processed_frames: Optional[int] = None
    persons_count: Optional[int] = None
    objects_count: Optional[int] = None
    detector_summary: Optional[Dict[str, Any]] = None
    no_candidate_reason: Optional[str] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class AnalysisManifestSizes(BaseModel):
    model_config = ConfigDict(extra="allow")
    original: Optional[int] = None
    analyzed: Optional[int] = None
    frames_jsonl: Optional[int] = None
    events_count: Optional[int] = None


class AnalysisManifestOut(BaseModel):
    model_config = ConfigDict(extra="allow")
    analysis_id: int
    job_id: int
    original_filename: str
    original_video: Optional[str] = None
    analyzed_video: Optional[str] = None
    frames_jsonl: Optional[str] = None
    event_clips: List[AnalysisManifestClip] = Field(default_factory=list)
    events: List[AnalysisManifestEvent] = Field(default_factory=list)
    # Free-form passthrough display data (timeline rows/markers are rendered
    # generically by the dashboard); pinned only at the list level.
    timeline: List[Dict[str, Any]] = Field(default_factory=list)
    markers: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Optional[AnalysisManifestMetadata] = None
    sizes_bytes: Optional[AnalysisManifestSizes] = None
    final_result: Optional[str] = None
