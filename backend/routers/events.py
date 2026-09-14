"""Event router — list (paginated), get one, create (inference reports)."""

from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend import models, schemas
from backend.database import get_db

router = APIRouter(prefix="/events", tags=["events"])

# --------------------------------------------------------------------------- #
# P0-3 (MASTER_REPAIR_PLAN): POST /api/events is an INTERNAL ingestion
# endpoint for the trusted inference pipeline only. Without a provenance
# check any HTTP client could create a "confirmed" violation indistinguishable
# from a real AI detection (exploited in production as event row 29).
#
# The live-camera pipeline (inference/pipeline.py::_maybe_post_backend) sends
# the shared token in the X-Internal-Token header. Keep DEFAULT_EVENT_INGEST_TOKEN
# in sync with inference/pipeline.py; set EVENT_INGEST_TOKEN in the environment
# to override it on both sides.
# --------------------------------------------------------------------------- #
EVENT_INGEST_HEADER = "X-Internal-Token"
DEFAULT_EVENT_INGEST_TOKEN = "littering-internal-ingest-v1"


def expected_ingest_token() -> str:
    return os.environ.get("EVENT_INGEST_TOKEN", DEFAULT_EVENT_INGEST_TOKEN)


@router.get("", response_model=schemas.EventListOut)
def list_events(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> schemas.EventListOut:
    """Return a paginated list of littering events (newest first)."""
    query = db.query(models.Event).order_by(models.Event.id.desc())
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    return schemas.EventListOut(
        items=[schemas.EventOut.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{event_id}", response_model=schemas.EventOut)
def get_event(event_id: int, db: Session = Depends(get_db)) -> models.Event:
    """Return a single event by id (including related evidence via schema)."""
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    return event


@router.get("/{event_id}/evidence", response_model=List[schemas.EvidenceOut])
def get_event_evidence(event_id: int, db: Session = Depends(get_db)) -> List[models.Evidence]:
    """Return all evidence records attached to a specific event.

    Historical events stay retrievable: each event references its analysis_job_id
    and its evidence rows are keyed to the event, so opening an old event never
    shows another analysis's evidence.
    """
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    return (
        db.query(models.Evidence)
        .filter(models.Evidence.event_id == event_id)
        .order_by(models.Evidence.id.asc())
        .all()
    )


@router.get("/{event_id}/review", response_model=schemas.EventReviewOut)
def get_event_review(event_id: int, db: Session = Depends(get_db)) -> schemas.EventReviewOut:
    """Aggregate event + evidence + analysis job/report for the review UI."""
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    evidence = (
        db.query(models.Evidence)
        .filter(models.Evidence.event_id == event_id)
        .order_by(models.Evidence.id.asc())
        .all()
    )
    job = None
    report = None
    if event.analysis_job_id:
        job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == event.analysis_job_id).first()
        if job and job.report_json:
            try:
                import json

                report = json.loads(job.report_json)
            except Exception:
                report = None
    return schemas.EventReviewOut(
        event=schemas.EventOut.model_validate(event),
        evidence=[schemas.EvidenceOut.model_validate(e) for e in evidence],
        job=schemas.VideoAnalysisJobOut.model_validate(job) if job else None,
        report=report,
    )


@router.post("/{event_id}/verdict", response_model=schemas.EventVerdictOut)
def submit_event_verdict(
    event_id: int,
    body: schemas.EventVerdictIn,
    db: Session = Depends(get_db),
) -> schemas.EventVerdictOut:
    """Record operator confirm/reject → ``data/incoming/.../verdict.json`` (Section 7).

    ``reject`` is treated as a hard-negative for offline YOLO fine-tuning.
    Does NOT mutate model weights or FSM thresholds mid-session.
    """
    from pathlib import Path

    from learning_offline.ingest import write_verdict

    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    verdict = str(body.verdict).strip().lower()
    if verdict not in ("confirm", "reject"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="verdict must be 'confirm' or 'reject'",
        )

    evidence = (
        db.query(models.Evidence)
        .filter(models.Evidence.event_id == event_id)
        .order_by(models.Evidence.id.asc())
        .all()
    )
    crop_paths = {}
    for ev in evidence:
        if ev.waste_image_path:
            crop_paths["crop_waste.jpg"] = Path(ev.waste_image_path)
        if ev.person_image_path:
            crop_paths["crop_person.jpg"] = Path(ev.person_image_path)
        if crop_paths:
            break

    try:
        out_dir = write_verdict(
            event_id,
            verdict,
            notes=body.notes or "",
            crop_paths=crop_paths,
            metadata={
                "camera_id": event.camera_id,
                "object_type": event.object_type,
                "confidence": event.confidence,
                "analysis_job_id": event.analysis_job_id,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # Mirror status for dashboard filters (confirmed / rejected).
    event.status = "confirmed" if verdict == "confirm" else "rejected"
    db.add(event)
    db.commit()

    return schemas.EventVerdictOut(
        event_id=event_id,
        verdict=verdict,
        status=event.status,
        incoming_dir=str(out_dir),
        verdict_path=str(out_dir / "verdict.json"),
    )


@router.post(
    "",
    response_model=schemas.EventOut,
    status_code=status.HTTP_201_CREATED,
)
def create_event(
    event: schemas.EventCreate,
    db: Session = Depends(get_db),
    x_internal_token: Optional[str] = Header(
        default=None, alias=EVENT_INGEST_HEADER
    ),
) -> models.Event:
    """Create a confirmed littering event.

    INTERNAL endpoint: only the trusted inference pipeline may call it. The
    caller must present the shared ingest token in the ``X-Internal-Token``
    header (see P0-3). The referenced ``camera_id`` must exist.
    """
    provided = x_internal_token or ""
    if not secrets.compare_digest(provided, expected_ingest_token()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid internal ingest token",
        )

    camera = (
        db.query(models.Camera).filter(models.Camera.id == event.camera_id).first()
    )
    if camera is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Camera {event.camera_id} does not exist",
        )

    data = event.model_dump(exclude_unset=True)
    if data.get("timestamp") is None:
        data["timestamp"] = datetime.now(timezone.utc)

    db_event = models.Event(**data)
    db.add(db_event)
    db.commit()
    db.refresh(db_event)
    return db_event
