"""Evidence router — retrieve evidence metadata and upload snapshot+video."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend import models, schemas
from backend.database import get_db

router = APIRouter(prefix="/evidence", tags=["evidence"])

# Where uploaded evidence files are stored on disk. Default is the repository
# root ``evidence_store/`` so Docker bind/volume mounts line up. Override with
# EVIDENCE_STORE for deployments that need a different path.
EVIDENCE_STORE = Path(os.environ.get("EVIDENCE_STORE", str(Path(__file__).resolve().parents[2] / "evidence_store")))


def _event_or_404(db: Session, event_id: int) -> models.Event:
    event = db.query(models.Event).filter(models.Event.id == event_id).first()
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Event {event_id} not found",
        )
    return event


@router.get("/{event_id}", response_model=List[schemas.EvidenceOut])
def get_evidence(event_id: int, db: Session = Depends(get_db)) -> List[models.Evidence]:
    """Return all evidence records attached to an event."""
    _event_or_404(db, event_id)
    return (
        db.query(models.Evidence)
        .filter(models.Evidence.event_id == event_id)
        .order_by(models.Evidence.id.asc())
        .all()
    )


@router.post(
    "/{event_id}/upload",
    response_model=schemas.EvidenceUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_evidence(
    event_id: int,
    snapshot: UploadFile = File(..., description="Snapshot image (jpg/png)"),
    video: UploadFile = File(..., description="Short video clip (mp4)"),
    duration_sec: Optional[float] = None,
    db: Session = Depends(get_db),
) -> schemas.EvidenceUploadResponse:
    """Upload a snapshot image and a short video clip for an event.

    Files are stored under ``backend/evidence_store/{event_id}/``. A matching
    ``Evidence`` row is created with the relative file paths.
    """
    _event_or_404(db, event_id)

    target_dir = EVIDENCE_STORE / str(event_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Save snapshot.
    snapshot_filename = f"snapshot_{event_id}_{snapshot.filename or 'image.jpg'}"
    snapshot_path = target_dir / snapshot_filename
    snapshot_bytes = await snapshot.read()
    snapshot_path.write_bytes(snapshot_bytes)

    # Save video.
    video_filename = f"video_{event_id}_{video.filename or 'clip.mp4'}"
    video_path = target_dir / video_filename
    video_bytes = await video.read()
    video_path.write_bytes(video_bytes)

    # Store paths relative to the evidence_store root for portability.
    rel_snapshot = str(snapshot_path.relative_to(EVIDENCE_STORE))
    rel_video = str(video_path.relative_to(EVIDENCE_STORE))

    evidence = models.Evidence(
        event_id=event_id,
        image_path=rel_snapshot,
        video_path=rel_video,
        duration_sec=duration_sec,
    )
    db.add(evidence)
    db.commit()
    db.refresh(evidence)

    return schemas.EvidenceUploadResponse(
        evidence=schemas.EvidenceOut.model_validate(evidence),
        snapshot_filename=snapshot_filename,
        video_filename=video_filename,
    )


# ---------------------------------------------------------------------- #
# File serving — the dashboard's EvidenceViewer needs to actually display
# the snapshot image and play the video clip. The audit found this route
# was missing, so the viewer referenced a non-existent URL. These routes
# serve files from the evidence store with path-traversal protection.
# ---------------------------------------------------------------------- #
def _safe_resolve(rel_path: str) -> Path:
    """Resolve a relative evidence path, refusing anything that escapes
    the evidence store root (prevents ../../etc/passwd traversal)."""
    try:
        candidate = (EVIDENCE_STORE / rel_path).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid evidence path")
    # ensure the resolved path is inside EVIDENCE_STORE
    try:
        candidate.relative_to(EVIDENCE_STORE.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="path outside evidence store")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="evidence file not found")
    return candidate


@router.get("/file/{rel_path:path}")
def serve_evidence_file(rel_path: str):
    """Serve a single evidence file (snapshot or video) by its path
    relative to the evidence store. Used by the dashboard's
    EvidenceViewer component."""
    path = _safe_resolve(rel_path)
    # pick a media type based on extension
    ext = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".mp4": "video/mp4", ".avi": "video/x-msvideo",
        ".mov": "video/quicktime", ".webm": "video/webm",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(path=str(path), media_type=media_type)


@router.get("/event/{event_id}/snapshot")
def serve_event_snapshot(event_id: int, db: Session = Depends(get_db)):
    """Convenience route: serve the snapshot for the LATEST evidence row
    of an event (most recent upload). The dashboard uses this when it
    only needs the thumbnail."""
    ev = (
        db.query(models.Evidence)
        .filter(models.Evidence.event_id == event_id)
        .order_by(models.Evidence.id.desc())
        .first()
    )
    if ev is None or not ev.image_path:
        raise HTTPException(status_code=404, detail="no snapshot for event")
    return _serve_path(ev.image_path)


@router.get("/event/{event_id}/video")
def serve_event_video(event_id: int, db: Session = Depends(get_db)):
    """Convenience route: serve the video clip for an event."""
    ev = (
        db.query(models.Evidence)
        .filter(models.Evidence.event_id == event_id)
        .order_by(models.Evidence.id.asc())
        .first()
    )
    if ev is None or not ev.video_path:
        raise HTTPException(status_code=404, detail="no video for event")
    return _serve_path(ev.video_path)


def _serve_path(rel_path: str) -> FileResponse:
    path = _safe_resolve(rel_path)
    ext = path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".mp4": "video/mp4", ".avi": "video/x-msvideo",
        ".mov": "video/quicktime", ".webm": "video/webm",
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(path=str(path), media_type=media_type)
