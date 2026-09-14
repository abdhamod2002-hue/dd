"""Create a DB analysis job for an existing video and run the production worker."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault("TFHUB_CACHE_DIR", str(REPO / "models" / "movenet"))
os.environ.setdefault(
    "DATABASE_URL",
    os.environ.get("DATABASE_URL", "postgresql://litter:litter@postgres:5432/littering"),
)

from backend.database import SessionLocal, create_all  # noqa: E402
from backend import models  # noqa: E402
from backend.routers.analysis import _run_video_analysis_job  # noqa: E402


def main(video: Path) -> int:
    create_all()
    video = video.resolve()
    if not video.is_file():
        raise SystemExit(f"missing video: {video}")
    # Store path relative to repo when under /app
    try:
        rel = str(video.relative_to(REPO)).replace("\\", "/")
    except ValueError:
        rel = str(video)

    db = SessionLocal()
    try:
        job = models.VideoAnalysisJob(
            filename=video.name,
            original_filename=video.name,
            status="queued",
            original_video_path=rel,
            file_path=rel,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = int(job.id)
        print(f"CREATED_JOB {job_id} path={rel}", flush=True)
    finally:
        db.close()

    _run_video_analysis_job(job_id)

    db = SessionLocal()
    try:
        job = db.get(models.VideoAnalysisJob, job_id)
        print(
            f"JOB_DONE id={job_id} status={job.status} events={job.events_count} "
            f"err={job.error_message}",
            flush=True,
        )
        events = (
            db.query(models.Event)
            .filter(models.Event.analysis_job_id == job_id)
            .all()
        )
        for ev in events:
            print(
                f"EVENT id={ev.id} conf={ev.confidence} "
                f"actor_uid={ev.event_actor_person_uid} obj_uid={ev.event_object_uid} "
                f"person_track={ev.person_track_id} object_track={ev.object_track_id}",
                flush=True,
            )
            evs = db.query(models.Evidence).filter(models.Evidence.event_id == ev.id).all()
            for e in evs:
                print(
                    f"  EVIDENCE clip={e.clip_path} person={e.person_image_path} "
                    f"waste={e.waste_image_path} face={e.face_image_path} "
                    f"ground={e.ground_image_path}",
                    flush=True,
                )
        return 0 if job.status == "completed" and job.events_count > 0 else 1
    finally:
        db.close()


if __name__ == "__main__":
    vid = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/app/backend/uploaded_videos/20260831_104342_IMG_5117.MOV"
    )
    raise SystemExit(main(vid))
