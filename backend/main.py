"""FastAPI application entrypoint for the AI Littering Detection backend."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import create_all
from backend.routers import analysis, cameras, events, evidence, statistics, status, stream

app = FastAPI(
    title="AI Littering Detection API",
    description=(
        "Backend API for the AI-Based CCTV Littering Detection and Evidence "
        "System. Exposes cameras, littering events, evidence uploads, "
        "video file analysis, and aggregate statistics."
    ),
    version="1.0.0",
)

# CORS — allow all origins for local development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Routers (mounted under /api)
# --------------------------------------------------------------------------- #
app.include_router(analysis.router, prefix="/api")
app.include_router(cameras.router, prefix="/api")
app.include_router(events.router, prefix="/api")
app.include_router(evidence.router, prefix="/api")
app.include_router(statistics.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(stream.router, prefix="/api")


# --------------------------------------------------------------------------- #
# Lifespan / startup
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def on_startup() -> None:
    """Create database tables and recover any analysis jobs orphaned by a
    previous process crash (e.g. a container restart with no restart policy,
    or a killed worker thread). Such jobs are left stuck in `processing` /
    `queued` with no worker thread, so we transparently re-run them. Because
    the job only commits its events AFTER the frame loop finishes, re-running
    an orphan can never duplicate persisted events.
    """
    create_all()
    try:
        import logging
        import threading

        log = logging.getLogger("ai_littering.main")
        from backend.database import SessionLocal
        from backend import models
        from backend.routers.analysis import _run_video_analysis_job

        db = SessionLocal()
        try:
            orphans = (
                db.query(models.VideoAnalysisJob)
                .filter(models.VideoAnalysisJob.status.in_(["processing", "queued"]))
                .all()
            )
        finally:
            db.close()

        for job in orphans:
            log.warning(
                "Startup recovery: re-launching orphaned analysis job %s (status=%s)",
                job.id, job.status,
            )
            t = threading.Thread(
                target=_run_video_analysis_job, args=(job.id,), daemon=True
            )
            t.start()
    except Exception:
        # Never let recovery break app startup.
        import logging
        logging.getLogger("ai_littering.main").exception(
            "Startup analysis-job recovery failed"
        )


# --------------------------------------------------------------------------- #
# Health / readiness (REPAIR-P0-06)
# --------------------------------------------------------------------------- #
_READINESS_CACHE: dict | None = None


def _compute_readiness() -> dict:
    """Fail readiness until tracker/detector imports and weight files exist."""
    checks: dict = {}
    ok = True
    try:
        import lap  # noqa: F401

        checks["lap"] = {"ok": True, "detail": "importable"}
    except Exception as exc:  # pragma: no cover - env dependent
        ok = False
        checks["lap"] = {"ok": False, "detail": str(exc)}

    try:
        import ultralytics  # noqa: F401

        checks["ultralytics"] = {"ok": True, "detail": "importable"}
    except Exception as exc:  # pragma: no cover
        ok = False
        checks["ultralytics"] = {"ok": False, "detail": str(exc)}

    from pathlib import Path

    weights_dir = Path(__file__).resolve().parents[1] / "inference" / "detection" / "weights"
    for name in ("best.pt", "garbage_bag_v2.pt", "yolov8n.pt"):
        path = weights_dir / name
        present = path.is_file() and path.stat().st_size > 1000
        checks[f"weights:{name}"] = {
            "ok": present,
            "detail": str(path) if present else f"missing:{path}",
        }
        if not present:
            ok = False

    return {"ready": ok, "checks": checks}


@app.get("/health", tags=["health"])
def health() -> dict:
    """Liveness probe (process is up)."""
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
def ready() -> dict:
    """Readiness probe — do not accept analysis jobs until dependencies load."""
    global _READINESS_CACHE
    if _READINESS_CACHE is None:
        _READINESS_CACHE = _compute_readiness()
    body = {"status": "ready" if _READINESS_CACHE["ready"] else "not_ready", **_READINESS_CACHE}
    if not _READINESS_CACHE["ready"]:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=503, content=body)
    return body
