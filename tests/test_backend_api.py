"""End-to-end backend API tests for the AI Littering Detection FastAPI app.

Run WITHOUT PostgreSQL: spins up a SQLite in-memory database, overrides the
``get_db`` dependency so every request uses the in-memory engine, and exercises
the real FastAPI app via ``TestClient``.

Run:
    cd /agent/workspace/ai-littering-detection
    PYTHONPATH=. python3 -m pytest tests/test_backend_api.py -v
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# --------------------------------------------------------------------------- #
# IMPORTANT: set DATABASE_URL to an in-memory SQLite database BEFORE importing
# the backend package.  backend.main registers an ``@app.on_event("startup")``
# handler that calls ``database.create_all()`` -> ``get_engine()``, which would
# otherwise try to connect to the default PostgreSQL URL and fail in CI / the
# sandbox.  Pointing it at SQLite keeps the startup handler harmless.
# --------------------------------------------------------------------------- #
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


def _cv2_available() -> bool:
    """The MJPEG stream endpoint renders frames via cv2. In headless CI/sandbox
    environments cv2 may fail to import (missing libGL.so.1). The stream tests
    should be skipped there, not fail — the route itself is correct; the failure
    is an environment limitation. On Windows (the deployment target) cv2 + its
    DLLs are present, so these tests run for real."""
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import models  # noqa: F401  (registers tables on Base.metadata)
from backend.database import Base, get_db
from backend.main import app
from backend.routers.evidence import EVIDENCE_STORE, _safe_resolve
from backend.routers.events import DEFAULT_EVENT_INGEST_TOKEN, EVENT_INGEST_HEADER


# --------------------------------------------------------------------------- #
# In-memory SQLite engine + session factory
# --------------------------------------------------------------------------- #
# CRITICAL: ``sqlite:///:memory:`` creates a SEPARATE database per connection.
# The default QueuePool hands each request session its own connection, so
# tables created via ``Base.metadata.create_all(engine)`` on one connection are
# invisible to request connections -> "no such table".
# StaticPool with a single shared connection keeps every session on the SAME
# in-memory database, so the tables created once at import time persist for the
# whole process and are visible to every request.
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    future=True,
)
TestingSessionLocal = sessionmaker(
    bind=engine, autocommit=False, autoflush=False, future=True
)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


# Create every table on the in-memory engine BEFORE the app starts handling
# requests.  All model types (Integer/String/Float/DateTime/Boolean) are
# portable to SQLite, so create_all must succeed here.
Base.metadata.create_all(bind=engine)

# Override the FastAPI dependency so routers receive sessions bound to the
# in-memory engine instead of the real (PostgreSQL) one.
app.dependency_overrides[get_db] = override_get_db

# Also override get_session_factory in database module so background tasks use in-memory engine
from backend import database
database.get_session_factory = lambda: TestingSessionLocal


@pytest.fixture(autouse=True)
def _reset_tables():
    """Clear all rows before each test so tests are isolated (e.g. the
    empty-statistics test must run against an empty database)."""
    with TestingSessionLocal() as db:
        for table in reversed(Base.metadata.sorted_tables):
            db.execute(delete(table))
        db.commit()
    yield


@pytest.fixture()
def client():
    """A TestClient that runs the app lifespan (startup/shutdown)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def camera_id(client: TestClient) -> int:
    """Create a camera and return its id, reused by event tests."""
    resp = client.post(
        "/api/cameras",
        json={"name": "Cam-A", "location": "Gate 1", "status": "active"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


@pytest.fixture()
def event_id(client: TestClient, camera_id: int) -> int:
    """Create an event referencing the camera and return its id.

    P0-3: POST /api/events is internal-only and requires the shared ingest
    token header, exactly as the live-camera pipeline sends it."""
    resp = client.post(
        "/api/events",
        json={
            "camera_id": camera_id,
            "object_type": "plastic_bottle",
            "confidence": 0.92,
            "person_track_id": "p-42",
            "object_track_id": "o-7",
            "status": "confirmed",
        },
        headers={EVENT_INGEST_HEADER: DEFAULT_EVENT_INGEST_TOKEN},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
def test_health(client: TestClient):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def test_create_and_list_camera(client: TestClient):
    payload = {"name": "Cam-List", "location": "North exit", "status": "active"}
    resp = client.post("/api/cameras", json=payload)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["name"] == "Cam-List"
    assert body["location"] == "North exit"
    assert body["status"] == "active"
    assert isinstance(body["id"], int)

    resp = client.get("/api/cameras")
    assert resp.status_code == 200
    cams = resp.json()
    assert isinstance(cams, list)
    assert any(c["id"] == body["id"] for c in cams)


def test_get_camera_by_id(client: TestClient, camera_id: int):
    resp = client.get(f"/api/cameras/{camera_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == camera_id


def test_get_camera_404(client: TestClient):
    resp = client.get("/api/cameras/999999")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #
def test_create_event(client: TestClient, camera_id: int):
    resp = client.post(
        "/api/events",
        json={
            "camera_id": camera_id,
            "object_type": "can",
            "confidence": 0.75,
        },
        headers={EVENT_INGEST_HEADER: DEFAULT_EVENT_INGEST_TOKEN},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["camera_id"] == camera_id
    assert body["object_type"] == "can"
    assert body["confidence"] == pytest.approx(0.75)
    assert body["status"] == "confirmed"
    assert body["timestamp"] is not None


def test_create_event_bad_camera(client: TestClient):
    resp = client.post(
        "/api/events",
        json={"camera_id": 777777, "object_type": "can", "confidence": 0.1},
        headers={EVENT_INGEST_HEADER: DEFAULT_EVENT_INGEST_TOKEN},
    )
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# P0-3 — provenance check on POST /api/events (MASTER_REPAIR_PLAN)
# --------------------------------------------------------------------------- #
def test_create_event_unauthenticated_rejected(client: TestClient, camera_id: int):
    """A request WITHOUT the internal ingest token must never create a
    confirmed event — 401, and no Event row may appear."""
    resp = client.post(
        "/api/events",
        json={"camera_id": camera_id, "object_type": "can", "confidence": 0.9},
    )
    assert resp.status_code == 401, resp.text
    assert resp.json()["detail"] == "Missing or invalid internal ingest token"


def test_create_event_wrong_token_rejected(client: TestClient, camera_id: int):
    """A wrong/forged token is also rejected (constant-time compare)."""
    resp = client.post(
        "/api/events",
        json={"camera_id": camera_id, "object_type": "can", "confidence": 0.9},
        headers={EVENT_INGEST_HEADER: "forged-token"},
    )
    assert resp.status_code == 401, resp.text


def test_create_event_token_env_override(monkeypatch, client: TestClient, camera_id: int):
    """Setting EVENT_INGEST_TOKEN replaces the default on the server side:
    the new token is accepted and the old default is rejected."""
    monkeypatch.setenv("EVENT_INGEST_TOKEN", "rotated-secret-123")
    ok = client.post(
        "/api/events",
        json={"camera_id": camera_id, "object_type": "can", "confidence": 0.9},
        headers={EVENT_INGEST_HEADER: "rotated-secret-123"},
    )
    assert ok.status_code == 201, ok.text
    stale = client.post(
        "/api/events",
        json={"camera_id": camera_id, "object_type": "can", "confidence": 0.9},
        headers={EVENT_INGEST_HEADER: DEFAULT_EVENT_INGEST_TOKEN},
    )
    assert stale.status_code == 401


def test_create_event_with_internal_token_persists_identity(client: TestClient, camera_id: int):
    """Authenticated internal callers still create events, and the stable
    actor/object identity fields round-trip (P0-1 + P0-3 together)."""
    resp = client.post(
        "/api/events",
        json={
            "camera_id": camera_id,
            "object_type": "trash_bag",
            "confidence": 0.9,
            "event_actor_person_track_id": 242,
            "event_actor_person_uid": 2,
            "event_object_track_id": 10244,
            "event_object_uid": 100004,
        },
        headers={EVENT_INGEST_HEADER: DEFAULT_EVENT_INGEST_TOKEN},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["event_actor_person_uid"] == 2
    assert body["event_object_uid"] == 100004
    assert body["event_actor_person_track_id"] == 242
    assert body["event_object_track_id"] == 10244


# --------------------------------------------------------------------------- #
# P1-2 — carry/release/ground sequence stills persist and are servable
# --------------------------------------------------------------------------- #
def test_sequence_image_paths_persist_and_are_served(client: TestClient, event_id: int):
    """An Evidence row carrying carry/release/ground paths must expose them via
    GET /api/evidence/{event_id} and the files must be servable through the
    existing evidence file route (P1-2: they used to be orphaned artifacts)."""
    from backend.routers.evidence import EVIDENCE_STORE
    from backend import models as m

    # write three real tiny JPEGs into the evidence store
    paths = {}
    import numpy as np

    import cv2

    for key in ("carry", "release", "ground"):
        target = EVIDENCE_STORE / str(event_id)
        target.mkdir(parents=True, exist_ok=True)
        f = target / f"{key}.jpg"
        cv2.imwrite(str(f), np.full((40, 60, 3), 128, dtype=np.uint8))
        paths[f"{key}_image_path"] = f"{event_id}/{key}.jpg"

    db = TestingSessionLocal()
    try:
        db.add(m.Evidence(event_id=event_id, image_path=f"{event_id}/snapshot.jpg", **paths))
        db.commit()
    finally:
        db.close()

    listing = client.get(f"/api/evidence/{event_id}")
    assert listing.status_code == 200, listing.text
    rows = listing.json()
    assert rows, "no evidence rows returned"
    body = rows[0]
    for field in ("carry_image_path", "release_image_path", "ground_image_path"):
        assert body[field] == paths[field], f"{field} not exposed by the API"
    for rel in paths.values():
        served = client.get(f"/api/evidence/file/{rel}")
        assert served.status_code == 200, f"{rel} not servable"
        assert len(served.content) > 0


def test_list_events_paginated(client: TestClient, event_id: int):
    resp = client.get("/api/events")
    assert resp.status_code == 200
    body = resp.json()
    # EventListOut = {items, total, limit, offset}
    assert set(body.keys()) == {"items", "total", "limit", "offset"}
    assert body["total"] >= 1
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert any(e["id"] == event_id for e in body["items"])


def test_list_events_pagination_params(client: TestClient, event_id: int):
    resp = client.get("/api/events", params={"limit": 1, "offset": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["limit"] == 1
    assert len(body["items"]) <= 1


def test_get_event_by_id(client: TestClient, event_id: int):
    resp = client.get(f"/api/events/{event_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == event_id


def test_get_event_404(client: TestClient):
    resp = client.get("/api/events/999999")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def test_statistics(client: TestClient, event_id: int):
    resp = client.get("/api/statistics")
    assert resp.status_code == 200
    body = resp.json()
    # StatisticsOut fields
    assert "total_events" in body
    assert "events_today" in body
    assert "per_object_type" in body
    assert "avg_confidence" in body
    assert body["total_events"] >= 1
    assert body["events_today"] >= 1  # event created moments ago
    assert isinstance(body["per_object_type"], dict)
    assert body["per_object_type"].get("plastic_bottle", 0) >= 1
    assert isinstance(body["avg_confidence"], (int, float))
    assert 0.0 <= body["avg_confidence"] <= 1.0


def test_statistics_empty(client: TestClient):
    """With no events the aggregate should be zeroed out, not error."""
    resp = client.get("/api/statistics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_events"] == 0
    assert body["events_today"] == 0
    assert body["per_object_type"] == {}
    assert body["avg_confidence"] == 0.0


# --------------------------------------------------------------------------- #
# Evidence metadata
# --------------------------------------------------------------------------- #
def test_get_evidence_empty_list(client: TestClient, event_id: int):
    resp = client.get(f"/api/evidence/{event_id}")
    assert resp.status_code == 200
    assert resp.json() == []


def test_get_evidence_event_404(client: TestClient):
    resp = client.get("/api/evidence/999999")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# File-serving route + path-traversal protection
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_evidence_file(event_id: int):
    """Create a fake snapshot on disk under evidence_store/{event_id}/."""
    target_dir = EVIDENCE_STORE / str(event_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    fake_file = target_dir / "snapshot.jpg"
    fake_file.write_bytes(b"\xff\xd8\xff\xe0FAKE-JPEG-DATA")
    rel = f"{event_id}/snapshot.jpg"
    yield rel
    # cleanup
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)


def test_serve_evidence_file(client: TestClient, fake_evidence_file: str):
    resp = client.get(f"/api/evidence/file/{fake_evidence_file}")
    assert resp.status_code == 200, resp.text
    assert resp.content.startswith(b"\xff\xd8\xff\xe0")
    assert "image/jpeg" in resp.headers.get("content-type", "")


def test_serve_evidence_file_404(client: TestClient):
    resp = client.get("/api/evidence/file/does/not/exist.jpg")
    assert resp.status_code == 404


def test_path_traversal_direct_unit():
    """Directly exercise the _safe_resolve guard with an escaping path."""
    with pytest.raises(Exception) as exc:
        _safe_resolve("../../etc/passwd")
    # FastAPI HTTPException carries a status_code attribute.
    status = getattr(exc.value, "status_code", None)
    assert status in (400, 403), f"expected 400/403, got status={status!r}"


def test_path_traversal_http_encoded(client: TestClient):
    """HTTP-level traversal test.

    httpx normalises literal ``../`` dot-segments out of the URL before
    sending, so a request like ``/api/evidence/file/../../etc/passwd`` would
    be rewritten to ``/api/etc/passwd`` and never reach the route (-> 404),
    which would NOT exercise the guard.  We therefore encode the slashes as
    ``%2f`` so the escaping path reaches ``serve_evidence_file`` intact and
    ``_safe_resolve`` rejects it.
    """
    resp = client.get("/api/evidence/file/..%2f..%2fetc%2fpasswd")
    assert resp.status_code in (400, 403), (
        f"expected 400/403 for traversal, got {resp.status_code}: {resp.text!r}"
    )
    # And it must not have leaked /etc/passwd contents.
    assert b"root:" not in resp.content


def test_path_traversal_http_literal(client: TestClient):
    """Document the literal-``..`` behaviour (httpx normalises it -> 404).

    This is intentionally NOT a 403 because the request never reaches the
    route.  We assert only that it does not return 200 and does not leak
    /etc/passwd, and record the actual status for honesty.
    """
    resp = client.get("/api/evidence/file/../../etc/passwd")
    assert resp.status_code != 200
    assert b"root:" not in resp.content
    # Expected: httpx collapses the dot-segments -> /api/etc/passwd -> 404.
    assert resp.status_code == 404, (
        f"expected 404 (httpx normalisation), got {resp.status_code}"
    )


# =========================================================================== #
# Status endpoint (GET /api/status)
# =========================================================================== #
def test_status_endpoint(client: TestClient):
    """GET /api/status returns 200 with all required status-bar fields."""
    resp = client.get("/api/status")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Top-level fields required by the dashboard status bar.
    assert body["system_online"] is True
    assert isinstance(body["ai_engine"], dict)
    assert isinstance(body["camera"], dict)
    assert isinstance(body["processing"], dict)
    assert isinstance(body["buffer"], dict)
    assert isinstance(body["events_today"], int)
    assert isinstance(body["active_cameras"], int)
    assert isinstance(body["updated_at"], str)

    # ai_engine sub-fields
    ai = body["ai_engine"]
    assert ai["status"] in ("online", "offline", "degraded")
    assert isinstance(ai["status"], str)
    assert isinstance(ai["model_loaded"], bool)
    assert isinstance(ai["classes"], list)
    # HONEST default: no pipeline has pushed status yet, so engine is offline.
    assert ai["status"] == "offline"
    assert ai["model_loaded"] is False

    # camera sub-fields
    cam = body["camera"]
    assert cam["status"] in ("online", "offline", "waiting")
    # No real camera connected in tests -> honest "offline".
    assert cam["status"] == "offline"
    assert "fps" in cam
    assert "resolution" in cam
    assert "source" in cam

    # processing sub-fields
    proc = body["processing"]
    assert "fps" in proc
    assert "latency_ms" in proc
    assert "analysis_fps" in proc

    # buffer sub-fields
    buf = body["buffer"]
    assert isinstance(buf["window_seconds"], (int, float))
    assert isinstance(buf["frames_buffered"], int)
    assert isinstance(buf["buffer_duration"], (int, float))


def test_status_reflects_pipeline_push(client: TestClient):
    """When the pipeline pushes metrics via set_status, the endpoint reflects them."""
    from backend.routers import status as status_router

    status_router.set_status(
        ai_engine={"status": "online", "model_loaded": True,
                   "classes": ["person", "bottle", "can"]},
        camera={"status": "online", "fps": 30.0, "resolution": "1920x1080",
                "source": "iPhone"},
        processing={"fps": 28.5, "latency_ms": 35.0, "analysis_fps": 30.0},
        buffer={"window_seconds": 20.0, "frames_buffered": 600,
                "buffer_duration": 20.0},
    )
    try:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ai_engine"]["status"] == "online"
        assert body["ai_engine"]["model_loaded"] is True
        assert body["ai_engine"]["classes"] == ["person", "bottle", "can"]
        assert body["camera"]["status"] == "online"
        assert body["camera"]["fps"] == pytest.approx(30.0)
        assert body["processing"]["latency_ms"] == pytest.approx(35.0)
        assert body["buffer"]["frames_buffered"] == 600
    finally:
        status_router.reset_status()


def test_status_events_today_and_active_cameras(client: TestClient, event_id: int):
    """events_today and active_cameras are derived from the DB and stay live."""
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["events_today"] >= 1  # the event_id fixture created one today
    assert body["active_cameras"] >= 1  # the camera_id fixture created an active cam


# =========================================================================== #
# MJPEG stream endpoint (GET /api/cameras/{id}/stream)
# =========================================================================== #
@pytest.mark.skipif(not _cv2_available(),
                    reason="cv2 not importable in this headless environment (libGL.so.1 missing) — stream route renders frames via cv2")
def test_stream_placeholder_frame(client: TestClient, camera_id: int):
    """GET /api/cameras/{id}/stream returns a multipart/x-mixed-replace stream.

    NOTE on streaming + TestClient: Starlette's ``TestClient`` drives the ASGI
    app to completion and buffers the response. A real MJPEG stream never
    ends, so an infinite generator would hang TestClient forever. We use the
    endpoint's ``max_frames`` query param (a legitimate feature — emit a
    bounded number of frames then stop) to make the response finite. We
    assert:

      * the content-type starts with ``multipart/x-mixed-replace``
      * the body begins with the ``--frame`` boundary
      * the body is non-empty and contains JPEG frame data (SOI marker)

    Full streaming verification (true indefinite chunked delivery) requires a
    real async HTTP client such as ``httpx.AsyncClient`` or a browser;
    TestClient cannot prove that the stream stays open indefinitely.
    """
    resp = client.get(f"/api/cameras/{camera_id}/stream", params={"max_frames": 1})
    assert resp.status_code == 200, resp.text
    ct = resp.headers.get("content-type", "")
    assert ct.startswith("multipart/x-mixed-replace"), (
        f"expected multipart/x-mixed-replace, got {ct!r}"
    )
    assert "boundary=frame" in ct, f"expected boundary=frame in {ct!r}"

    body = resp.content
    assert len(body) > 0, "stream produced no bytes"
    # The multipart body starts with the boundary.
    assert body.startswith(b"--frame"), (
        f"expected body to start with --frame, got {body[:32]!r}"
    )
    # Contains a JPEG SOI marker (frame payload).
    assert b"\xff\xd8\xff" in body, "stream body did not contain JPEG frame data"


def test_stream_camera_404(client: TestClient):
    """Streaming a non-existent camera returns 404."""
    resp = client.get("/api/cameras/999999/stream")
    assert resp.status_code == 404



# =========================================================================== #
# Evidence file round-trip: AI creates file → DB stores path → API serves file
# (the audit's section 9 — prove the path matches the filesystem, not just strings)
# =========================================================================== #
def test_evidence_full_roundtrip(client: TestClient, camera_id: int, event_id: int):
    """Create a real evidence file on disk, store its path in the DB via the
    upload endpoint, then retrieve it through the file-serving route and verify
    the bytes match. This is the end-to-end evidence path the dashboard's
    EvidenceViewer depends on."""
    from backend.routers.evidence import EVIDENCE_STORE
    import shutil

    target_dir = EVIDENCE_STORE / str(event_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    real_jpeg = b"\xff\xd8\xff\xe0REAL-EVIDENCE-SNAPSHOT-DATA\xff\xd9"
    real_mp4 = b"\x00\x00\x00\x1cftypREAL-EVIDENCE-VIDEO"
    snap = target_dir / "snapshot.jpg"
    vid = target_dir / "evidence.mp4"
    snap.write_bytes(real_jpeg)
    vid.write_bytes(real_mp4)

    try:
        # upload via the API (multipart) so a DB Evidence row is created with
        # the relative paths
        with open(snap, "rb") as sf, open(vid, "rb") as vf:
            resp = client.post(
                f"/api/evidence/{event_id}/upload",
                files={
                    "snapshot": ("snapshot.jpg", sf, "image/jpeg"),
                    "video": ("evidence.mp4", vf, "video/mp4"),
                },
                data={"duration_sec": "6.0"},
            )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        rel_snap = body["evidence"]["image_path"]
        rel_vid = body["evidence"]["video_path"]

        # retrieve the snapshot through the file-serving route
        r1 = client.get(f"/api/evidence/file/{rel_snap}")
        assert r1.status_code == 200
        assert r1.content == real_jpeg, "served snapshot bytes != written bytes"
        assert "image/jpeg" in r1.headers["content-type"]

        # retrieve the video through the file-serving route
        r2 = client.get(f"/api/evidence/file/{rel_vid}")
        assert r2.status_code == 200
        assert r2.content == real_mp4, "served video bytes != written bytes"
        assert "video/mp4" in r2.headers["content-type"]

        # the metadata endpoint returns the same paths the frontend uses
        r3 = client.get(f"/api/evidence/{event_id}")
        assert r3.status_code == 200
        ev_rows = r3.json()
        assert len(ev_rows) == 1
        assert ev_rows[0]["image_path"] == rel_snap
        assert ev_rows[0]["video_path"] == rel_vid
    finally:
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)


# =========================================================================== #
# Pipeline → backend status wiring (audit section: pipeline never called set_status)
# =========================================================================== #
def test_pipeline_push_status_reflects_in_endpoint(client: TestClient):
    """When the inference pipeline calls push_status(), the /api/status
    endpoint must reflect the real AI engine + camera state — not stay
    stuck on 'offline'."""
    from backend.routers import status as status_router

    # simulate what run_pipeline.py does each stats tick
    status_router.set_status(
        ai_engine={"status": "online", "model_loaded": True, "classes": ["person", "bottle"]},
        camera={"status": "online", "fps": 28.0, "resolution": "1920x1080", "source": "iPhone"},
        processing={"fps": 28.0, "latency_ms": 35.0, "analysis_fps": 10.0},
        buffer={"window_seconds": 6.0, "frames_buffered": 180, "buffer_duration": 6.0},
    )
    try:
        resp = client.get("/api/status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ai_engine"]["status"] == "online"
        assert body["ai_engine"]["model_loaded"] is True
        assert body["camera"]["status"] == "online"
        assert body["camera"]["fps"] == 28.0
        assert body["processing"]["latency_ms"] == 35.0
    finally:
        status_router.reset_status()


# =========================================================================== #
# Video Analysis Endpoints
# =========================================================================== #
def test_video_analysis_endpoints(client: TestClient):
    """Test the /api/analysis/jobs listing, upload rejection of invalid files, and single job query."""
    import io
    # 1. list jobs
    r_list = client.get("/api/analysis/jobs")
    assert r_list.status_code == 200
    assert "items" in r_list.json()
    assert "total" in r_list.json()

    # 2. upload invalid format -> 400
    bad_file = io.BytesIO(b"not a video")
    r_bad = client.post(
        "/api/analysis/upload",
        files={"file": ("test.txt", bad_file, "text/plain")}
    )
    assert r_bad.status_code == 400
    assert "Unsupported video format" in r_bad.json()["detail"]

    # 3. upload empty video -> 400
    empty_file = io.BytesIO(b"")
    r_empty = client.post(
        "/api/analysis/upload",
        files={"file": ("empty.mp4", empty_file, "video/mp4")}
    )
    assert r_empty.status_code == 400
    assert "empty" in r_empty.json()["detail"].lower()

    # 4. upload valid dummy MP4 -> 201 queued job
    dummy_mp4 = io.BytesIO(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41")
    r_upload = client.post(
        "/api/analysis/upload",
        files={"file": ("test_run.mp4", dummy_mp4, "video/mp4")}
    )
    assert r_upload.status_code == 201
    job_id = r_upload.json()["id"]
    assert r_upload.json()["original_filename"] == "test_run.mp4"
    assert r_upload.json()["status"] in ["queued", "processing", "completed", "failed"]

    # 5. get job details
    r_job = client.get(f"/api/analysis/jobs/{job_id}")
    assert r_job.status_code == 200
    assert r_job.json()["id"] == job_id
