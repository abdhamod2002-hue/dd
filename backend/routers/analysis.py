"""Video Analysis Router — upload video files and run the real production AI pipeline."""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend import models, schemas
from backend.database import SessionLocal, get_db

router = APIRouter(prefix="/analysis", tags=["analysis"])
log = logging.getLogger("ai_littering.analysis")

# Repo root (host) and its container-equivalent (/app) are both searched by
# ``_resolve_archive_path`` so original/analyzed/evidence files resolve whether
# the backend runs on the host (Windows) or inside a Docker container after a
# restart. The absolute host path is NOT stored; only the port:
# uploads are under ``backend/uploaded_videos/...`` relative to the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINER_ROOT = Path("/app")

UPLOAD_DIR = REPO_ROOT / "backend" / "uploaded_videos"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_archive_path(rel_path: Optional[str]) -> Optional[Path]:
    """Resolve a repo-root-relative archive path on host or inside the container.

    Portability matters for the Docker backend: after ``docker compose down`` +
    ``up`` the original/analyzed/evidence files must still open. We check the
    container path first (when running inside a container), then the host repo
    root. ``rel_path`` is always stored relative to the repo root, never as an
    absolute host path.
    """
    rel_path = (rel_path or "").strip()
    if not rel_path or rel_path in (".", ".."):
        return None
    clean = rel_path.lstrip("/\\")
    for base in (CONTAINER_ROOT, REPO_ROOT):
        candidate = base / clean
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _compact_detector_event(event) -> dict:
    data = event.to_dict()
    return {
        "event_id": data.get("event_id"),
        "person_track_id": data.get("person_track_id"),
        "bag_track_id": data.get("bag_track_id"),
        "bag_uid": data.get("bag_uid"),
        # Phase C — authoritative event-actor record, frozen at carry time. Every
        # evidence artifact downstream must reference these exact IDs.
        "event_actor_person_track_id": data.get("event_actor_person_track_id"),
        "event_actor_person_uid": data.get("event_actor_person_uid"),
        "event_object_track_id": data.get("event_object_track_id"),
        "event_object_uid": data.get("event_object_uid"),
        "state": data.get("state"),
        "confirmed": data.get("confirmed"),
        "reason": data.get("reason"),
        "confidence": data.get("confidence"),
        "evidence": data.get("evidence"),
        "frames": data.get("frames"),
        "timestamps": data.get("timestamps"),
        "bag_class": data.get("bag_class"),
        "fallback_used": data.get("fallback_used"),
        "yolo_reconfirmed": data.get("yolo_reconfirmed"),
        "detector_source": data.get("detector_source"),
        "other_person_closer": data.get("other_person_closer"),
    }


def _fit_report_json(report: dict, max_chars: int = 3900) -> str:
    """Serialize report while keeping the legacy varchar(4096) column safe."""
    def _dump() -> str:
        return json.dumps(report, ensure_ascii=False)

    text = _dump()
    if len(text) <= max_chars:
        return text

    detector = report.get("event_detector")
    if isinstance(detector, dict):
        detector["confirmed_violations"] = detector.get("confirmed_violations", [])[:2]
        detector["rejected_candidates"] = detector.get("rejected_candidates", [])[:2]
        text = _dump()
        if len(text) <= max_chars:
            return text

    if isinstance(report.get("timeline"), list):
        report["timeline"] = report["timeline"][-12:]
        text = _dump()
        if len(text) <= max_chars:
            return text

    if isinstance(detector, dict):
        detector["confirmed_violations"] = []
        detector["rejected_candidates"] = []
        detector["config"] = {}
        text = _dump()
        if len(text) <= max_chars:
            return text

    minimal = {
        "truncated": True,
        "source": report.get("source"),
        "confirmed_events": report.get("confirmed_events", 0),
        "no_candidate_reason": report.get("no_candidate_reason"),
        "event_detector": {
            "summary": report.get("event_detector", {}).get("summary", {}) if isinstance(report.get("event_detector"), dict) else {},
            "truncated": True,
        },
    }
    text = json.dumps(minimal, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return json.dumps({"truncated": True, "event_detector": {"summary": {}}}, ensure_ascii=False)


def _current_detector_state(detector) -> str:
    """Return the most advanced active detector state for timeline reporting."""
    priority = {
        "NO_BAG": 0,
        "OBJECT_NEAR_PERSON": 1,
        "BAG_CARRIED": 2,
        "BAG_RELEASED": 3,
        "BAG_ON_GROUND": 4,
        "PERSON_DEPARTED": 5,
        "VIOLATION_CONFIRMED": 6,
    }
    best = "NO_BAG"
    best_score = -1
    for mem in detector._pairs.values():
        score = priority.get(mem.state.value, 0)
        if score > best_score:
            best = mem.state.value
            best_score = score
    return best


def _no_candidate_reason(summary: dict, persons_count: int, objects_count: int) -> str:
    """Explain why no littering candidate was confirmed using measured signals."""
    if persons_count == 0:
        return "PERSON_NOT_DETECTED"
    if objects_count == 0:
        return "OBJECT_NOT_DETECTED"
    reasons = summary.get("rejection_reason_counts") or {}
    if reasons:
        return max(reasons.items(), key=lambda kv: kv[1])[0]
    if summary.get("total_candidates", 0) == 0:
        return "ASSOCIATION_FAILED"
    return "NO_LITTERING_EVENT_CANDIDATE"


def _dominant_detector_source(object_sources) -> str:
    """Honestly label which detector produced the waste-object candidates.

    best.pt contains NO bag class, so any waste-bag detection is produced by the
    HSV color fallback. This is surfaced explicitly so the reviewer knows the
    event source rather than assuming a YOLO object model fired.
    """
    has_yolo = "yolo" in object_sources
    has_color = "color" in object_sources
    if has_yolo and has_color:
        return "yolo_and_color"
    if has_color:
        return "color_fallback_only"
    if has_yolo:
        return "yolo_only"
    return "none"


def _build_stages(
    *,
    processed_frames: int,
    persons_count: int,
    objects_count: int,
    object_sources,
    tracking_ok: bool,
    detector_summary: dict,
    confirmed_events: int,
    created_event_ids,
    evidence_packages,
    current_state: str,
) -> List[dict]:
    """Build a structured pipeline-progress stepper from REAL measured signals.

    Every stage status is derived from an actual signal produced by the run;
    nothing here is hardcoded or video-specific. The dashboard renders this as
    a step-by-step view of what the AI actually did.
    """
    candidates = detector_summary.get("total_candidates", 0)

    def stage(name: str, status: str, detail: str, order: int) -> dict:
        return {"name": name, "status": status, "detail": detail, "order": order}

    stages = [
        stage(
            "video_input",
            "PASS" if processed_frames > 0 else "FAIL",
            f"{processed_frames} frames decoded",
            1,
        ),
        stage(
            "person_detection",
            "PASS" if persons_count > 0 else "FAIL",
            f"{persons_count} person track(s) detected",
            2,
        ),
        stage(
            "waste_detection",
            "PASS" if objects_count > 0 else "FAIL",
            f"{objects_count} object track(s); source={_dominant_detector_source(object_sources)}",
            3,
        ),
        stage(
            "tracking",
            "PASS" if tracking_ok else "FAIL",
            "track identities maintained across frames",
            4,
        ),
        stage(
            "pose_estimation",
            "PASS" if persons_count > 0 else "FAIL",
            "MoveNet keypoints (wrists) computed for detected persons",
            5,
        ),
        stage(
            "association",
            "PASS" if candidates > 0 else "FAIL",
            f"{candidates} person-object candidate pair(s)",
            6,
        ),
        stage(
            "behavior_state_machine",
            "PASS" if candidates > 0 else "PENDING",
            f"most advanced state reached: {current_state}",
            7,
        ),
        stage(
            "event",
            "CONFIRMED" if confirmed_events > 0 else ("CANDIDATE" if candidates > 0 else "NONE"),
            (
                f"{confirmed_events} confirmed violation(s)"
                if confirmed_events > 0
                else ("candidate(s) evaluated and rejected — see rejection reasons" if candidates > 0 else "no littering candidate produced")
            ),
            8,
        ),
        stage(
            "evidence",
            "PASS" if (confirmed_events == 0 or evidence_packages) else "PENDING",
            f"{len(evidence_packages)} evidence package(s) assembled",
            9,
        ),
        stage(
            "database",
            "PASS" if (confirmed_events == 0 or created_event_ids) else "WARN",
            "event records persisted to PostgreSQL" if created_event_ids else "no confirmed events to persist",
            10,
        ),
    ]
    return stages


def _build_markers(
    first_timestamp: Optional[float],
    person_first_seen: Dict[int, Tuple[int, float]],
    object_first_seen: Dict[int, Tuple[int, float]],
    confirmed_event_dicts: Dict[str, dict],
) -> List[dict]:
    """Build clickable timeline markers from real detector/track events."""
    def rel(ts: float) -> float:
        return round(float(ts - (first_timestamp or ts)), 2)

    markers: List[dict] = []
    for tid, (frame, ts) in sorted(person_first_seen.items(), key=lambda kv: kv[1][0])[:5]:
        markers.append({"label": f"PERSON #{tid} FIRST SEEN", "frame": frame, "timestamp": rel(ts), "kind": "person"})
    for tid, (frame, ts) in sorted(object_first_seen.items(), key=lambda kv: kv[1][0])[:5]:
        markers.append({"label": f"OBJECT #{tid} FIRST SEEN", "frame": frame, "timestamp": rel(ts), "kind": "object"})

    state_labels = {
        "carry_start": "BAG CARRIED",
        "release": "RELEASE",
        "ground": "GROUND",
        "departure": "DEPARTURE",
        "confirmed": "EVENT",
    }
    for event in confirmed_event_dicts.values():
        frames = event.get("frames") or {}
        timestamps = event.get("timestamps") or {}
        for key, label in state_labels.items():
            frame = frames.get(key)
            ts = timestamps.get(key)
            if frame is None or ts is None:
                continue
            markers.append({
                "label": label,
                "frame": int(frame),
                "timestamp": rel(float(ts)),
                "kind": "event",
                "event_id": event.get("event_id"),
                "person_track_id": event.get("person_track_id"),
                "object_track_id": event.get("bag_track_id"),
            })
    return sorted(markers, key=lambda m: m["timestamp"])


def _build_trajectories(confirmed_event_dicts: Dict[str, dict], frame_records: List[dict], first_timestamp: Optional[float]) -> List[dict]:
    """Extract real person/object trajectories for confirmed event windows."""
    def rel(ts: float) -> float:
        return round(float(ts - (first_timestamp or ts)), 2)

    trajectories: List[dict] = []
    for event in confirmed_event_dicts.values():
        frames = event.get("frames") or {}
        start = frames.get("carry_start") or frames.get("release") or 0
        end = frames.get("confirmed") or frames.get("departure") or frames.get("ground") or start
        pid = str(event.get("person_track_id"))
        oid = str(event.get("bag_track_id"))
        person_pts: List[List[float]] = []
        object_pts: List[List[float]] = []
        for rec in frame_records:
            fr = int(rec.get("frame_number", 0))
            if fr < int(start) or fr > int(end):
                continue
            ts = rel(float(rec.get("timestamp", 0.0)))
            for p in rec.get("persons", []):
                if str(p.get("track_id")) == pid:
                    person_pts.append([ts, round(float(p["bbox"][0]), 1), round(float(p["bbox"][1]), 1), round(float(p["bbox"][2]), 1), round(float(p["bbox"][3]), 1)])
            for o in rec.get("objects", []):
                if str(o.get("track_id")) == oid:
                    object_pts.append([ts, round(float(o["bbox"][0]), 1), round(float(o["bbox"][1]), 1), round(float(o["bbox"][2]), 1), round(float(o["bbox"][3]), 1)])
        trajectories.append({
            "event_id": event.get("event_id"),
            "person_track_id": event.get("person_track_id"),
            "object_track_id": event.get("bag_track_id"),
            "person": person_pts[-120:],
            "object": object_pts[-120:],
        })
    return trajectories


def _run_video_analysis_job(job_id: int):
    """Background task running the REAL production AI pipeline on the uploaded video."""
    from backend.database import get_db
    # Use the configured get_db generator (or dependency override if testing)
    db_gen = get_db()
    db: Session = next(db_gen)
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        try:
            next(db_gen, None)
        except Exception:
            pass
        return

    try:
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        job.completed_at = None
        db.commit()

        # Import AI pipeline components
        import cv2
        from backend.routers.evidence import EVIDENCE_STORE
        from inference.capture.camera_source import VideoFileSource
        from inference.detection.yolo_detector import YoloDetector
        from inference.pose.movenet_pose import MovenetPose
        from inference.tracking.bytetrack_tracker import BytetrackTracker
        from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
        from inference.pipeline import InferencePipeline, PipelineConfig
        from inference.visualization import build_frame_analysis, render_analysis_frame
        from inference.visualization.evidence_package import write_event_evidence_package
        from scripts.run_pipeline import build_tracks_real
        from littering_event_detector import annotate_detector_frame

        # Ensure Camera record exists for "Uploaded Video"
        cam = db.query(models.Camera).filter(models.Camera.name == f"Video: {job.original_filename}").first()
        if not cam:
            cam = models.Camera(
                name=f"Video: {job.original_filename}",
                location="Uploaded File Analysis",
                status="active"
            )
            db.add(cam)
            db.commit()
            db.refresh(cam)

        # Resolve the source video portably (host or container) so a job can be
        # re-run from stored files and old analyses can re-open their original.
        raw_path = job.original_video_path or job.file_path
        path = _resolve_archive_path(raw_path)
        if path is None:
            # Fall back to the raw path only if it is already absolute and exists
            # (legacy rows before the archive migration).
            candidate = Path(raw_path)
            if not (candidate.exists() and candidate.is_file()):
                job.status = "failed"
                job.error_message = f"Could not open video file {raw_path}"
                job.completed_at = datetime.now(timezone.utc)
                db.commit()
                db.close()
                return
            path = candidate

        source = VideoFileSource(str(path))
        if not source.open():
            job.status = "failed"
            job.error_message = f"Could not open video file {job.file_path}"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            db.close()
            return

        job.total_frames = source.total_frames
        job.fps = source.fps
        job.duration_sec = source.duration_seconds
        db.commit()

        # Load models
        detector = YoloDetector()
        detector.load()
        tracker = BytetrackTracker()
        tracker.load()

        # Class-agnostic NOVELTY (scene-change) detector — additive, separate
        # source. Reads NOVELTY_* from config/events.yaml; NOVELTY_ENABLED gates it.
        nov_cfg = NoveltyConfig.from_yaml()
        nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
        movenet = MovenetPose()
        movenet.load()

        pipeline_cfg = PipelineConfig(
            buffer_seconds=8.0,
            analysis_fps=8.0,
            camera_id=str(cam.id),
            # The job runs INSIDE the backend process and persists events +
            # evidence directly to the DB below (finalize -> verify -> create
            # event -> store files). An HTTP self-call would be fragile and
            # is unnecessary here. Live-camera mode keeps the HTTP path.
            post_backend_url=None,
            # Adaptive self-tuning: concurrent tier ladder + online learning
            # (adaptive_tuner.py). Learning is recorded per uploaded video at
            # finalize() (below) into learning/learning.json; the tier ladder
            # rescues threshold-brittle videos WITHOUT touching events that
            # the production config already confirms.
            auto_tune=True,
        )
        pipe = InferencePipeline(pipeline_cfg)
        pipe.event_detector.reset()
        # Attribute the learning record to the uploaded video file.
        try:
            pipe.event_detector.learning_video = job.original_filename
        except Exception:
            pass

        analysis_dir = EVIDENCE_STORE / "analysis" / str(job.id)
        analysis_dir.mkdir(parents=True, exist_ok=True)
        analyzed_path = analysis_dir / "analyzed.mp4"
        writer: Optional[cv2.VideoWriter] = None
        writer_fps = float(source.fps or 30.0)

        t_start = time.time()
        frame_idx = 0
        last_timestamp = 0.0
        first_timestamp: Optional[float] = None
        detected_persons_set = set()
        detected_objects_set = set()
        object_sources = set()
        history_timeline = []
        frame_records: List[dict] = []
        person_first_seen: Dict[int, Tuple[int, float]] = {}
        object_first_seen: Dict[int, Tuple[int, float]] = {}
        active_events: List[dict] = []
        confirmed_event_dicts: Dict[str, dict] = {}
        annotated = None
        inference_time = 0.0
        render_time = 0.0
        encode_time = 0.0

        def _rel(ts: float) -> float:
            return float(ts - (first_timestamp or ts))

        current_state = "NO_BAG"  # safe default for the report stepper (if zero frames)
        # Cache of the most recent ANALYSIS tick's detections, reused on the
        # skipped (non-analysis) frames so the video overlay stays populated
        # without re-running the heavy AI path. See the loop below.
        _last_persons: list = []
        _last_objects: list = []
        _last_event = None
        for pkt in source:
            if first_timestamp is None:
                first_timestamp = float(pkt.timestamp)

            # ---- Detection/tracking run on EVERY source frame. The temporal
            # layers (ByteTrack person ids, HSV color-bag tracker ids, and the
            # pipeline's frame-index bookkeeping) require full-rate input:
            # sampling them at analysis ticks caused bag-ID churn, so the
            # association latched onto a spurious track and the state machine
            # stalled at BAG_CARRIED with NO_RELEASE_TRANSITION (regression on
            # IMG_5117; audit evidence: .audit/runs_nov_cont vs DB jobs 58/59).
            # The EXPENSIVE per-frame work (MoveNet pose + the event detector)
            # stays throttled to analysis_fps via run_pose and process_frame's
            # internal gate — that is where the CPU savings live.
            infer_start = time.time()
            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker, frame_idx,
                run_pose=run_pose, nov=nov,
            )

            # process_frame buffers EVERY frame (evidence clips need full-rate
            # input) and internally throttles the event detector to analysis_fps.
            new_events = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)

            for p in persons:
                detected_persons_set.add(p.track_id)
                person_first_seen.setdefault(int(p.track_id), (frame_idx, float(pkt.timestamp)))
            for o in objects:
                detected_objects_set.add(f"{o.track_id}:{o.class_name}")
                object_first_seen.setdefault(int(o.track_id), (frame_idx, float(pkt.timestamp)))
                object_sources.add(str(getattr(o, "source", "yolo") or "yolo"))

            # Register newly confirmed detector events for the visual banner.
            for ev in new_events:
                dev = next((d for d in pipe.event_detector.confirmed_events if d.event_id == ev.event_id), None)
                if dev is None:
                    continue
                event_dict = _compact_detector_event(dev)
                event_dict["relative_timestamp"] = round(_rel(float(event_dict.get("timestamps", {}).get("confirmed") or pkt.timestamp)), 2)
                event_dict["frame"] = int(event_dict.get("frames", {}).get("confirmed") or frame_idx)
                confirmed_event_dicts[ev.event_id] = event_dict
                active_events.append(event_dict)

            # Keep only events whose banner should still be visible.
            active_events = [
                e for e in active_events
                if 0.0 <= _rel(float(e.get("timestamps", {}).get("confirmed") or pkt.timestamp)) <= _rel(pkt.timestamp) + 2.0
            ]
            current_event = active_events[-1] if active_events else None
            inference_time += time.time() - infer_start
            _last_persons, _last_objects, _last_event = persons, objects, current_event

            current_state = _current_detector_state(pipe.event_detector)
            if current_state != "NO_BAG" and (not history_timeline or history_timeline[-1]["state"] != current_state):
                history_timeline.append({
                    "timestamp": round(_rel(pkt.timestamp), 2),
                    "frame": frame_idx,
                    "state": current_state,
                })

            analysis = build_frame_analysis(
                persons,
                objects,
                pipe.event_detector,
                tracker,
                float(pkt.timestamp),
                frame_idx,
                source_fps=float(source.fps or 0.0),
                analysis_fps=float(pipe.config.analysis_fps),
                video_name=job.original_filename,
                event=current_event,
            )
            if run_pose:
                frame_records.append(analysis.to_dict())

            render_start = time.time()
            try:
                annotated = render_analysis_frame(pkt.frame, analysis, event=current_event)
            except Exception:
                log.exception("Job %s: visualization renderer failed at frame %s", job.id, frame_idx)
                try:
                    det_persons = [pipe._detector_person(p) for p in persons]
                    det_bags = [pipe._detector_bag(o) for o in objects]
                    annotated = annotate_detector_frame(pkt.frame, det_persons, det_bags, detector=pipe.event_detector)
                except Exception:
                    # Graceful degradation: if every renderer failed (e.g. the
                    # process is memory-starved and even a 2.6 MiB frame.copy()
                    # cannot be allocated), write the ORIGINAL decoded frame
                    # WITHOUT copying it. We must not allocate a fresh buffer
                    # here, otherwise the OOM that triggered this path simply
                    # recurs and the whole analysis job dies. The analyzed video
                    # still completes -- that frame just lacks the AI overlay.
                    annotated = pkt.frame
            render_time += time.time() - render_start

            encode_start = time.time()
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(analyzed_path), fourcc, writer_fps, (w, h))
                if not writer.isOpened():
                    log.warning("Job %s: could not open analyzed-video writer for %s", job.id, analyzed_path)
                    writer = None
            if writer is not None:
                writer.write(annotated)
            encode_time += time.time() - encode_start

            last_timestamp = float(pkt.timestamp)
            frame_idx += 1
            if frame_idx % 15 == 0:
                job.processed_frames = frame_idx
                job.persons_detected = len(detected_persons_set)
                job.objects_detected = len(detected_objects_set)
                job.events_count = len(pipe.events)
                elapsed = max(1e-6, time.time() - t_start)
                job.processing_fps = round(frame_idx / elapsed, 1)
                db.commit()
            # Periodically release Python/torch/TF debris so a single long video
            # analysis does not accumulate memory until the encoder OOMs.
            if frame_idx % 30 == 0:
                gc.collect()

        source.release()
        if writer is not None:
            writer.release()

        # Flush detector candidates that were still open when the video ended.
        final_events = pipe.finalize(last_timestamp)
        for ev in final_events:
            dev = next((d for d in pipe.event_detector.confirmed_events if d.event_id == ev.event_id), None)
            if dev is None:
                continue
            event_dict = _compact_detector_event(dev)
            event_dict["relative_timestamp"] = round(_rel(float(event_dict.get("timestamps", {}).get("confirmed") or last_timestamp)), 2)
            event_dict["frame"] = int(event_dict.get("frames", {}).get("confirmed") or frame_idx)
            confirmed_event_dicts[ev.event_id] = event_dict

        # ------------------------------------------------------------------
        # Persist confirmed candidates: finalize -> verify -> create event
        # -> store evidence files -> (retrievable via /api/evidence/*).
        # The pipeline already finalized + verified each artifact's files
        # during the frame loop; pipe.finalized_artifacts holds only the
        # artifacts whose snapshot/video exist and are non-empty.
        # ------------------------------------------------------------------
        # PHASE 2: after the loop, the analyzed video is re-encoded from
        # mp4v (which browsers cannot decode) to H.264 so the dashboard's
        # "AI Analyzed Video" panel actually plays. Evidence packages are
        # now built from the ORIGINAL video (clean frames) — see
        # write_event_evidence_package.
        from inference.visualization.h264 import transcode_to_h264

        # Re-encode the full analyzed overlay video for browser playback.
        # ~700MB mp4v for a 5k-frame video becomes ~100-200MB H.264; done
        # once per job, not per event, before the persistence loop below
        # (which references analyzed_path only for existence checks).
        if analyzed_path.exists() and analyzed_path.stat().st_size > 0:
            transcode_to_h264(str(analyzed_path), timeout_sec=3600)

        created_event_ids = []
        evidence_packages: Dict[str, Dict[str, Optional[str]]] = {}
        for ev in pipe.events:
            art = pipe.finalized_artifacts.get(ev.event_id)
            if art is None:
                log.warning(
                    "Job %s: event %s has no verified evidence artifact — "
                    "skipping DB persistence", job.id, ev.event_id
                )
                continue
            snap_path = art.snapshot_path
            vid_path = art.video_path
            snap_ok = snap_path and os.path.exists(snap_path) and os.path.getsize(snap_path) > 0
            vid_ok = vid_path and os.path.exists(vid_path) and os.path.getsize(vid_path) > 0
            if not snap_ok:
                log.error("Job %s: snapshot missing/empty for %s", job.id, ev.event_id)
                continue

            # 1) create the backend Event row with stable event-actor/object ownership (spec #9).
            # WRITE-PATH HARDENING: a confirmed event must never be persisted
            # without its frozen event-actor / event-object stable IDs (Phase C
            # contract). If the detector event dict is missing them, this is an
            # invariant violation — refuse to persist and surface it loudly.
            det_ev = confirmed_event_dicts.get(ev.event_id) if ev.event_id in confirmed_event_dicts else None
            actor_uid = det_ev.get("event_actor_person_uid") if det_ev else None
            actor_track = det_ev.get("event_actor_person_track_id") if det_ev else None
            object_uid = det_ev.get("event_object_uid") if det_ev else None
            object_track = det_ev.get("event_object_track_id") if det_ev else None
            assert actor_uid is not None and object_uid is not None, (
                f"WRITE-PATH INVARIANT VIOLATION for job {job.id} event {ev.event_id}: "
                f"confirmed event is missing stable identity fields "
                f"(event_actor_person_uid={actor_uid!r}, event_object_uid={object_uid!r}); "
                f"refusing to persist a NULL-identity event"
            )
            db_event = models.Event(
                camera_id=cam.id,
                person_track_id=str(ev.person_track_id),
                object_track_id=str(ev.object_track_id),
                object_type=ev.object_type,
                confidence=float(ev.confidence),
                timestamp=datetime.now(timezone.utc),
                status="confirmed",
                analysis_job_id=job.id,
                event_actor_person_track_id=actor_track,
                event_actor_person_uid=actor_uid,
                event_object_track_id=object_track,
                event_object_uid=object_uid,
            )
            db.add(db_event)
            db.commit()
            db.refresh(db_event)
            created_event_ids.append(db_event.id)

            # 2) copy evidence files into the backend evidence store and
            #    create the Evidence row so the dashboard can play them.
            target_dir = EVIDENCE_STORE / str(db_event.id)
            target_dir.mkdir(parents=True, exist_ok=True)
            rel_snapshot = rel_video = rel_person = rel_waste = rel_clip = rel_face = None
            rel_carry = rel_release = rel_ground = None
            if snap_ok:
                dst_snap = target_dir / f"snapshot_{db_event.id}.jpg"
                shutil.copyfile(snap_path, dst_snap)
                rel_snapshot = str(dst_snap.relative_to(EVIDENCE_STORE))
            if vid_ok:
                dst_vid = target_dir / f"evidence_{db_event.id}.mp4"
                shutil.copyfile(vid_path, dst_vid)
                rel_video = str(dst_vid.relative_to(EVIDENCE_STORE))

            # 3) generate the professional evidence package from the real
            #    ORIGINAL video + real frame-analysis records. The original
            #    is the clean (no-overlay) source the pipeline decoded, so
            #    snapshot/crops show the event itself instead of stacked
            #    tracking boxes from the whole analyzed overlay.
            detector_event = confirmed_event_dicts.get(ev.event_id)
            if detector_event is not None and analyzed_path.exists():
                try:
                    pkg = write_event_evidence_package(
                        original_video_path=str(path),
                        target_dir=str(target_dir),
                        event=detector_event,
                        frame_records=frame_records,
                        job_id=job.id,
                        original_filename=job.original_filename,
                        source_fps=writer_fps,
                        pre_seconds=5.0,
                        post_seconds=5.0,
                        analyzed_video_path=str(analyzed_path),
                    )
                    evidence_packages[ev.event_id] = pkg
                    if pkg.get("snapshot"):
                        rel_snapshot = str(Path(pkg["snapshot"]).relative_to(EVIDENCE_STORE))
                    if pkg.get("person"):
                        rel_person = str(Path(pkg["person"]).relative_to(EVIDENCE_STORE))
                    if pkg.get("waste"):
                        rel_waste = str(Path(pkg["waste"]).relative_to(EVIDENCE_STORE))
                    if pkg.get("clip"):
                        rel_clip = str(Path(pkg["clip"]).relative_to(EVIDENCE_STORE))
                    if pkg.get("face"):
                        rel_face = str(Path(pkg["face"]).relative_to(EVIDENCE_STORE))
                    # P1-2: persist the temporal sequence stills when written
                    if pkg.get("carry"):
                        rel_carry = str(Path(pkg["carry"]).relative_to(EVIDENCE_STORE))
                    if pkg.get("release"):
                        rel_release = str(Path(pkg["release"]).relative_to(EVIDENCE_STORE))
                    if pkg.get("ground"):
                        rel_ground = str(Path(pkg["ground"]).relative_to(EVIDENCE_STORE))
                except Exception:
                    log.exception("Job %s: evidence package generation failed for event %s", job.id, ev.event_id)

            db_evidence = models.Evidence(
                event_id=db_event.id,
                image_path=rel_snapshot,
                video_path=rel_video,
                person_image_path=rel_person,
                waste_image_path=rel_waste,
                clip_path=rel_clip,
                face_image_path=rel_face,
                carry_image_path=rel_carry,
                release_image_path=rel_release,
                ground_image_path=rel_ground,
                duration_sec=art.duration_seconds,
            )
            db.add(db_evidence)
            db.commit()
            log.info(
                "Job %s: persisted event #%d (%s) with evidence%s",
                job.id, db_event.id, ev.object_type,
                " (snapshot+video)" if (snap_ok and vid_ok) else " (snapshot only)",
            )

        elapsed = max(1e-6, time.time() - t_start)
        job.processed_frames = frame_idx
        job.processing_fps = round(frame_idx / elapsed, 1)
        performance = {
            "total_fps": round(frame_idx / elapsed, 2),
            "inference_fps": round(frame_idx / max(1e-6, inference_time), 2),
            "render_fps": round(frame_idx / max(1e-6, render_time), 2),
            "encode_fps": round(frame_idx / max(1e-6, encode_time), 2),
            "inference_sec": round(inference_time, 2),
            "render_sec": round(render_time, 2),
            "encode_sec": round(encode_time, 2),
        }
        job.persons_detected = len(detected_persons_set)
        job.objects_detected = len(detected_objects_set)
        job.events_count = len(pipe.events)
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        if analyzed_path.exists() and analyzed_path.stat().st_size > 0:
            job.analyzed_video_path = str(analyzed_path.relative_to(EVIDENCE_STORE))
        else:
            job.analyzed_video_path = None

        # Write the structured frame-analysis contract for external review.
        frame_analysis_path = analysis_dir / "frames.jsonl"
        try:
            with frame_analysis_path.open("w", encoding="utf-8") as f:
                for rec in frame_records:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        except Exception:
            log.exception("Job %s: failed to write frame analysis JSONL", job.id)

        # Build diagnostic report
        detector_summary = pipe.event_detector.summary()
        no_candidate_reason = None
        if not pipe.events:
            no_candidate_reason = _no_candidate_reason(detector_summary, len(detected_persons_set), len(detected_objects_set))
        markers = _build_markers(first_timestamp, person_first_seen, object_first_seen, confirmed_event_dicts)
        trajectories = _build_trajectories(confirmed_event_dicts, frame_records, first_timestamp)
        report = {
            "source": job.original_filename,
            "duration_sec": round(source.duration_seconds, 2),
            "total_frames": source.total_frames,
            "processed_frames": frame_idx,
            "processing_fps": job.processing_fps,
            "performance": performance,
            "persons_count": len(detected_persons_set),
            "objects_count": len(detected_objects_set),
            "object_sources": sorted(object_sources),
            "detector_source": _dominant_detector_source(object_sources),
            "confirmed_events": len(pipe.events),
            "persisted_event_ids": created_event_ids,
            "analyzed_video": bool(job.analyzed_video_path),
            "timeline": history_timeline,
            "markers": markers,
            "trajectories": trajectories,
            "frame_analysis_count": len(frame_records),
            "evidence_packages": evidence_packages,
            "no_candidate_reason": no_candidate_reason,
            "diagnosis": {
                "yolo_person": "PASS" if detected_persons_set else "FAIL",
                "yolo_object": "PASS" if "yolo" in object_sources else "FAIL",
                "color_object": "PASS" if "color" in object_sources else "FAIL",
                "detector_source": _dominant_detector_source(object_sources),
                "tracking": "PASS" if tracker.store_size > 0 else "FAIL",
                "association": "PASS" if detector_summary.get("total_candidates", 0) > 0 else "FAIL",
                "littering_candidate": "CONFIRMED" if pipe.events else "NO_CANDIDATE_DETECTED",
                "no_candidate_reason": no_candidate_reason,
                "event_detector_confirmed": detector_summary.get("confirmed_violations", 0),
                "event_detector_rejected": detector_summary.get("rejected_candidates", 0),
            },
            "stages": _build_stages(
                processed_frames=frame_idx,
                persons_count=len(detected_persons_set),
                objects_count=len(detected_objects_set),
                object_sources=object_sources,
                tracking_ok=tracker.store_size > 0,
                detector_summary=detector_summary,
                confirmed_events=len(pipe.events),
                created_event_ids=created_event_ids,
                evidence_packages=evidence_packages,
                current_state=current_state,
            ),
            "event_detector": {
                "summary": detector_summary,
                "config": {
                    "analysis_fps": pipe.config.event_detector_config.analysis_fps,
                    "detection_low_conf": pipe.config.event_detector_config.detection_low_conf,
                    "detection_high_conf": pipe.config.event_detector_config.detection_high_conf,
                    "min_carried_frames": pipe.config.event_detector_config.min_carried_frames,
                    "min_stationary_frames": pipe.config.event_detector_config.min_stationary_frames,
                    "near_distance_ratio": pipe.config.event_detector_config.near_distance_ratio,
                    "departure_distance_ratio": pipe.config.event_detector_config.departure_distance_ratio,
                    "departure_motion_ratio": pipe.config.event_detector_config.departure_motion_ratio,
                    "stationary_distance_ratio": pipe.config.event_detector_config.stationary_distance_ratio,
                    "stationary_window_frames": pipe.config.event_detector_config.stationary_window_frames,
                    "min_event_confidence": pipe.config.event_detector_config.min_event_confidence,
                    "max_fallback_tracker_gap_frames": pipe.config.event_detector_config.max_fallback_tracker_gap_frames,
                },
                "confirmed_violations": [_compact_detector_event(e) for e in pipe.event_detector.confirmed_events[:5]],
                "rejected_candidates": [_compact_detector_event(e) for e in pipe.event_detector.rejected_events[:5]],
            },
        }
        job.report_json = _fit_report_json(report, max_chars=200000)

        # ------------------------------------------------------------------
        # Persistent-analysis manifest — analysis_id-keyed, immutable snapshot
        # of every artifact (original/analyzed/frames/events + timestamps +
        # sizes). Written both to disk (evidence_store/analysis/<id>/manifest.json)
        # and into the DB row so the analysis reopens without re-running AI.
        # ------------------------------------------------------------------
        try:
            manifest = _build_analysis_manifest(
                analysis_id=job.id,
                original_filename=job.original_filename,
                original_rel_path=job.original_video_path,
                analyzed_rel_path=job.analyzed_video_path,
                frames_rel_path=f"evidence_store/analysis/{job.id}/frames.jsonl",
                duration_sec=job.duration_sec,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                status=job.status,
                events_count=len(pipe.events),
                confirmed_event_dicts=confirmed_event_dicts,
                evidence_packages=evidence_packages,
                timeline=history_timeline,
                markers=markers,
                no_candidate_reason=no_candidate_reason,
                detector_summary=detector_summary,
                persons_count=len(detected_persons_set),
                objects_count=len(detected_objects_set),
                processed_frames=frame_idx,
                source_fps=source.fps,
                width=source.width,
                height=source.height,
            )
            job.manifest_json = json.dumps(manifest, ensure_ascii=False, default=str)
            manifest_path = analysis_dir / "manifest.json"
            manifest_path.write_text(job.manifest_json, encoding="utf-8")
        except Exception:
            log.exception("Job %s: failed to write analysis manifest", job.id)

        db.commit()
        log.info("Completed analysis job %s with %d events", job.id, len(pipe.events))

    except Exception as e:
        log.exception("Analysis job %s failed: %s", job_id, e)
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
    finally:
        try:
            next(db_gen, None)
        except Exception:
            pass


@router.post("/upload", response_model=schemas.VideoAnalysisJobOut, status_code=status.HTTP_201_CREATED)
async def upload_video_for_analysis(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a real video file (.mp4, .avi, .mov, .mkv, .webm) for real AI pipeline analysis."""
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video format '{ext}'. Allowed: .mp4, .avi, .mov, .mkv, .webm"
        )

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp_str}_{file.filename or 'video.mp4'}"
    target_path = UPLOAD_DIR / safe_name

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty")

    target_path.write_bytes(content)

    # Store the original-upload path RELATIVE to the repo root so it resolves
    # both on the host (D:\HO\...) and in the container (/app/...) after a
    # restart. Absolute host paths would break reopening old analyses.
    original_rel_path = str(target_path.relative_to(REPO_ROOT))

    job = models.VideoAnalysisJob(
        filename=safe_name,
        original_filename=file.filename or safe_name,
        file_path=str(target_path),
        original_video_path=original_rel_path,
        status="queued",
        processed_frames=0
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Launch real analysis job in background
    background_tasks.add_task(_run_video_analysis_job, job.id)

    return job


def _build_analysis_manifest(
    *,
    analysis_id: int,
    original_filename: str,
    original_rel_path: Optional[str],
    analyzed_rel_path: Optional[str],
    frames_rel_path: Optional[str],
    duration_sec: Optional[float],
    created_at,
    started_at,
    completed_at,
    status: str,
    events_count: int,
    confirmed_event_dicts: Dict[str, dict],
    evidence_packages: Dict[str, Dict[str, Optional[str]]],
    timeline: List[dict],
    markers: List[dict],
    no_candidate_reason: Optional[str],
    detector_summary: dict,
    persons_count: int,
    objects_count: int,
    processed_frames: int,
    source_fps: Optional[float],
    width: Optional[int],
    height: Optional[int],
    error_message: Optional[str] = None,
) -> dict:
    """Build the per-analysis artifact manifest (analysis_id-keyed, immutable).

    This is what makes an analysis reopenable without re-running the AI: it
    enumerates the original/analyzed/frames/event/evidence paths (all stored
    relative to the repo root so they resolve on host AND in the container
    after a restart), plus timestamps, sizes, timeline, markers, and the
    detector verdict. Nothing here is derived from other analyses — each job
    gets its own manifest written to disk under evidence_store/analysis/<id>/.
    """
    def _size(rel: Optional[str]) -> Optional[int]:
        if not rel:
            return None
        p = _resolve_archive_path(rel)
        if p is None:
            return None
        try:
            return p.stat().st_size
        except OSError:
            return None

    original_size = _size(original_rel_path)
    analyzed_size = _size(analyzed_rel_path)
    frames_size = _size(frames_rel_path)

    events: List[dict] = []
    for ev in confirmed_event_dicts.values():
        events.append({
            "event_id": ev.get("event_id"),
            "person_track_id": ev.get("person_track_id"),
            "bag_track_id": ev.get("bag_track_id"),
            "confidence": ev.get("confidence"),
            "state": ev.get("state"),
            "reason": ev.get("reason"),
            "frames": ev.get("frames"),
            "timestamps": ev.get("timestamps"),
        })

    manifest = {
        "analysis_id": analysis_id,
        "job_id": analysis_id,
        "original_filename": original_filename,
        "original_video": original_rel_path,
        "analyzed_video": analyzed_rel_path,
        "frames_jsonl": frames_rel_path,
        "event_clips": [
            {
                "event_id": ev.get("event_id"),
                "evidence_dir": (
                    str(Path(p.get("metadata", "")).parent)
                    if p.get("metadata") else None
                ),
                "snapshot": p.get("snapshot"),
                "person": p.get("person"),
                "waste": p.get("waste"),
                "carry": p.get("carry"),
                "release": p.get("release"),
                "ground": p.get("ground"),
                "clip": p.get("clip"),
                "face": p.get("face"),
            }
            for ev, p in zip(confirmed_event_dicts.values(), evidence_packages.values())
        ],
        "events": events,
        "timeline": timeline,
        "markers": markers,
        "metadata": {
            "duration_sec": duration_sec,
            "source_fps": source_fps,
            "resolution": [width, height] if (width and height) else None,
            "processed_frames": processed_frames,
            "persons_count": persons_count,
            "objects_count": objects_count,
            "detector_summary": detector_summary,
            "no_candidate_reason": no_candidate_reason,
            "status": status,
            "error_message": error_message,
            "created_at": created_at.isoformat() if created_at else None,
            "started_at": started_at.isoformat() if started_at else None,
            "completed_at": completed_at.isoformat() if completed_at else None,
        },
        "sizes_bytes": {
            "original": original_size,
            "analyzed": analyzed_size,
            "frames_jsonl": frames_size,
            "events_count": events_count,
        },
        "final_result": "LITTERING_EVENT_CANDIDATE" if events_count and events_count > 0 else "NO_EVENT",
    }
    return manifest


@router.get("/jobs", response_model=schemas.VideoAnalysisJobListOut)
def list_analysis_jobs(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all video analysis jobs with pagination."""
    q = db.query(models.VideoAnalysisJob).order_by(models.VideoAnalysisJob.id.desc())
    total = q.count()
    items = q.offset(offset).limit(limit).all()
    return schemas.VideoAnalysisJobListOut(items=items, total=total, limit=limit, offset=offset)


@router.get("/jobs/{job_id}", response_model=schemas.VideoAnalysisJobOut)
def get_analysis_job(job_id: int, db: Session = Depends(get_db)):
    """Retrieve details, progress, and diagnostic report for a specific analysis job."""
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return job


@router.get("/jobs/{job_id}/manifest", response_model=schemas.AnalysisManifestOut)
def get_analysis_manifest(job_id: int, db: Session = Depends(get_db)):
    """Serve the per-analysis artifact manifest, or build it on the fly for
    legacy rows that predate the archive columns.

    The manifest is how the history detail page reopens a stored analysis
    without re-running the AI: it lists original/analyzed/frames/event/evidence
    paths and the immutable timeline/metadata for THIS analysis_id only.
    """
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    if job.manifest_json:
        try:
            return json.loads(job.manifest_json)
        except Exception:
            pass
    # Legacy fallback: derive a manifest from the diagnostic report.
    try:
        report = json.loads(job.report_json or "{}")
    except Exception:
        report = {}
    return {
        "analysis_id": job.id,
        "job_id": job.id,
        "original_filename": job.original_filename,
        "original_video": job.original_video_path,
        "analyzed_video": job.analyzed_video_path,
        "frames_jsonl": f"evidence_store/analysis/{job.id}/frames.jsonl",
        "events": report.get("persisted_event_ids", []),
        "timeline": report.get("timeline", []),
        "markers": report.get("markers", []),
        "metadata": {
            "duration_sec": job.duration_sec,
            "source_fps": job.fps,
            "processed_frames": job.processed_frames,
            "persons_count": job.persons_detected,
            "objects_count": job.objects_detected,
            "no_candidate_reason": report.get("no_candidate_reason"),
            "status": job.status,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        },
        "final_result": "LITTERING_EVENT_CANDIDATE" if job.events_count and job.events_count > 0 else "NO_EVENT",
    }


@router.get("/jobs/{job_id}/events", response_model=List[schemas.EventOut])
def get_analysis_job_events(job_id: int, db: Session = Depends(get_db)):
    """Return every event persisted for a specific analysis job (job_id)."""
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    events = (
        db.query(models.Event)
        .filter(models.Event.analysis_job_id == job_id)
        .order_by(models.Event.id.asc())
        .all()
    )
    return [schemas.EventOut.model_validate(e).model_dump() for e in events]


@router.get("/jobs/{job_id}/analyzed-video")
def get_analyzed_video(job_id: int, db: Session = Depends(get_db)):
    """Serve the real analyzed video with inference overlays for a completed job."""
    from backend.routers.evidence import EVIDENCE_STORE

    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    if not job.analyzed_video_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analyzed video is not available")
    # analyzed_video_path is EVIDENCE_STORE-relative (evidence_store/analysis/<id>/analyzed.mp4).
    # Prefer direct evidence-store resolution (host/container volume), then the
    # portable repo-root fallback so historical analyses always reopen.
    direct = EVIDENCE_STORE / job.analyzed_video_path
    if direct.exists() and direct.stat().st_size > 0:
        return FileResponse(str(direct), media_type="video/mp4", filename=f"analyzed_{job_id}.mp4")
    rel = _resolve_archive_path(f"evidence_store/{job.analyzed_video_path}")
    if rel is None or rel.stat().st_size == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analyzed video file is missing")
    return FileResponse(str(rel), media_type="video/mp4", filename=f"analyzed_{job_id}.mp4")


@router.get("/jobs/{job_id}/frame-analysis")
def get_frame_analysis(job_id: int, db: Session = Depends(get_db)):
    """Return the structured FrameAnalysis JSONL produced by the real pipeline."""
    from backend.routers.evidence import EVIDENCE_STORE

    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    path = _resolve_archive_path(f"evidence_store/analysis/{job_id}/frames.jsonl")
    if path is None or path.stat().st_size == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame analysis data is not available")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return {"job_id": job_id, "count": len(records), "frames": records}


@router.get("/jobs/{job_id}/original-video")
def get_original_video(job_id: int, db: Session = Depends(get_db)):
    """Serve the uploaded original video for side-by-side comparison.

    Resolves the stored repo-root-relative path (host OR container) so the
    original remains viewable for every historical analysis after restarts —
    never only the latest job.
    """
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    path = _resolve_archive_path(job.original_video_path or job.file_path)
    if path is None or path.stat().st_size == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Original video file is missing")
    media = "video/quicktime" if path.suffix.lower() == ".mov" else "video/mp4"
    return FileResponse(str(path), media_type=media, filename=job.original_filename)
