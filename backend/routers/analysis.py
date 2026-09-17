"""Video Analysis Router — upload video files and run the real production AI pipeline."""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    p = Path(rel_path)
    if p.exists() and p.is_file():
        return p
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
    """Serialize report while keeping the legacy varchar(4096) column safe.

    Order matters: the FIRST thing we must never lose is the failure/stage
    telemetry, because a truncated report that drops ``stages`` /
    ``pipeline_telemetry`` makes the dashboard fall back to "Stage 1 —
    Video Input FAILED" even when the real failure was Stage 9 (temporal
    event detection). Compact the stage matrix BEFORE falling back to the
    minimal payload.
    """
    def _dump() -> str:
        return json.dumps(report, ensure_ascii=False)

    text = _dump()
    if len(text) <= max_chars:
        return text

    # Preserve the stage lifecycle in a compact form (id/order/status/error).
    telemetry = report.get("pipeline_telemetry") if isinstance(report, dict) else None
    raw_stages = report.get("stages") if isinstance(report, dict) else None
    if isinstance(raw_stages, list) or isinstance(raw_stages, dict):
        items = (
            raw_stages
            if isinstance(raw_stages, list)
            else [dict(v, id=k) if isinstance(v, dict) else {"id": k} for k, v in raw_stages.items()]
        )
        report["stages"] = [
            {
                "id": it.get("id"),
                "order": it.get("order"),
                "status": it.get("status"),
                "error": it.get("error"),
            }
            for it in items
            if isinstance(it, dict)
        ]
        if isinstance(telemetry, dict):
            report["pipeline_telemetry"] = {
                k: telemetry.get(k)
                for k in (
                    "current_stage",
                    "current_stage_id",
                    "current_stage_status",
                    "current_step",
                    "stage_step",
                    "stage_name_display",
                    "total_stages",
                    "last_successful_stage",
                    "failed_stage_id",
                    "status",
                    "processed_frames",
                    "total_frames",
                    "error_message",
                    "progress_pct",
                )
            }
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
        # Keep stage attribution even in the last-resort payload.
        "pipeline_telemetry": report.get("pipeline_telemetry"),
        "stages": report.get("stages"),
        "event_detector": {
            "summary": report.get("event_detector", {}).get("summary", {}) if isinstance(report.get("event_detector"), dict) else {},
            "truncated": True,
        },
    }
    text = json.dumps(minimal, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return json.dumps(
        {
            "truncated": True,
            "pipeline_telemetry": report.get("pipeline_telemetry"),
            "stages": report.get("stages"),
            "event_detector": {"summary": {}},
        },
        ensure_ascii=False,
    )


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


# --------------------------------------------------------------------------- #
# Thread-safe job cancellation tokens
# --------------------------------------------------------------------------- #
_JOB_CANCELLATION_TOKENS: Dict[int, threading.Event] = {}
_CANCELLATION_LOCK = threading.Lock()


def request_job_cancellation(job_id: int) -> bool:
    """Request graceful cancellation of a running background analysis job."""
    with _CANCELLATION_LOCK:
        token = _JOB_CANCELLATION_TOKENS.get(job_id)
        if token is not None:
            token.set()
            return True
        token = threading.Event()
        token.set()
        _JOB_CANCELLATION_TOKENS[job_id] = token
        return True


def is_job_cancellation_requested(job_id: int) -> bool:
    """Check if cancellation has been requested for a specific job_id."""
    with _CANCELLATION_LOCK:
        token = _JOB_CANCELLATION_TOKENS.get(job_id)
        return token.is_set() if token is not None else False


def _extract_person_identity_info(event_detector, detected_persons_set: set) -> Tuple[int, Dict[int, dict]]:
    """Safely extract unique persons count and identity mapping from detector without AttributeError."""
    try:
        p_id_mgr = getattr(event_detector, "_person_identity", None)
        if p_id_mgr is None and hasattr(event_detector, "primary"):
            p_id_mgr = getattr(event_detector.primary, "_person_identity", None)

        if p_id_mgr is not None and hasattr(p_id_mgr, "_uids") and p_id_mgr._uids:
            uids_map = {}
            for uid, snap in p_id_mgr._uids.items():
                raw_ids = getattr(snap, "raw_ids", [])
                last_frame = getattr(snap, "last_frame", -1)
                uids_map[uid] = {"raw_track_ids": raw_ids, "last_frame": last_frame}
            return len(p_id_mgr._uids), uids_map
    except Exception as e:
        log.warning("Could not extract person identity UIDs: %s", e)

    fallback_count = 1 if detected_persons_set else 0
    return fallback_count, {}


def _extract_object_identity_count(event_detector, detected_objects_set: set) -> int:
    """Safely extract unique objects count from ObjectIdentityManager without AttributeError."""
    try:
        obj_id_mgr = getattr(event_detector, "_object_identity", None)
        if obj_id_mgr is None and hasattr(event_detector, "primary"):
            obj_id_mgr = getattr(event_detector.primary, "_object_identity", None)

        if obj_id_mgr is not None:
            records = getattr(obj_id_mgr, "records", None)
            if records is not None:
                return len(records)
            uids = getattr(obj_id_mgr, "_uids", None)
            if uids is not None:
                return len(uids)
    except Exception as e:
        log.warning("Could not extract object identity count: %s", e)

    return len(detected_objects_set)


# --------------------------------------------------------------------------- #
# 12-Stage Real-Time Pipeline Tracker
# --------------------------------------------------------------------------- #
PIPELINE_STAGES = [
    {"id": "video_input", "name": "1. VIDEO INPUT", "order": 1},
    {"id": "person_detection", "name": "2. PERSON DETECTION", "order": 2},
    {"id": "waste_detection", "name": "3. OBJECT / WASTE DETECTION", "order": 3},
    {"id": "tracking", "name": "4. TRACKING", "order": 4},
    {"id": "person_identity", "name": "5. PERSON IDENTITY", "order": 5},
    {"id": "object_identity", "name": "6. OBJECT IDENTITY", "order": 6},
    {"id": "pose_estimation", "name": "7. POSE", "order": 7},
    {"id": "ownership_association", "name": "8. OWNERSHIP / ASSOCIATION", "order": 8},
    {"id": "temporal_event_detection", "name": "9. TEMPORAL EVENT DETECTION", "order": 9},
    {"id": "evidence_assembly", "name": "10. EVIDENCE", "order": 10},
    {"id": "database_persistence", "name": "11. DATABASE", "order": 11},
    {"id": "api_dashboard", "name": "12. API / DASHBOARD", "order": 12},
]


def _write_failure_forensics(
    *,
    job_id: int,
    video_name: Optional[str],
    stage_id: str,
    exc: BaseException,
    traceback_text: str,
    frame_idx: Optional[int],
    timestamp: Optional[float],
    source_fps: Optional[float],
    frame: Any = None,
    persons: Any = None,
    objects: Any = None,
    detector: Any = None,
    pair_states: Optional[List[dict]] = None,
) -> Dict[str, Any]:
    """Persist a real crash snapshot (JSON + annotated failure frame).

    Called ONLY from the analysis job's unexpected-exception handler. It never
    invents data: when the frame object is unavailable (crash before decode,
    or the frame was released) the image is skipped and the JSON records
    ``frame_image: null`` with ``nearest_frame: false``.
    """
    out: Dict[str, Any] = {
        "job_id": job_id,
        "video": video_name,
        "stage": stage_id,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback_text,
        "frame_index": frame_idx,
        "timestamp_sec": None if timestamp is None else round(float(timestamp), 3),
        "source_fps": source_fps,
        "detector": None,
        "entities": _entity_forensics(persons, objects),
        "pairs": pair_states or [],
        "frame_image": None,
        "nearest_frame": False,
    }

    if detector is not None:
        try:
            cfg = getattr(detector, "config", None)
            thresholds = None
            if cfg is not None:
                thresholds = {
                    k: getattr(cfg, k)
                    for k in dir(cfg)
                    if not k.startswith("_")
                    and isinstance(getattr(cfg, k, None), (int, float, str, type(None)))
                }
            out["detector"] = {
                "type": type(detector).__name__,
                "active_thresholds": thresholds,
            }
        except Exception as exc2:  # diagnostics must never raise
            out["detector"] = {"error": f"context capture failed: {exc2}"}

    try:
        dbg_dir = REPO_ROOT / "evidence_store" / "debug" / f"job_{job_id}"
        dbg_dir.mkdir(parents=True, exist_ok=True)
        if frame is not None:
            try:
                from littering_event_detector import annotate_detector_frame

                annotated = annotate_detector_frame(
                    frame,
                    list(persons or []),
                    list(objects or []),
                    detector=detector,
                )
                try:
                    import cv2  # type: ignore

                    banner = (
                        f"FAILED stage={stage_id} frame={frame_idx} "
                        f"t={out['timestamp_sec']}s {type(exc).__name__}"
                    )
                    cv2.putText(
                        annotated, banner, (10, max(20, annotated.shape[0] - 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                    )
                except Exception:
                    pass
                img_path = dbg_dir / "failure_frame.jpg"
                wrote = False
                try:
                    import cv2  # type: ignore

                    wrote = bool(cv2.imwrite(str(img_path), annotated))
                except Exception:
                    wrote = False
                out["frame_image"] = str(img_path) if wrote else None
                out["nearest_frame"] = not wrote
            except Exception as exc3:
                out["frame_image_error"] = str(exc3)
        (dbg_dir / "failure_context.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        out["context_path"] = str(dbg_dir / "failure_context.json")
    except Exception as exc4:
        out["forensics_error"] = str(exc4)

    log.error(
        "Job %s FAILURE FORENSICS stage=%s frame=%s t=%ss err=%s: %s",
        job_id, stage_id, frame_idx, out["timestamp_sec"], type(exc).__name__, exc,
    )
    return out


def _pair_forensics(detector: Any) -> List[dict]:
    """Snapshot every live FSM pair (ids, state, key measurements)."""
    rows: List[dict] = []
    try:
        for (pid, bid), mem in dict(getattr(detector, "_pairs", {}) or {}).items():
            rows.append({
                "person_track_id": int(pid),
                "bag_track_id": int(bid),
                "bag_uid": getattr(mem, "bag_uid", None),
                "person_uid": getattr(mem, "person_uid", None),
                "state": getattr(getattr(mem, "state", None), "value", str(getattr(mem, "state", None))),
                "carried_frames": getattr(mem, "carried_frames", None),
                "stationary_frames": getattr(mem, "stationary_frames", None),
                "departed_frames": getattr(mem, "departed_frames", None),
                "ground_evidence_frames": getattr(mem, "ground_evidence_frames", None),
                "bin_zone_frames": getattr(mem, "bin_zone_frames", None),
                "max_post_release_norm_distance": getattr(mem, "max_post_release_norm_distance", None),
                "separated_frames": getattr(mem, "separated_frames", None),
                "aidm_separated": getattr(mem, "aidm_separated", None),
                "ever_aidm_attached": getattr(mem, "ever_aidm_attached", None),
                "release_frame": getattr(mem, "release_frame", None),
                "ground_frame": getattr(mem, "ground_frame", None),
            })
    except Exception:
        pass
    return rows


def _entity_forensics(persons: Any, objects: Any) -> Dict[str, Any]:
    """Snapshot active person/object tracks with bboxes, class, confidence, UID."""
    def _rows(items):
        rows = []
        for t in list(items or []):
            rows.append({
                "track_id": int(getattr(t, "track_id", -1)),
                "bbox": list(getattr(t, "bbox", []) or []),
                "confidence": getattr(t, "confidence", None),
                "class_name": getattr(t, "class_name", None),
                "source": getattr(t, "source", None),
                "object_uid": getattr(t, "object_uid", None),
            })
        return rows

    return {"persons": _rows(persons), "objects": _rows(objects)}


class PipelineStageTracker:
    def __init__(self):
        self.stages: Dict[str, dict] = {
            s["id"]: {
                "id": s["id"],
                "name": s["name"],
                "order": s["order"],
                "status": "PENDING",  # PENDING, RUNNING, COMPLETED, FAILED, SKIPPED, CANCELLED
                "detail": "Pending execution",
                "error": None,
                "started_at": None,
                "completed_at": None,
                "last_frame": None,
            }
            for s in PIPELINE_STAGES
        }
        self.current_stage_id: str = "video_input"
        self.last_successful_stage_id: Optional[str] = None
        self.failed_stage_id: Optional[str] = None
        self.error_message: Optional[str] = None

    def start_stage(self, stage_id: str, detail: str, frame_idx: Optional[int] = None):
        if stage_id in self.stages:
            self.current_stage_id = stage_id
            st = self.stages[stage_id]
            st["status"] = "RUNNING"
            st["detail"] = detail
            if not st["started_at"]:
                st["started_at"] = datetime.now(timezone.utc).isoformat()
            if frame_idx is not None:
                st["last_frame"] = frame_idx

    def update_stage_detail(self, stage_id: str, detail: str, frame_idx: Optional[int] = None):
        if stage_id in self.stages:
            st = self.stages[stage_id]
            st["detail"] = detail
            if frame_idx is not None:
                st["last_frame"] = frame_idx

    def complete_stage(self, stage_id: str, detail: str, frame_idx: Optional[int] = None):
        if stage_id in self.stages:
            st = self.stages[stage_id]
            st["status"] = "COMPLETED"
            st["detail"] = detail
            st["completed_at"] = datetime.now(timezone.utc).isoformat()
            if frame_idx is not None:
                st["last_frame"] = frame_idx
            self.last_successful_stage_id = stage_id

    def fail_stage(self, stage_id: str, error: str, frame_idx: Optional[int] = None):
        if stage_id in self.stages:
            st = self.stages[stage_id]
            st["status"] = "FAILED"
            st["error"] = error
            st["detail"] = f"FAILED: {error}"
            st["completed_at"] = datetime.now(timezone.utc).isoformat()
            if frame_idx is not None:
                st["last_frame"] = frame_idx
            self.failed_stage_id = stage_id
            self.error_message = error
            curr_order = st["order"]
            for other_id, other in self.stages.items():
                if other["order"] > curr_order and other["status"] in ("PENDING", "RUNNING"):
                    other["status"] = "SKIPPED"
                    other["detail"] = f"Skipped due to failure in {st['name']}"

    def cancel(self, reason: str = "Analysis stopped by user request", frame_idx: Optional[int] = None):
        curr = self.stages.get(self.current_stage_id)
        curr_order = 0
        if curr:
            curr_order = curr["order"]
            if curr["status"] == "RUNNING":
                curr["status"] = "CANCELLED"
                curr["detail"] = f"Cancelled: {reason}"
                curr["completed_at"] = datetime.now(timezone.utc).isoformat()
                if frame_idx is not None:
                    curr["last_frame"] = frame_idx
        for other_id, other in self.stages.items():
            if other["order"] > curr_order and other["status"] in ("PENDING", "RUNNING"):
                other["status"] = "SKIPPED"
                other["detail"] = "Skipped due to cancellation"

    def current_stage_display(self) -> str:
        curr = self.stages.get(self.current_stage_id)
        if not curr:
            return "Stage 1/12 — VIDEO INPUT"
        return f"Stage {curr['order']}/12 — {curr['name'].split('. ', 1)[-1]}"

    def last_successful_stage_display(self) -> Optional[str]:
        if not self.last_successful_stage_id:
            return None
        st = self.stages.get(self.last_successful_stage_id)
        if not st:
            return None
        return f"Stage {st['order']}/12 — {st['name'].split('. ', 1)[-1]}"

    def to_list(self) -> List[dict]:
        return sorted(list(self.stages.values()), key=lambda x: x["order"])

    def build_telemetry(
        self,
        *,
        job_id: int,
        status: str,
        processed_frames: int,
        total_frames: Optional[int],
        fps: Optional[float] = None,
        processing_fps: Optional[float] = None,
        duration_sec: Optional[float] = None,
        active_persons_count: int = 0,
        peak_concurrent_people: int = 0,
        unique_persons_count: int = 0,
        total_person_track_ids: int = 0,
        person_identity_uids: Optional[dict] = None,
        objects_count: int = 0,
        candidates_count: int = 0,
        confirmed_events_count: int = 0,
        rejected_events_count: int = 0,
        current_fsm_state: str = "NO_BAG",
        rejection_reasons: Optional[dict] = None,
        error_message: Optional[str] = None,
        elapsed_sec: float = 0.0,
    ) -> dict:
        total = total_frames or 0
        progress_pct = int(min(100, round((processed_frames / total) * 100))) if total > 0 else (100 if status == "completed" else 0)
        curr_stage_name = self.current_stage_display()
        curr_stage_obj = self.stages.get(self.current_stage_id, {})
        curr_status = curr_stage_obj.get("status", "PENDING")
        curr_order = curr_stage_obj.get("order", 1)
        curr_display_name = curr_stage_obj.get("name", "1. VIDEO INPUT")

        return {
            "job_id": job_id,
            "status": status,
            "current_stage": curr_stage_name,
            "current_stage_id": self.current_stage_id,
            "current_stage_status": curr_status,
            "current_step": curr_order,
            "stage_step": curr_order,
            "total_stages": 12,
            "stage_name_display": curr_display_name,
            "last_successful_stage": self.last_successful_stage_display(),
            "overall_progress_percent": progress_pct,
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "last_processed_frame": processed_frames,
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(elapsed_sec, 2),
            "processing_fps": processing_fps,
            "error_message": error_message or self.error_message,
            "person_metrics": {
                "active_people": active_persons_count,
                "peak_concurrent_people": peak_concurrent_people,
                "unique_people_seen": unique_persons_count,
                "total_track_ids": total_person_track_ids,
                "identity_mapping": person_identity_uids or {},
                "explanation": "Active people = currently in frame; Unique people seen = stable IDs from PersonIdentityManager; Total track IDs = raw ByteTrack tracker churn."
            },
            "active_persons_count": active_persons_count,
            "unique_persons_count": unique_persons_count,
            "total_person_track_ids": total_person_track_ids,
            "objects_count": objects_count,
            "event_metrics": {
                "candidates_count": candidates_count,
                "confirmed_events_count": confirmed_events_count,
                "rejected_events_count": rejected_events_count,
                "active_candidates_count": max(0, candidates_count - confirmed_events_count - rejected_events_count),
                "current_fsm_state": current_fsm_state,
                "rejection_reasons": rejection_reasons or {},
            },
            "candidates_count": candidates_count,
            "rejected_count": rejected_events_count,
            "confirmed_events": confirmed_events_count,
            "stages": self.to_list(),
            "pipeline_telemetry": {
                "current_stage": self.current_stage_id,
                "current_stage_id": self.current_stage_id,
                "current_stage_status": curr_status.lower(),
                "current_step": curr_order,
                "total_stages": 12,
                "last_successful_stage": self.last_successful_stage_id,
                "last_processed_frame": processed_frames,
                "total_frames": total_frames,
                "last_update_time": datetime.now(timezone.utc).isoformat(),
                "error": error_message or self.error_message,
                "stages": {s["id"]: {
                    "step": s["order"],
                    "name": s["id"],
                    "display_name": s["name"].split(". ", 1)[-1],
                    "status": s["status"].lower(),
                    "error": s.get("error"),
                    "detail": s.get("detail"),
                    "started_at": s.get("started_at"),
                    "completed_at": s.get("completed_at"),
                } for s in self.stages.values()},
                "metrics": {
                    "active_persons": active_persons_count,
                    "peak_concurrent_persons": peak_concurrent_people,
                    "unique_persons_seen": unique_persons_count,
                    "total_track_ids": total_person_track_ids,
                    "objects_detected": objects_count,
                    "candidates_count": candidates_count,
                    "confirmed_events_count": confirmed_events_count,
                    "rejected_events_count": rejected_events_count,
                    "active_fsm_state": current_fsm_state,
                    "elapsed_sec": round(elapsed_sec, 2),
                }
            },
        }


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
    stage_tracker: Optional[PipelineStageTracker] = None,
) -> List[dict]:
    """Build a structured 12-stage pipeline-progress stepper from REAL measured signals."""
    if stage_tracker is not None:
        return stage_tracker.to_list()

    tracker = PipelineStageTracker()
    tracker.complete_stage("video_input", f"{processed_frames} frames decoded")
    tracker.complete_stage("person_detection", f"{persons_count} person track(s) detected")
    tracker.complete_stage("waste_detection", f"{objects_count} object track(s); source={_dominant_detector_source(object_sources)}")
    tracker.complete_stage("tracking", "track identities maintained across frames" if tracking_ok else "tracking degraded")
    tracker.complete_stage("person_identity", f"Person identity consolidated ({persons_count} tracks)")
    tracker.complete_stage("object_identity", f"Object identity maintained ({objects_count} objects)")
    tracker.complete_stage("pose_estimation", "MoveNet keypoints (wrists) computed for detected persons")
    candidates = detector_summary.get("total_candidates", 0)
    tracker.complete_stage("ownership_association", f"{candidates} person-object candidate pair(s)")
    tracker.complete_stage("temporal_event_detection", f"most advanced state reached: {current_state}")
    tracker.complete_stage("evidence_assembly", f"{len(evidence_packages)} evidence package(s) assembled")
    tracker.complete_stage("database_persistence", "event records persisted to PostgreSQL" if created_event_ids else "no confirmed events to persist")
    tracker.complete_stage("api_dashboard", "Ready for forensic review")
    return tracker.to_list()


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

    stage_tracker = PipelineStageTracker()
    with _CANCELLATION_LOCK:
        _JOB_CANCELLATION_TOKENS[job_id] = threading.Event()

    try:
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        job.completed_at = None
        stage_tracker.start_stage("video_input", "Opening video file and decoding stream headers")
        init_telemetry = stage_tracker.build_telemetry(
            job_id=job.id,
            status="processing",
            processed_frames=0,
            total_frames=job.total_frames,
        )
        job.report_json = _fit_report_json(init_telemetry)
        db.commit()

        # Import AI pipeline components
        import cv2
        from backend.routers.evidence import EVIDENCE_STORE
        from backend.services.video_normalizer import ensure_cfr_source
        from inference.capture.camera_source import VideoFileSource
        from inference.detection.yolo_detector import YoloDetector
        from inference.pose.movenet_pose import MovenetPose
        from inference.tracking.bytetrack_tracker import BytetrackTracker
        from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
        from inference.pipeline import InferencePipeline, PipelineConfig
        from inference.runtime_determinism import (
            configure_determinism,
            determinism_enabled_from_env,
        )
        from inference.visualization import build_frame_analysis, render_analysis_frame
        from inference.visualization.evidence_package import write_event_evidence_package
        from scripts.run_pipeline import build_tracks_real
        from littering_event_detector import annotate_detector_frame

        deterministic = determinism_enabled_from_env()
        determinism_report = configure_determinism(0) if deterministic else {"configured": False}

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
                stage_tracker.fail_stage("video_input", f"Could not open video file {raw_path}")
                job.status = "failed"
                job.error_message = f"Could not open video file {raw_path}"
                job.completed_at = datetime.now(timezone.utc)
                job.report_json = _fit_report_json(stage_tracker.build_telemetry(
                    job_id=job.id,
                    status="failed",
                    processed_frames=0,
                    total_frames=None,
                    error_message=job.error_message,
                ))
                db.commit()
                db.close()
                return
            path = candidate

        # Per-job analysis workspace (CFR source + overlays live here).
        analysis_dir = EVIDENCE_STORE / "analysis" / str(job.id)
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # Section 2 — normalize VFR (iPhone) → CFR before OpenCV reads frames.
        # Pipeline always consumes the CFR file; the archived original stays
        # at job.original_video_path for side-by-side review.
        stage_tracker.update_stage_detail(
            "video_input",
            f"Re-encoding to constant frame rate (CFR) from {path.name}",
        )
        try:
            cfr_path = Path(ensure_cfr_source(path, analysis_dir, stem="source_CFR"))
            path = cfr_path
        except Exception as exc:
            stage_tracker.fail_stage("video_input", f"Video CFR normalize failed: {exc}")
            job.status = "failed"
            job.error_message = f"Video CFR normalize failed: {exc}"
            job.completed_at = datetime.now(timezone.utc)
            job.report_json = _fit_report_json(stage_tracker.build_telemetry(
                job_id=job.id,
                status="failed",
                processed_frames=0,
                total_frames=None,
                error_message=job.error_message,
            ))
            db.commit()
            db.close()
            return

        source = VideoFileSource(str(path))
        if not source.open():
            stage_tracker.fail_stage("video_input", f"OpenCV could not open CFR stream {path}")
            job.status = "failed"
            job.error_message = f"Could not open normalized video file {path}"
            job.completed_at = datetime.now(timezone.utc)
            job.report_json = _fit_report_json(stage_tracker.build_telemetry(
                job_id=job.id,
                status="failed",
                processed_frames=0,
                total_frames=None,
                error_message=job.error_message,
            ))
            db.commit()
            db.close()
            return

        job.total_frames = source.total_frames
        job.fps = source.fps
        job.duration_sec = source.duration_seconds
        stage_tracker.complete_stage(
            "video_input",
            f"Decoded CFR stream: {source.total_frames} frames ({float(source.fps or 0):.1f} fps, {source.width}x{source.height})"
        )
        db.commit()

        # Load models — brand-new detector/tracker/pose/novelty per job (no
        # cross-video ByteTrack / color-tracker / FSM state reuse).
        detector = YoloDetector()
        detector.load()
        detector.reset_tracking()
        tracker = BytetrackTracker()
        tracker.load()

        # Class-agnostic NOVELTY (scene-change) detector — additive, separate
        # source. Reads NOVELTY_* from config/events.yaml; NOVELTY_ENABLED gates it.
        nov_cfg = NoveltyConfig.from_yaml()
        nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
        movenet = MovenetPose()
        movenet.load()

        from scripts.run_pipeline import build_shared_file_pipeline

        pipeline_cfg = build_shared_file_pipeline(
            # P0-B: Path A (upload) shares ONE file-mode loop with Path B
            # (scripts/run_pipeline.py --source file): CFR upstream +
            # configure_determinism + identical AdaptiveEventDetector defaults.
            # Learning is recorded per uploaded video at finalize() below;
            # deterministic runs use the P0-A pin, never live learning.json.
            analysis_fps=8.0,
            camera_id=str(cam.id),
            deterministic=deterministic,
            post_backend_url=None,
            learning_tag=job.original_filename,
        )
        pipe = InferencePipeline(pipeline_cfg)
        pipe.event_detector.reset()

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
        stage_tracker.start_stage("person_detection", "ByteTrack / YOLO person detector active")
        stage_tracker.start_stage("waste_detection", "YOLO / Novelty / Color handheld bag detector active")
        stage_tracker.start_stage("tracking", "ByteTrack spatial tracking active")
        stage_tracker.start_stage("person_identity", "PersonIdentityManager active")
        stage_tracker.start_stage("object_identity", "ObjectIdentityManager active")
        stage_tracker.start_stage("pose_estimation", "MoveNet 17-keypoint skeleton active")
        stage_tracker.start_stage("ownership_association", "Candidate pair association active")
        stage_tracker.start_stage("temporal_event_detection", "FSM event state machine active")

        peak_concurrent_people = 0
        # Cache of the most recent ANALYSIS tick's detections, reused on the
        # skipped (non-analysis) frames so the video overlay stays populated
        # without re-running the heavy AI path. See the loop below.
        _last_persons: list = []
        _last_objects: list = []
        _last_event = None
        # Unconditional pre-loop binding: `persons` / `objects` are read by the
        # cancellation and exception paths below. Previously they were only
        # bound inside the loop body, so an early cancel/error referenced a
        # name that existed on some paths only (the `sep_floor` NameError
        # class, IMG_5291 / job 43). Declaring them here makes their existence
        # independent of how far the loop got.
        persons: List[Any] = []
        objects: List[Any] = []
        last_failure_frame: Any = None
        last_pkt_timestamp: float = 0.0
        for pkt in source:
            # Forensics anchors: the exact frame + timestamp currently being
            # processed. On an unexpected exception the handler reads these to
            # capture a REAL snapshot (never a synthesized placeholder image).
            if pkt.frame is not None:
                last_failure_frame = pkt.frame
            last_pkt_timestamp = float(pkt.timestamp)
            if is_job_cancellation_requested(job.id):
                log.info("Job %s: Cancellation requested at frame %s. Stopping gracefully.", job.id, frame_idx)
                stage_tracker.cancel(reason="Analysis stopped by user cancellation request", frame_idx=frame_idx)
                source.release()
                if writer is not None:
                    writer.release()
                job.status = "cancelled"
                job.processed_frames = frame_idx
                job.completed_at = datetime.now(timezone.utc)
                unique_persons, _ = _extract_person_identity_info(pipe.event_detector, detected_persons_set)
                cancelled_report = stage_tracker.build_telemetry(
                    job_id=job.id,
                    status="cancelled",
                    processed_frames=frame_idx,
                    total_frames=source.total_frames,
                    fps=source.fps,
                    processing_fps=job.processing_fps,
                    duration_sec=source.duration_seconds,
                    active_persons_count=len(persons),
                    peak_concurrent_people=peak_concurrent_people,
                    unique_persons_count=unique_persons,
                    total_person_track_ids=len(detected_persons_set),
                    objects_count=len(detected_objects_set),
                    candidates_count=len(pipe.event_detector._pairs) + len(pipe.events) + len(pipe.event_detector.rejected_events),
                    confirmed_events_count=len(pipe.events),
                    rejected_events_count=len(pipe.event_detector.rejected_events),
                    current_fsm_state=current_state,
                    elapsed_sec=time.time() - t_start,
                )
                job.report_json = _fit_report_json(cancelled_report)
                db.commit()
                return

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
            stage_tracker.current_stage_id = "person_detection"
            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            if run_pose:
                stage_tracker.current_stage_id = "pose_estimation"
            else:
                stage_tracker.current_stage_id = "waste_detection"
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker, frame_idx,
                run_pose=run_pose, nov=nov,
            )

            # process_frame buffers EVERY frame (evidence clips need full-rate
            # input) and internally throttles the event detector to analysis_fps.
            stage_tracker.current_stage_id = "temporal_event_detection"
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
            active_count = len([p for p in persons if float(p.confidence) >= 0.3])
            peak_concurrent_people = max(peak_concurrent_people, active_count)

            if frame_idx % 15 == 0:
                elapsed = max(1e-6, time.time() - t_start)
                job.processed_frames = frame_idx
                job.processing_fps = round(frame_idx / elapsed, 1)
                unique_persons, person_identity_uids = _extract_person_identity_info(pipe.event_detector, detected_persons_set)
                total_tracks = len(detected_persons_set)
                object_uids_count = _extract_object_identity_count(pipe.event_detector, detected_objects_set)
                job.persons_detected = unique_persons
                job.objects_detected = len(detected_objects_set)
                job.events_count = len(pipe.events)

                stage_tracker.update_stage_detail("person_detection", f"{total_tracks} track ID(s) emitted | {active_count} active in frame", frame_idx=frame_idx)
                stage_tracker.update_stage_detail("waste_detection", f"{len(detected_objects_set)} track(s) | source={_dominant_detector_source(object_sources)}", frame_idx=frame_idx)
                stage_tracker.update_stage_detail("tracking", f"ByteTrack active ({tracker.store_size} stored tracks)", frame_idx=frame_idx)
                stage_tracker.update_stage_detail("person_identity", f"{unique_persons} unique person UID(s) consolidated", frame_idx=frame_idx)
                stage_tracker.update_stage_detail("object_identity", f"{object_uids_count} unique object UID(s) tracked", frame_idx=frame_idx)
                stage_tracker.update_stage_detail("pose_estimation", "MoveNet keypoints computed at analysis ticks", frame_idx=frame_idx)
                stage_tracker.update_stage_detail("ownership_association", f"{len(pipe.event_detector._pairs)} candidate pair(s) in evaluation", frame_idx=frame_idx)
                stage_tracker.update_stage_detail("temporal_event_detection", f"FSM: {current_state} | {len(pipe.events)} confirmed, {len(pipe.event_detector.rejected_events)} rejected", frame_idx=frame_idx)

                det_summary = pipe.event_detector.summary()
                live_telemetry = stage_tracker.build_telemetry(
                    job_id=job.id,
                    status="processing",
                    processed_frames=frame_idx,
                    total_frames=source.total_frames,
                    fps=source.fps,
                    processing_fps=job.processing_fps,
                    duration_sec=source.duration_seconds,
                    active_persons_count=active_count,
                    peak_concurrent_people=peak_concurrent_people,
                    unique_persons_count=unique_persons,
                    total_person_track_ids=total_tracks,
                    person_identity_uids=person_identity_uids,
                    objects_count=len(detected_objects_set),
                    candidates_count=det_summary.get("total_candidates", 0),
                    confirmed_events_count=len(pipe.events),
                    rejected_events_count=len(pipe.event_detector.rejected_events),
                    current_fsm_state=current_state,
                    rejection_reasons=det_summary.get("rejection_reason_counts", {}),
                    elapsed_sec=elapsed,
                )
                job.report_json = _fit_report_json(live_telemetry)
                db.commit()
            # Periodically release Python/torch/TF debris so a single long video
            # analysis does not accumulate memory until the encoder OOMs.
            if frame_idx % 30 == 0:
                gc.collect()

        source.release()
        if writer is not None:
            writer.release()

        unique_persons, final_person_uids = _extract_person_identity_info(pipe.event_detector, detected_persons_set)
        total_tracks = len(detected_persons_set)
        final_object_uids_count = _extract_object_identity_count(pipe.event_detector, detected_objects_set)
        det_summary_temp = pipe.event_detector.summary()

        stage_tracker.complete_stage("person_detection", f"Completed: {total_tracks} track IDs emitted, {unique_persons} unique people seen", frame_idx=frame_idx)
        stage_tracker.complete_stage("waste_detection", f"Completed: {len(detected_objects_set)} object tracks (source: {_dominant_detector_source(object_sources)})", frame_idx=frame_idx)
        stage_tracker.complete_stage("tracking", "Completed: ByteTrack track identities maintained across full video", frame_idx=frame_idx)
        stage_tracker.complete_stage("person_identity", f"Completed: {unique_persons} unique person UID(s) consolidated by PersonIdentityManager", frame_idx=frame_idx)
        stage_tracker.complete_stage("object_identity", f"Completed: {final_object_uids_count} unique object UID(s) tracked", frame_idx=frame_idx)
        stage_tracker.complete_stage("pose_estimation", "Completed: MoveNet skeletal keypoints evaluated across all analysis ticks", frame_idx=frame_idx)
        stage_tracker.complete_stage("ownership_association", f"Completed: {det_summary_temp.get('total_candidates', 0)} candidate pairs evaluated", frame_idx=frame_idx)
        stage_tracker.complete_stage("temporal_event_detection", f"Completed: FSM evaluated ({len(pipe.events)} confirmed, {len(pipe.event_detector.rejected_events)} rejected)", frame_idx=frame_idx)

        # Stage 10: EVIDENCE
        stage_tracker.start_stage("evidence_assembly", "Generating forensic evidence packages and H.264 video transcode", frame_idx=frame_idx)
        job.report_json = _fit_report_json(stage_tracker.build_telemetry(
            job_id=job.id,
            status="processing",
            processed_frames=frame_idx,
            total_frames=source.total_frames,
            fps=source.fps,
            processing_fps=job.processing_fps,
            duration_sec=source.duration_seconds,
            active_persons_count=0,
            peak_concurrent_people=peak_concurrent_people,
            unique_persons_count=unique_persons,
            total_person_track_ids=total_tracks,
            objects_count=len(detected_objects_set),
            candidates_count=det_summary_temp.get("total_candidates", 0),
            confirmed_events_count=len(pipe.events),
            rejected_events_count=len(pipe.event_detector.rejected_events),
            current_fsm_state=current_state,
            elapsed_sec=time.time() - t_start,
        ))
        db.commit()

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

        # Authoritative Backend Event Deduplication (spec #5: ONE EVENT = ONE RESULT)
        # Deduplicate candidate events by (actor_uid, object_uid) within a 20s window.
        seen_event_clusters: List[Dict[str, Any]] = []
        for ev in pipe.events:
            det_ev = confirmed_event_dicts.get(ev.event_id) or {}
            actor_uid = det_ev.get("event_actor_person_uid") or ev.person_track_id
            object_uid = det_ev.get("event_object_uid") or ev.object_track_id
            # PipelineEvent uses event_timestamp (float seconds), not .timestamp.
            # Prefer detector relative/confirmed timestamps when available for
            # stable temporal clustering across the same physical incident.
            det_ts = det_ev.get("relative_timestamp")
            if det_ts is None:
                raw_confirmed = (det_ev.get("timestamps") or {}).get("confirmed")
                det_ts = raw_confirmed
            if det_ts is not None:
                ev_ts = float(det_ts)
            else:
                ev_ts = float(getattr(ev, "event_timestamp", 0.0) or 0.0)
            conf = float(ev.confidence)

            matched_cluster = None
            for cluster in seen_event_clusters:
                time_diff = abs(cluster["timestamp"] - ev_ts)
                same_actor = (actor_uid is not None and actor_uid == cluster["actor_uid"])
                same_object = (object_uid is not None and object_uid == cluster["object_uid"])
                if (same_actor and same_object) or (same_object and time_diff < 30.0) or (same_actor and time_diff < 15.0):
                    matched_cluster = cluster
                    break

            if matched_cluster is None:
                new_cluster = {
                    "actor_uid": actor_uid,
                    "object_uid": object_uid,
                    "timestamp": ev_ts,
                    "best_event": ev,
                    "best_conf": conf,
                }
                seen_event_clusters.append(new_cluster)
            else:
                if conf > matched_cluster["best_conf"]:
                    matched_cluster["best_event"] = ev
                    matched_cluster["best_conf"] = conf

        deduped_pipe_events = [cluster["best_event"] for cluster in seen_event_clusters]

        created_event_ids = []
        evidence_packages: Dict[str, Dict[str, Optional[str]]] = {}
        for ev in deduped_pipe_events:
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
            det_ev = confirmed_event_dicts.get(ev.event_id) if ev.event_id in confirmed_event_dicts else None
            actor_uid = det_ev.get("event_actor_person_uid") if det_ev else None
            actor_track = det_ev.get("event_actor_person_track_id") if det_ev else None
            object_uid = det_ev.get("event_object_uid") if det_ev else None
            object_track = det_ev.get("event_object_track_id") if det_ev else None
            if actor_uid is None or object_uid is None:
                log.warning(
                    "Job %s event %s: confirmed event missing stable identity fields "
                    "(event_actor_person_uid=%r, event_object_uid=%r, det_ev_present=%s); "
                    "persisting with legacy track-id attribution only",
                    job.id, ev.event_id, actor_uid, object_uid, det_ev is not None,
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

            # 3) generate the professional evidence package from the real ORIGINAL video
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

        stage_tracker.complete_stage(
            "evidence_assembly",
            f"Completed: {len(evidence_packages)} evidence package(s) assembled, H.264 video ready",
            frame_idx=frame_idx
        )

        # Stage 11: DATABASE
        stage_tracker.start_stage("database_persistence", "Persisting events, evidence records, and metadata to PostgreSQL", frame_idx=frame_idx)
        stage_tracker.complete_stage(
            "database_persistence",
            f"Completed: {len(created_event_ids)} event(s) and evidence records committed",
            frame_idx=frame_idx
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
        job.persons_detected = unique_persons
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

        # Stage 12: API / DASHBOARD
        stage_tracker.start_stage("api_dashboard", "Writing analysis manifest and finalizing review contract", frame_idx=frame_idx)

        # Build diagnostic report
        detector_summary = pipe.event_detector.summary()
        no_candidate_reason = None
        if not pipe.events:
            no_candidate_reason = _no_candidate_reason(detector_summary, unique_persons, len(detected_objects_set))
        markers = _build_markers(first_timestamp, person_first_seen, object_first_seen, confirmed_event_dicts)
        trajectories = _build_trajectories(confirmed_event_dicts, frame_records, first_timestamp)

        stage_tracker.complete_stage("api_dashboard", "Completed: Analysis complete and ready for interactive forensic review", frame_idx=frame_idx)

        report = {
            "source": job.original_filename,
            "analysis_source": "source_CFR.mp4",
            "determinism": {
                "enabled": deterministic,
                "report": determinism_report,
                "learning_writes_frozen": bool(
                    getattr(pipe.event_detector, "freeze_learning_writes", False)
                ),
            },
            "duration_sec": round(source.duration_seconds, 2),
            "total_frames": source.total_frames,
            "processed_frames": frame_idx,
            "processing_fps": job.processing_fps,
            "performance": performance,
            "persons_count": unique_persons,
            "unique_persons_count": unique_persons,
            "total_person_track_ids": total_tracks,
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
            "stages": stage_tracker.to_list(),
            "current_stage": "Stage 12/12 — API / DASHBOARD",
            "current_stage_status": "COMPLETED",
            "last_successful_stage": "Stage 12/12 — API / DASHBOARD",
            "overall_progress_percent": 100,
            "last_processed_frame": frame_idx,
            "last_update_time": datetime.now(timezone.utc).isoformat(),
            "person_metrics": {
                "active_people": 0,
                "peak_concurrent_people": peak_concurrent_people,
                "unique_people_seen": unique_persons,
                "total_track_ids": total_tracks,
                "identity_mapping": final_person_uids,
                "explanation": "Active people = currently in frame; Unique people seen = stable IDs from PersonIdentityManager; Total track IDs = raw ByteTrack tracker churn."
            },
            "active_persons_count": 0,
            "candidates_count": detector_summary.get("total_candidates", 0),
            "rejected_count": detector_summary.get("rejected_candidates", 0),
            "confirmed_events_count": len(pipe.events),
            "current_fsm_state": current_state,
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
                persons_count=unique_persons,
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
        if 'stage_tracker' in locals():
            stage_tracker.fail_stage(
                stage_tracker.current_stage_id,
                str(e),
                frame_idx=frame_idx if 'frame_idx' in locals() else None,
            )
            try:
                # Failure forensics: JSON + annotated failure frame (real frame
                # only; labelled nearest_frame when it could not be written).
                _write_failure_forensics(
                    job_id=job.id if job else job_id,
                    video_name=getattr(job, "original_filename", None) if job else None,
                    stage_id=stage_tracker.current_stage_id,
                    exc=e,
                    traceback_text=traceback.format_exc(),
                    frame_idx=frame_idx if 'frame_idx' in locals() else None,
                    timestamp=locals().get("last_pkt_timestamp"),
                    source_fps=getattr(locals().get("source"), "fps", None),
                    frame=locals().get("last_failure_frame"),
                    persons=locals().get("persons"),
                    objects=locals().get("objects"),
                    detector=getattr(pipe, "event_detector", None) if 'pipe' in locals() else None,
                    pair_states=_pair_forensics(
                        getattr(pipe, "event_detector", None) if 'pipe' in locals() else None
                    ),
                )
            except Exception:
                log.exception("Job %s: failure forensics capture failed", job_id)
            failed_report = stage_tracker.build_telemetry(
                job_id=job.id if job else job_id,
                status="failed",
                processed_frames=frame_idx if 'frame_idx' in locals() else 0,
                total_frames=source.total_frames if 'source' in locals() and hasattr(source, 'total_frames') else None,
                error_message=str(e),
            )
            if job:
                job.report_json = _fit_report_json(failed_report)
        if job:
            job.status = "failed"
            job.error_message = f"Failed in {stage_tracker.current_stage_id if 'stage_tracker' in locals() else 'pipeline'}: {e}"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        with _CANCELLATION_LOCK:
            _JOB_CANCELLATION_TOKENS.pop(job_id, None)
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


# --------------------------------------------------------------------------- #
# Job Lifecycle Management: Stop & Delete Controls (P0 Requirements)
# --------------------------------------------------------------------------- #

@router.post("/jobs/{job_id}/stop", response_model=schemas.VideoAnalysisJobOut)
@router.post("/jobs/{job_id}/cancel", response_model=schemas.VideoAnalysisJobOut)
def stop_analysis_job(job_id: int, db: Session = Depends(get_db)):
    """Request safe, immediate cancellation of an active or queued analysis job."""
    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    if job.status in ("completed", "failed", "cancelled"):
        return job

    if job.status == "queued":
        job.status = "cancelled"
        job.completed_at = datetime.now(timezone.utc)
        tracker = PipelineStageTracker()
        tracker.cancel(reason="Job cancelled by operator before processing began")
        cancelled_report = tracker.build_telemetry(
            job_id=job.id,
            status="cancelled",
            processed_frames=0,
            total_frames=job.total_frames,
            error_message="Job cancelled while in queue",
        )
        job.report_json = _fit_report_json(cancelled_report)
        db.commit()
        db.refresh(job)
        return job

    if job.status == "processing":
        # Signal cancellation to the running thread
        requested = request_job_cancellation(job_id)
        log.info("Requested cancellation for active job %s (token_found=%s)", job_id, requested)
        # If thread wasn't running (e.g. backend was restarted), transition directly
        if not requested:
            job.status = "cancelled"
            job.completed_at = datetime.now(timezone.utc)
            tracker = PipelineStageTracker()
            tracker.cancel(reason="Job cancelled by operator")
            cancelled_report = tracker.build_telemetry(
                job_id=job.id,
                status="cancelled",
                processed_frames=job.processed_frames,
                total_frames=job.total_frames,
            )
            job.report_json = _fit_report_json(cancelled_report)
            db.commit()
            db.refresh(job)
        return job

    return job


@router.delete("/jobs/{job_id}")
def delete_analysis_job(
    job_id: int,
    confirm: bool = Query(False, description="Explicit confirmation required (?confirm=true)"),
    db: Session = Depends(get_db)
):
    """Delete an analysis job and its associated artifacts from database and storage."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deletion requires explicit confirmation. Pass ?confirm=true to proceed."
        )

    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    if job.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete an actively running analysis job. Stop the analysis first via POST /api/analysis/jobs/{job_id}/stop."
        )

    # 1. Delete associated events (cascades to evidence)
    events = db.query(models.Event).filter(models.Event.analysis_job_id == job_id).all()
    for ev in events:
        db.delete(ev)

    # 2. Delete artifact directory on disk if it exists
    from backend.routers.evidence import EVIDENCE_STORE
    job_artifact_dir = EVIDENCE_STORE / "analysis" / str(job_id)
    if job_artifact_dir.exists():
        try:
            shutil.rmtree(job_artifact_dir, ignore_errors=True)
        except Exception as e:
            log.warning("Could not remove artifact dir %s: %s", job_artifact_dir, e)

    # 3. Delete job record
    db.delete(job)
    db.commit()

    return {
        "deleted": True,
        "job_id": job_id,
        "message": f"Analysis Job #{job_id} and associated evidence records deleted successfully."
    }


@router.delete("/jobs/{job_id}/video")
def delete_analysis_source_video(
    job_id: int,
    confirm: bool = Query(False, description="Explicit confirmation required (?confirm=true)"),
    db: Session = Depends(get_db)
):
    """Delete the uploaded source video file to free disk space, leaving job metadata intact."""
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Deletion requires explicit confirmation. Pass ?confirm=true to proceed."
        )

    job = db.query(models.VideoAnalysisJob).filter(models.VideoAnalysisJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")

    if job.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete source video while analysis is actively processing."
        )

    raw_path = job.original_video_path or job.file_path
    path = _resolve_archive_path(raw_path)
    deleted = False
    if path and path.exists() and path.is_file():
        try:
            path.unlink()
            deleted = True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete video file {path}: {e}")

    job.original_video_path = None
    db.commit()

    return {
        "deleted_video": deleted,
        "job_id": job_id,
        "message": "Uploaded source video file deleted successfully to reclaim disk space."
    }

