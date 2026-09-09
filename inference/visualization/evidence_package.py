"""Evidence package writer for real confirmed littering events.

The files written here are extracted from the ORIGINAL source video and the
real FrameAnalysis records produced by the production pipeline. No frame
is synthesized and no bbox is invented.

PHASE 2 REWORK (dashboard evidence-display fixes):
  * snapshot.jpg / person.jpg / waste.jpg are now cut from the ORIGINAL
    video (no AI overlay burned in) — the previous version cut them from
    the ANALYZED video, so the snapshot stacked dozens of tracking boxes
    from the whole video and the crops were unreadable.
  * snapshot.jpg shows ONLY the confirmed event's person box (green) and
    object box (red) drawn cleanly on the single best explanatory frame
    (ground/release), with one small caption.
  * person.jpg / waste.jpg are clean crops with ~15% margin, upscaled if
    needed so the smallest dimension is >= 300 px, single label drawn once.
  * event_clip.mp4 window: ~5 s BEFORE the object first appears in the
    person's hands (carry_start) to ~5 s AFTER the person departs
    (departure), re-encoded to browser-playable H.264 (was mp4v which no
    browser can decode — the "empty evidence video" root cause).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _closest_record(records: List[Dict[str, Any]], frame_number: Optional[int], timestamp: Optional[float]) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    if frame_number is not None:
        return min(records, key=lambda r: abs(int(r.get("frame_number", 0)) - int(frame_number)))
    if timestamp is not None:
        return min(records, key=lambda r: abs(float(r.get("timestamp", 0.0)) - float(timestamp)))
    return records[-1]


def _find_entity(record: Dict[str, Any], kind: str, track_id: Any) -> Optional[Dict[str, Any]]:
    items = record.get(kind, []) or []
    tid = str(track_id)
    for item in items:
        if str(item.get("track_id")) == tid:
            return item
    return None


def _find_object(record: Dict[str, Any], track_id: Any, object_uid: Any) -> Optional[Dict[str, Any]]:
    """Locate the event object by STABLE object_uid first, then churned track_id.

    Phase B: a physical bag is re-born under many track ids, so anchoring
    evidence to the churned ``bag_track_id`` can miss the object at the carry
    frame. The stable ``object_uid`` lets us find the SAME physical object
    across the whole carry->release->ground arc.
    """
    items = record.get("objects", []) or []
    if object_uid is not None:
        for item in items:
            if item.get("object_uid") is not None and str(item.get("object_uid")) == str(object_uid):
                return item
    if track_id is not None:
        for item in items:
            if str(item.get("track_id")) == str(track_id):
                return item
    return None


# --------------------------------------------------------------------------- #
# Clean-crop helpers (Phase 2)
# --------------------------------------------------------------------------- #
_CROP_PAD_RATIO = 0.15   # 15% margin around the bbox (spec: 10–20%)
_CROP_MIN_DIM = 300      # upscale so the smallest dimension >= 300 px
_LABEL_FONT_SCALE = 0.6
_LABEL_THICKNESS = 1


def _crop_with_margin(cv2, frame, bbox: Tuple[float, float, float, float]):
    """Crop bbox + 15% margin, clipped to frame. Returns the crop (or None)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    px, py = bw * _CROP_PAD_RATIO, bh * _CROP_PAD_RATIO
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    if cx2 - cx1 < 2 or cy2 - cy1 < 2:
        return None
    return frame[cy1:cy2, cx1:cx2]


def _draw_label_once(cv2, crop: Any, text: str, color: Tuple[int, int, int]) -> Any:
    """Draw ONE small label at the crop's top-left. Never repeated."""
    if crop is None or crop.size == 0:
        return crop
    ch = crop.shape[0]
    cw = crop.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = _LABEL_FONT_SCALE
    thickness = _LABEL_THICKNESS
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Fit text within crop width; shrink font if a long label overflows.
    while tw + 12 > cw and scale > 0.25:
        scale -= 0.05
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    # Semi-transparent banner for readability on any background.
    banner_h = min(ch - 1, th + baseline + 10)
    overlay = crop.copy()
    cv2.rectangle(overlay, (0, 0), (min(cw - 1, tw + 12), banner_h), (0, 0, 0), -1)
    alpha = 0.55
    cv2.addWeighted(overlay[:banner_h, : min(cw, tw + 12)], alpha, crop[:banner_h, : min(cw, tw + 12)], 1.0 - alpha, 0)
    cv2.putText(crop, text, (6, th + 5), font, scale, color, thickness, cv2.LINE_AA)
    return crop


def _crop_to_file(cv2, frame, bbox: Tuple[float, float, float, float], path: str,
                  label: str = "", pad_ratio: float = _CROP_PAD_RATIO,
                  min_dim: int = _CROP_MIN_DIM) -> bool:
    """Clean evidence crop: margin, >=300px smallest dim, single label."""
    crop = _crop_with_margin(cv2, frame, bbox, )
    if crop is None:
        return False
    ch, cw = crop.shape[:2]
    # Upscale if too small (but never downscale) so the smallest side >= 300px.
    if min(ch, cw) < min_dim:
        s = min_dim / max(1, min(ch, cw))
        crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_CUBIC)
    if label:
        _draw_label_once(cv2, crop, label, (255, 255, 255))
    return bool(cv2.imwrite(path, crop))


def _extract_frame(video_path: str, frame_number: int):
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_number)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


class _StillFrameCache:
    """Decode every needed still frame exactly once from ONE VideoCapture open.

    P1-9: the previous code called ``_extract_frame`` once per artifact
    (snapshot + carry/release/ground), re-opening and re-seeking the source
    video up to 4x per confirmed event. All needed frame numbers are now
    collected first and read with a single open (sorted seeks).
    """

    def __init__(self, video_path: Optional[str]):
        self._video_path = video_path
        self._requested: set = set()
        self._frames: Dict[int, Any] = {}

    def request(self, frame_number: Optional[int]) -> None:
        if self._video_path is None or frame_number is None:
            return
        fn = int(frame_number)
        if fn >= 0 and fn not in self._frames:
            self._requested.add(fn)

    def load(self) -> None:
        if not self._requested or not self._video_path:
            return
        import cv2  # type: ignore

        cap = cv2.VideoCapture(self._video_path)
        try:
            if not cap.isOpened():
                logger.warning("still-frame cache: could not open %s", self._video_path)
                return
            for fn in sorted(self._requested):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self._frames[fn] = frame
        finally:
            cap.release()
        self._requested.clear()

    def get(self, frame_number: Optional[int]):
        if frame_number is None:
            return None
        return self._frames.get(int(frame_number))


def _write_clip(cv2, source_video: str, target_video: str, start_ts: float, end_ts: float, fps: float) -> bool:
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        return False
    writer = cv2.VideoWriter(target_video, fourcc, max(1.0, float(fps)), (w, h))
    if not writer.isOpened():
        cap.release()
        return False
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_ts * 1000.0))
    wrote = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if current > end_ts:
            break
        writer.write(frame)
        wrote = True
    writer.release()
    cap.release()
    return wrote and os.path.exists(target_video) and os.path.getsize(target_video) > 0


def _capture_face_evidence(cv2, original_video_path, frame_records, event, target_dir, source_fps, actor_pid=None):
    """Capture the clearest face of the involved person as ``face_evidence.jpg``.

    ADDITIVE and FAIL-SAFE: any failure returns a ``FACE_NOT_CAPTURED`` dict
    instead of raising — the rest of the evidence package must never break
    because of face capture. We also NEVER fabricate an image: if no clear
    face is found the path is ``None``.

    GATED TO CONFIRMED EVENTS: this must only ever be reached for a genuinely
    CONFIRMED littering event. As a defensive guarantee, if the supplied
    ``event`` is not confirmed we immediately return ``FACE_NOT_CAPTURED``.

    PHASE 2: backend order changed to "region" (deterministic head-region
    crop from the tracked person box) because the production container does
    NOT ship insightface/onnxruntime — the previous "retinaface" default
    silently degraded to FACE_NOT_CAPTURED for every Phase-1 event. If a
    pixel-accurate detector is later installed, pass backend="retinaface".
    The head/face area still comes only from REAL tracked person boxes;
    nothing is invented. Sensitive data: gated behind human review.
    """
    if not event.get("confirmed"):
        return {
            "captured": False, "status": "FACE_NOT_CAPTURED",
            "face_evidence_path": None, "face_detection_confidence": None,
            "frame_number": None, "face_bbox": None, "face_size_px": None,
            "blur_score": None, "reason": "event not confirmed — face capture skipped",
            "sensitive": True,
        }
    try:
        from inference.evidence.face_evidence import FaceEvidenceCapture
        cap = FaceEvidenceCapture(backend="region")
    except Exception as e:  # detector unavailable — degrade gracefully
        return {
            "captured": False, "status": "FACE_NOT_CAPTURED",
            "face_evidence_path": None, "face_detection_confidence": None,
            "frame_number": None, "face_bbox": None, "face_size_px": None,
            "blur_score": None, "reason": f"face detector unavailable: {e}",
            "sensitive": True,
        }
    frs = event.get("frames") or {}
    fns = [int(v) for v in frs.values() if isinstance(v, (int, float))]
    window = (min(fns), max(fns)) if fns else None
    pid = actor_pid if actor_pid is not None else event.get("person_track_id")
    try:
        return cap.capture_best(
            original_video_path, frame_records, pid,
            frame_window=window, out_dir=target_dir, out_name="face_evidence.jpg",
        )
    except Exception as e:
        return {
            "captured": False, "status": "FACE_NOT_CAPTURED",
            "face_evidence_path": None, "face_detection_confidence": None,
            "frame_number": None, "face_bbox": None, "face_size_px": None,
            "blur_score": None, "reason": f"face capture error: {e}",
            "sensitive": True,
        }


def write_event_evidence_package(
    *,
    original_video_path: Optional[str] = None,
    target_dir: str,
    event: Dict[str, Any],
    frame_records: List[Dict[str, Any]],
    job_id: int,
    original_filename: str,
    source_fps: float,
    pre_seconds: float = 5.0,
    post_seconds: float = 5.0,
    analyzed_video_path: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Write snapshot/person/waste/face/event_clip/metadata from real artifacts.

    ``original_video_path`` is the source video the pipeline actually
    analyzed (clean frames, no overlay) — snapshot/crops/clip are cut from
    it. ``analyzed_video_path`` is accepted for backward compatibility with
    older callers (tests/scripts): if ``original_video_path`` is not given,
    the analyzed video is used as the extraction source (previous behavior).
    """
    import cv2  # type: ignore
    from inference.visualization.h264 import extract_clip_h264, transcode_to_h264

    if not original_video_path:
        original_video_path = analyzed_video_path
    os.makedirs(target_dir, exist_ok=True)
    frames = event.get("frames") or {}
    timestamps = event.get("timestamps") or {}

    # Phase C — resolve the AUTHORITATIVE actor/object identity. Prefer the
    # frozen event-actor record; fall back to the legacy fields so historical
    # events still render. Evidence is anchored to these exact IDs (spec #11/#17):
    # a passing person/car can never be substituted in.
    actor_pid = event.get("event_actor_person_track_id")
    if actor_pid is None:
        actor_pid = event.get("person_track_id")
    obj_tid = event.get("event_object_track_id")
    if obj_tid is None:
        obj_tid = event.get("bag_track_id")
    obj_uid = event.get("event_object_uid")
    if obj_uid is None:
        obj_uid = event.get("bag_uid")

    # Prefer the ground frame for the best explanatory still: it usually shows
    # person + waste + ground context together. Fall back to release/confirmed.
    preferred = [
        frames.get("ground"),
        frames.get("release"),
        frames.get("confirmed"),
        frames.get("departure"),
        frames.get("carry_start"),
    ]
    best_record = None
    for fr in preferred:
        rec = _closest_record(frame_records, fr, None)
        if rec is None:
            continue
        person = _find_entity(rec, "persons", actor_pid)
        obj = _find_object(rec, obj_tid, obj_uid)
        if person and obj:
            best_record = rec
            break
    if best_record is None:
        best_record = _closest_record(frame_records, frames.get("confirmed") or frames.get("ground"), timestamps.get("confirmed"))

    snapshot_path = os.path.join(target_dir, "snapshot.jpg")
    person_path = os.path.join(target_dir, "person.jpg")
    waste_path = os.path.join(target_dir, "waste.jpg")
    carry_path = os.path.join(target_dir, "carry.jpg")
    release_path = os.path.join(target_dir, "release.jpg")
    ground_path = os.path.join(target_dir, "ground.jpg")
    clip_path = os.path.join(target_dir, "event_clip.mp4")
    metadata_path = os.path.join(target_dir, "metadata.json")

    snapshot_written = False
    frame_number_used = None

    # P1-9: resolve EVERY still frame needed (snapshot + carry/release/ground)
    # BEFORE decoding anything, so all stills come from ONE VideoCapture open
    # (previously: a fresh open+seek per artifact — up to 4 opens per event).
    # P1-2: every failure to produce a sequence image is LOGGED (was a silent
    # None return, so operators could never see how often these fail).
    cache = _StillFrameCache(original_video_path)

    def _resolve_sequence(key: str, label_prefix: str):
        fr = frames.get(key)
        if fr is None:
            logger.info("sequence image %s: no '%s' frame recorded for this event", label_prefix, key)
            return None
        rec = _closest_record(frame_records, fr, None)
        if rec is None:
            logger.info("sequence image %s: no frame record near frame %s", label_prefix, fr)
            return None
        person_e = _find_entity(rec, "persons", actor_pid)
        obj_e = _find_object(rec, obj_tid, obj_uid)
        # Both actor and object must be visible for the sequence to be valid
        if not (person_e and obj_e and person_e.get("bbox") and obj_e.get("bbox")):
            logger.info(
                "sequence image %s: actor(object track %s) or object(object uid %s) "
                "bbox missing in frame record %s", label_prefix, actor_pid, obj_uid,
                rec.get("frame_number"))
            return None
        cache.request(rec.get("frame_number", fr))
        return rec, person_e, obj_e

    seq_resolved = {
        "carry_start": _resolve_sequence("carry_start", "CARRY"),
        "release": _resolve_sequence("release", "RELEASE"),
        "ground": _resolve_sequence("ground", "GROUND"),
    }
    if best_record is not None:
        frame_number = int(best_record.get("frame_number", frames.get("confirmed") or 0))
        frame_number_used = frame_number
        cache.request(frame_number)
    cache.load()

    def _write_sequence_image(key: str, out_path: str, label_prefix: str) -> Optional[str]:
        resolved = seq_resolved.get(key)
        if resolved is None:
            return None  # already logged during resolution
        rec, person_e, obj_e = resolved
        frm = cache.get(int(rec.get("frame_number", frames.get(key) or 0)))
        if frm is None:
            logger.warning("sequence image %s: could not extract frame %s from source video", label_prefix, rec.get("frame_number"))
            return None
        snap_seq = frm.copy()
        x1, y1, x2, y2 = [int(v) for v in person_e["bbox"]]
        cv2.rectangle(snap_seq, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(snap_seq, "PERSON", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2, cv2.LINE_AA)
        x1o, y1o, x2o, y2o = [int(v) for v in obj_e["bbox"]]
        cv2.rectangle(snap_seq, (x1o, y1o), (x2o, y2o), (30, 30, 240), 2)
        label = str(obj_e.get("class_name") or "WASTE").upper()[:20]
        cv2.putText(snap_seq, label, (x1o, max(20, y1o - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 240), 2, cv2.LINE_AA)
        cv2.putText(snap_seq, label_prefix, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if cv2.imwrite(out_path, snap_seq):
            return out_path
        logger.warning("sequence image %s: cv2.imwrite failed for %s", label_prefix, out_path)
        return None

    if best_record is not None:
        frame = cache.get(frame_number_used)
        if frame is not None:
            person = _find_entity(best_record, "persons", actor_pid)
            obj = _find_object(best_record, obj_tid, obj_uid)
            # --- snapshot.jpg: ONE clean frame, ONLY the event's two boxes ---
            snap = frame.copy()
            if person and person.get("bbox"):
                x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
                cv2.rectangle(snap, (x1, y1), (x2, y2), (40, 220, 40), 2)   # green = person
                cv2.putText(snap, "PERSON", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 40), 2, cv2.LINE_AA)
            if obj and obj.get("bbox"):
                x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
                cv2.rectangle(snap, (x1, y1), (x2, y2), (30, 30, 240), 2)   # red = object
                label = str(obj.get("class_name") or "WASTE").upper()[:20]
                cv2.putText(snap, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 240), 2, cv2.LINE_AA)
            snapshot_written = bool(cv2.imwrite(snapshot_path, snap))
            # --- clean crops from the SAME original frame ---
            if person and person.get("bbox"):
                _crop_to_file(cv2, frame, tuple(person["bbox"]), person_path, label="PERSON")
            if obj and obj.get("bbox"):
                _crop_to_file(cv2, frame, tuple(obj["bbox"]), waste_path, label=obj.get("class_name", "WASTE").upper()[:20])
    # Write the three phase frames — all anchored to SAME actor+object UIDs
    carry_written = _write_sequence_image("carry_start", carry_path, "CARRY")
    release_written = _write_sequence_image("release", release_path, "RELEASE")
    ground_written = _write_sequence_image("ground", ground_path, "GROUND")

    # ---- event_clip.mp4: ~5s before first appearance (carry_start) to
    # ---- ~5s after departure, from the ORIGINAL video, H.264 encoded. ----
    first_ts = float(frame_records[0].get("timestamp", 0.0)) if frame_records else 0.0
    carry_ts = float(timestamps.get("carry_start") or (first_ts + (frames.get("carry_start") or 0) / max(1.0, source_fps)))
    departure_ts = float(timestamps.get("departure") or timestamps.get("confirmed") or carry_ts)
    rel_start = max(0.0, (carry_ts - first_ts) - pre_seconds)
    rel_end = (departure_ts - first_ts) + post_seconds
    # P1-9: prefer a SINGLE ffmpeg pass (decode + trim + H.264 encode in one
    # process). The old path wrote an mp4v intermediate with OpenCV and then
    # had ffmpeg re-decode and re-encode the same footage — two decode passes
    # and two encodes for overlapping content. The legacy path is kept as the
    # honest fallback when ffmpeg is unavailable or fails.
    clip_written = extract_clip_h264(original_video_path, clip_path, rel_start, rel_end)
    if not clip_written:
        clip_written = _write_clip(cv2, original_video_path, clip_path, rel_start, rel_end, source_fps)
        if clip_written:
            # mp4v -> H.264 so the dashboard <video> can actually decode it.
            transcode_to_h264(clip_path)
            clip_written = os.path.exists(clip_path) and os.path.getsize(clip_path) > 0

    # ---- face_evidence.jpg (separate card; fail-safe) ----
    face_ev = _capture_face_evidence(cv2, original_video_path, frame_records, event, target_dir, source_fps, actor_pid=actor_pid)
    face_path = face_ev.get("face_evidence_path")
    if not (face_path and os.path.exists(face_path) and os.path.getsize(face_path) > 0):
        face_path = None

    metadata = {
        "job_id": job_id,
        "original_filename": original_filename,
        "original_video": original_video_path,
        "analyzed_video": analyzed_video_path,
        "event": event,
        "best_frame_record": best_record,
        "snapshot_frame_number": frame_number_used,
        "files": {
            "snapshot": snapshot_path if snapshot_written else None,
            "person": person_path if os.path.exists(person_path) else None,
            "waste": waste_path if os.path.exists(waste_path) else None,
            "carry": carry_written,
            "release": release_written,
            "ground": ground_written,
            "event_clip": clip_path if clip_written else None,
            "face_evidence": face_path,
        },
        "clip_window": {
            "pre_seconds": pre_seconds,
            "post_seconds": post_seconds,
            "start_ts_rel": round(rel_start, 3),
            "end_ts_rel": round(rel_end, 3),
        },
        "face_evidence": {
            "face_evidence_path": face_path,
            "face_detection_confidence": face_ev.get("face_detection_confidence"),
            "frame_number": face_ev.get("frame_number"),
            "face_bbox": face_ev.get("face_bbox"),
            "face_size_px": face_ev.get("face_size_px"),
            "blur_score": face_ev.get("blur_score"),
            "status": face_ev.get("status", "FACE_NOT_CAPTURED"),
            "reason": face_ev.get("reason"),
            "sensitive": True,
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    return {
        "snapshot": snapshot_path if snapshot_written else None,
        "person": person_path if os.path.exists(person_path) and os.path.getsize(person_path) > 0 else None,
        "waste": waste_path if os.path.exists(waste_path) and os.path.getsize(waste_path) > 0 else None,
        "carry": carry_written,
        "release": release_written,
        "ground": ground_written,
        "clip": clip_path if clip_written else None,
        "face": face_path,
        "metadata": metadata_path if os.path.exists(metadata_path) else None,
    }
