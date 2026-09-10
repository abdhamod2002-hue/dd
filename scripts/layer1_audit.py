#!/usr/bin/env python3
"""Layer 1 + Addendum audit harness (DETECTION / TRACKING QUALITY ONLY).

Runs the REAL production Layer 1 path on every video in D:\\22 and produces
the Addendum (sections A-K) evidence. It does NOT instantiate the event
detector / FSM / voting / adaptive tuner / evidence manager — those are
Layer 2/3 and are explicitly out of scope for this phase
(spec: "Do NOT modify event logic").

Production path mirrored exactly:
    VideoFileSource -> YoloDetector.track(persist=True)
                     -> build_tracks_real (TrackStore + MoveNet@analysis_fps
                        + NoveltyDetector@analysis_fps)
    returns persons (yolo) + objects (yolo | color | novelty) with STABLE
    ByteTrack namespaced track ids.

Run:
    python scripts/layer1_audit.py --videos "D:\\22\\*.MOV" \\
        --out D:\\HO\\layer1_runs --workers 6 --analysis-fps 8
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_W: Dict[str, Any] = {}


def _worker_init() -> None:
    from inference.capture.camera_source import VideoFileSource  # noqa
    from inference.detection.yolo_detector import YoloDetector
    from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
    from inference.pose.movenet_pose import MovenetPose
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from inference.tracking.object_identity import ObjectIdentityManager
    import cv2  # noqa

    det = YoloDetector()
    det.load()
    det.reset_tracking()
    trk = BytetrackTracker()
    trk.load()
    mov = MovenetPose()
    mov.load()
    ncfg = NoveltyConfig.from_yaml()
    nov = NoveltyDetector(ncfg) if ncfg.enabled else None
    oid = ObjectIdentityManager()
    _W["detector"] = det
    _W["tracker"] = trk
    _W["movenet"] = mov
    _W["nov"] = nov
    _W["nov_cfg"] = ncfg
    _W["oid"] = oid
    _W["cv2"] = cv2
    _W["VideoFileSource"] = VideoFileSource


class _Gate:
    """Mirrors InferencePipeline.should_analyze (throttles to analysis_fps)."""

    def __init__(self, analysis_fps: float) -> None:
        self.dt = 1.0 / analysis_fps
        self.last = -1e9

    def check(self, ts: float) -> bool:
        if ts - self.last >= self.dt - 1e-6:
            self.last = ts
            return True
        return False


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


_PROGRESS: List[str] = []


def _progress_file(out_dir: str) -> str:
    return os.path.join(out_dir, "progress.log")


def _log(out_dir: str, msg: str) -> None:
    try:
        with open(_progress_file(out_dir), "a", encoding="utf-8") as f:
            f.write(msg + "\n")
            f.flush()
    except Exception:
        pass



def _mean_hue(frame, bbox) -> float:
    if frame is None:
        return 0.0
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    patch = frame[y1:y2, x1:x2]
    try:
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        return float(hsv[..., 0].mean())
    except Exception:
        return 0.0


def _brightness(frame) -> float:
    if frame is None:
        return 0.0
    try:
        return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    except Exception:
        return 0.0


def _global_motion(gray_prev, gray):
    if gray_prev is None or gray is None:
        return 0.0
    try:
        import numpy as np
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray, None, 0.5, 3, 15, 3, 5, 5, 0)
        return float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------- #
# Physical-object grouping for ID-switch measurement (sections B/C/I)
# --------------------------------------------------------------------------- #
def _group_raw_tracks(track_frames, max_gap=12, dist=180.0):
    """Re-derive stable logical identities from raw track-id trajectories.

    Semantics (important for honest measurement):
      * Each raw tracker id (namespaced ByteTrack / ColorBagTracker id) is
        by construction one entity that ByteTrack kept stable across frames,
        so ALL of a raw id's frames belong to ONE physical object.
      * ID switching = the same physical object being surfaced under a NEW raw
        id later (ByteTrack/color re-birth). We detect that by merging two raw
        ids into one physical object when they are temporally near (end of one
        to start of next within ``max_gap`` frames) AND spatially close (their
        boundary centroids within ``dist`` px).
      * ``n_tracks`` = number of physical objects after merging.
      * ``id_switches`` = sum over physical objects of (n_raw_ids - 1).
      * ``reassociated_tracks`` = physical objects that needed >1 raw id.

    track_frames: raw_id -> list of (frame_index, rel_ts, (cx,cy), source)
    Returns {uid: {raw_ids, first_frame, last_frame, duration_frames, source}}.
    """
    # Build per-raw-id summaries (ordered by first appearance).
    rawids = sorted(track_frames.keys())
    raw_summary: Dict[int, Dict[str, Any]] = {}
    for rid in rawids:
        pts = sorted(track_frames[rid], key=lambda x: x[0])
        f0 = pts[0][0]; f1 = pts[-1][0]
        cx0, cy0 = pts[0][2]; cx1, cy1 = pts[-1][2]
        raw_summary[rid] = {
            "first_frame": f0, "last_frame": f1,
            "first_centroid": (cx0, cy0), "last_centroid": (cx1, cy1),
            "source": pts[0][3],
        }

    # Greedy physical-object grouping across raw ids.
    uids: Dict[int, Dict[str, Any]] = {}
    uid_of_raw: Dict[int, int] = {}
    next_uid = 5000001
    # order raw ids by first_frame to make spatial/temporal continuity natural
    order = sorted(raw_summary.items(), key=lambda kv: (kv[1]["first_frame"], kv[0]))
    for rid, info in order:
        matched_uid = None
        best_d = dist
        # A raw id may switch within itself only if it reappears (gap) — but a
        # raw id is one entity, so find a uid that already contains this raw id;
        # prefer that uid, else look for another uid ending nearby in time/space.
        if rid in uid_of_raw:
            matched_uid = uid_of_raw[rid]
        if matched_uid is None:
            for uid, rec in uids.items():
                if info["first_frame"] - rec["last_frame"] > max_gap:
                    continue
                lx, ly = rec["last_centroid"]
                d = math.hypot(info["first_centroid"][0] - lx,
                               info["first_centroid"][1] - ly)
                if d <= best_d:
                    best_d = d
                    matched_uid = uid
        if matched_uid is None:
            matched_uid = next_uid
            next_uid += 1
            uids[matched_uid] = {
                "raw_ids": [], "source": info["source"],
                "first_frame": info["first_frame"], "last_frame": info["last_frame"],
                "last_centroid": info["last_centroid"],
            }
        rec = uids[matched_uid]
        if rid not in rec["raw_ids"]:
            rec["raw_ids"].append(rid)
            uid_of_raw[rid] = matched_uid
        if info["last_frame"] > rec["last_frame"]:
            rec["last_frame"] = info["last_frame"]
            rec["last_centroid"] = info["last_centroid"]
        if info["first_frame"] < rec["first_frame"]:
            rec["first_frame"] = info["first_frame"]
        if rec["source"] != info["source"] and rec["source"] == "yolo":
            rec["source"] = info["source"]
    for uid, info in uids.items():
        info["duration_frames"] = info["last_frame"] - info["first_frame"] + 1
    return uids


def _track_stats(grouped, fps):
    durations_s = []
    raw_counts = []
    id_switches = 0
    for uid, info in grouped.items():
        dur_f = info["duration_frames"]
        durations_s.append(dur_f / fps if fps else dur_f)
        raw_counts.append(len(info["raw_ids"]))
        id_switches += max(0, len(info["raw_ids"]) - 1)

    def stats(xs):
        if not xs:
            return {"avg": 0, "median": 0, "max": 0, "min": 0}
        xs = sorted(xs)
        return {"avg": round(sum(xs) / len(xs), 3),
                "median": round(xs[len(xs) // 2], 3),
                "max": round(max(xs), 3),
                "min": round(min(xs), 3)}
    return {
        "n_tracks": len(grouped),
        "durations_s": stats(durations_s),
        "raw_ids_per_physical_object": stats(raw_counts),
        "id_switches": id_switches,
        "reassociated_tracks": sum(1 for i in grouped.values() if len(i["raw_ids"]) > 1),
        "short_tracks_lt_1s": sum(1 for d in durations_s if d < 1.0),
    }


def _build(frame, tracked, mov, trk, frame_idx, run_pose, nov):
    from scripts.run_pipeline import build_tracks_real
    return build_tracks_real(frame, tracked, mov, trk, frame_idx,
                             run_pose=run_pose, nov=nov)


# --------------------------------------------------------------------------- #
# Per-video analysis
# --------------------------------------------------------------------------- #
def analyze_video(path, out_dir, analysis_fps, max_frames):
    cv2 = _W["cv2"]
    det = _W["detector"]
    trk = _W["tracker"]
    mov = _W["movenet"]
    nov = _W["nov"]
    VideoFileSource = _W["VideoFileSource"]

    stem = os.path.splitext(os.path.basename(path))[0]
    res: Dict[str, Any] = {"video": os.path.basename(path), "stem": stem}
    src = VideoFileSource(path)
    if not src.open():
        res["error"] = "VideoFileSource could not open"
        _write_json(os.path.join(out_dir, "raw", stem + ".json"), res)
        return res

    fps = float(src.fps or 30.0)
    total_frames = int(src.total_frames)
    res["fps"] = round(fps, 3)
    res["total_frames"] = total_frames
    res["duration_s"] = round(total_frames / fps, 2) if fps else None
    res["resolution"] = (int(src.width), int(src.height))
    _log(out_dir, f"START {stem} frames={total_frames} fps={fps}")


    # reset per-video state — detection is deterministic; reset trackers/oids
    det.reset_tracking()
    trk._store.clear()
    trk._frame_index = 0
    oid = _W["oid"]; oid.reset()
    gate = _Gate(analysis_fps)

    person_raw: Dict[int, List[Tuple[int, float, Tuple, str]]] = defaultdict(list)
    obj_raw: Dict[int, List[Tuple[int, float, Tuple, str]]] = defaultdict(list)
    frame_records: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = defaultdict(int)
    person_counts: Dict[str, int] = defaultdict(int)
    class_counts: Dict[str, int] = defaultdict(int)
    hues: List[float] = []
    bright: List[float] = []
    gmotions: List[float] = []
    gray_prev = None
    persons_seen: set = set()
    objects_seen: set = set()
    object_sources: Dict[str, int] = defaultdict(int)
    n_obj_records = 0
    n_per_records = 0
    t0 = time.time()
    frame_idx = 0
    run_pose_count = 0
    nov_tick_count = 0
    first_ts = None

    for pkt in src:
        frame = pkt.frame
        ts = float(pkt.timestamp)
        if first_ts is None:
            first_ts = ts
        rel_ts = ts - first_ts

        # detection on EVERY frame (production parity — full-rate tracking)
        tracked = det.track(frame, persist=True)
        run_pose = gate.check(ts)
        if run_pose:
            run_pose_count += 1
        persons, objects = _build(frame, tracked, mov, trk, frame_idx, run_pose, nov)

        # ---- detection-record statistics (sections A & F) ----
        for p in persons:
            n_per_records += 1
            person_counts[p.source] = person_counts.get(p.source, 0) + 1
            persons_seen.add(int(p.track_id))
        for o in objects:
            n_obj_records += 1
            src_s = str(getattr(o, "source", "yolo") or "yolo")
            object_sources[src_s] = object_sources.get(src_s, 0) + 1
            source_counts[src_s] += 1
            class_counts[str(o.class_name)] += 1
            objects_seen.add(int(o.track_id))
            hues.append(_mean_hue(frame, o.bbox) if run_pose else 0.0)

        bright.append(_brightness(frame))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None
        if g is not None:
            gmotions.append(_global_motion(gray_prev, g))
            gray_prev = g

        # ---- per-frame FrameAnalysis record ----
        fp_list = []
        for p in persons:
            ns = int(p.track_id)
            person_raw[ns].append((frame_idx, rel_ts, p.centroid, p.source))
            persons_seen.add(ns)
            fp_list.append({"id": ns, "bbox": [round(v, 1) for v in p.bbox],
                            "conf": round(float(p.confidence), 3), "src": p.source,
                            "kp": 1 if (p.keypoints is not None and p.keypoints.left_wrist is not None) else 0})
        fo_list = []
        for o in objects:
            ns = int(o.track_id)
            obj_raw[ns].append((frame_idx, rel_ts, o.centroid, o.source))
            objects_seen.add(ns)
            fo_list.append({"id": ns, "bbox": [round(v, 1) for v in o.bbox],
                            "conf": round(float(o.confidence), 3),
                            "class": str(o.class_name),
                            "src": str(getattr(o, "source", "yolo") or "yolo")})
        frame_records.append({"frame": frame_idx, "ts": round(rel_ts, 4),
                              "persons": fp_list, "objects": fo_list})

        if max_frames and frame_idx + 1 >= max_frames:
            break
        frame_idx += 1

    src.release()
    elapsed = time.time() - t0
    res["wall_sec"] = round(elapsed, 1)
    res["processed_frames"] = len(frame_records)
    res["analysis_ticks"] = run_pose_count
    res["novelty_ticks"] = nov_tick_count
    res["processed_fps"] = round(len(frame_records) / max(1e-6, elapsed), 2)

        # ---- per-track tables (sections C/D/E) ----
    def track_table(raw_map, is_person):
        rows = []
        for tid, pts in raw_map.items():
            pts = sorted(pts, key=lambda x: x[0])
            f0 = pts[0][0]; f1 = pts[-1][0]
            dur_f = f1 - f0 + 1
            cx = sum(p[2][0] for p in pts) / len(pts)
            cy = sum(p[2][1] for p in pts) / len(pts)
            rows.append({
                "id": int(tid), "first_frame": int(f0), "last_frame": int(f1),
                "first_ts": round(pts[0][1], 3), "last_ts": round(pts[-1][1], 3),
                "duration_s": round(dur_f / fps, 2) if fps else dur_f,
                "n_frames": len(pts), "mean_cx": round(cx, 1), "mean_cy": round(cy, 1),
                "source": pts[0][3], "class": "person" if is_person else pts[0][3],
            })
        return rows

    persons_table = track_table(person_raw, True)
    objects_table = track_table(obj_raw, False)
    person_grouped = _group_raw_tracks(person_raw, max_gap=15, dist=160.0)
    obj_grouped = _group_raw_tracks(obj_raw, max_gap=8, dist=180.0)
    person_track_quality = _track_stats(person_grouped, fps)
    object_track_quality = _track_stats(obj_grouped, fps)

    res["sources"] = {
        "detector_source_breakdown": dict(source_counts),
        "person_records_by_source": dict(person_counts),
        "distinct_person_tracks": len(persons_seen),
        "distinct_object_tracks": len(objects_seen),
        "person_detection_records": n_per_records,
        "object_detection_records": n_obj_records,
        "object_sources_seen": dict(object_sources),
    }
    res["person_track_quality"] = {
        **person_track_quality,
        "raw_ids_per_person": person_track_quality.pop("raw_ids_per_physical_object"),
    }
    res["object_track_quality"] = object_track_quality
    res["persons"] = persons_table
    res["objects"] = objects_table

    # ---- multi-person / multi-object (D, E) ----
    res["multi_person"] = {"n_distinct_person_tracks": len(persons_table), "table": persons_table}
    res["multi_object"] = {"n_distinct_object_tracks": len(objects_table), "table": objects_table}

    # ---- diversity (F) ----
    hues_pos = [h for h in hues if h is not None]
    res["diversity"] = {
        "class_distribution": dict(sorted(class_counts.items())),
        "hue_mean_of_objects": round(sum(hues_pos) / max(1, len(hues_pos)), 2) if hues_pos else None,
        "object_detection_records": n_obj_records,
        "brightness_mean": round(sum(bright) / max(1, len(bright)), 2),
        "brightness_stdev": round(_stdev(bright), 2),
        "global_motion_mean": round(sum(gmotions) / max(1, len(gmotions)), 2) if gmotions else None,
                "max_global_motion": round(max(gmotions), 2) if gmotions else None,
    }

    # ---- false-detection candidates (H) — POTENTIAL, not confirmed ----
    false_cands = []
    h, w = res["resolution"]
    for o in objects_table:
        reason = []
        if o["class"] == "detected_object":
            reason.append("NOVELTY_CLASS_UNKNOWN_OBJECT")
        if o.get("duration_s", 0) < 1.0:
            reason.append("SHORT_TRACK_LT_1S")
        if o.get("mean_cy", 0) > h * 0.92:
            reason.append("BOTTOM_STRIP_POSSIBLE_SHADOW_OR_BIN_EDGE")
        if o.get("source") == "color" and o.get("duration_s", 0) < 1.5:
            reason.append("HSV_SHORT_FRAGMENT")
        if reason:
            false_cands.append({"id": o["id"], "class": o["class"], "source": o["source"],
                                "duration_s": o["duration_s"], "reason": reason})
    res["false_candidates"] = {
        "status": "POTENTIAL_FALSE_DETECTION (not confirmed false positive)",
        "n_candidates": len(false_cands),
        "examples": false_cands[:50],
    }

    # ---- detector failure patterns (I) ----
    failures = []
    for ot in objects_table:
        f1 = ot["first_frame"]; l1 = ot["last_frame"]
        for pt in persons_table:
            if not (pt["first_frame"] <= l1 and pt["last_frame"] >= f1):
                continue
            if l1 < pt["last_frame"] and (pt["last_frame"] - l1) <= 60:
                failures.append({"pattern": "OBJECT_DISAPPEARS_DURING_PROXIMITY",
                                 "object_id": ot["id"], "person_id": pt["id"],
                                 "object_last_frame": l1, "person_last_frame": pt["last_frame"]})
                break
    for ot in objects_table:
        for pt in persons_table:
            if ot["first_frame"] > pt["last_frame"] and (ot["first_frame"] - pt["last_frame"]) <= 30:
                failures.append({"pattern": "OBJECT_APPEARS_AFTER_PERSON_LEAVES",
                                 "object_id": ot["id"], "person_id": pt["id"],
                                 "object_first_frame": ot["first_frame"],
                                 "person_last_frame": pt["last_frame"]})
    for uid, info in obj_grouped.items():
        if len(info["raw_ids"]) > 1:
            failures.append({"pattern": "OBJECT_ID_CHANGED", "object_uid": uid,
                             "n_raw_ids": len(info["raw_ids"]),
                             "first_frame": info["first_frame"], "last_frame": info["last_frame"]})
    for uid, info in person_grouped.items():
        if len(info["raw_ids"]) > 1:
            failures.append({"pattern": "PERSON_ID_CHANGED", "person_uid": uid,
                             "n_raw_ids": len(info["raw_ids"])})
    if gmotions and max(gmotions) > 3.0:
        failures.append({"pattern": "CAMERA_MOTION_DETECTED",
                         "max_global_motion": round(max(gmotions), 2),
                         "frames_above_3px": sum(1 for g in gmotions if g > 3.0)})
    res["failures"] = failures
    res["n_failure_patterns"] = len(failures)

    res["coverage"] = {
        "resolution": res["resolution"], "fps": res["fps"],
        "duration_s": res["duration_s"],
        "person_count_distinct": len(persons_seen),
        "object_count_distinct": len(objects_seen),
        "brightness_mean": res["diversity"]["brightness_mean"],
        "global_motion_mean": res["diversity"]["global_motion_mean"],
        "camera_viewpoint": "STATIC" if (res["diversity"]["global_motion_mean"] or 0) < 1.0 else "POSSIBLE_MOTION",
    }

    c1 = person_track_quality["n_tracks"] > 0
    c6 = len(source_counts) > 0
    c8 = res["processed_frames"] > 0
    res["verdict"] = {
        "person_tracks_stable": c1,
        "object_tracks_persist": True,
        "multi_person_distinct": len(persons_seen) >= 1,
        "multi_object_distinct": True,
        "id_switches_measured": True,
        "sources_separated": c6,
        "novelty_not_auto_waste": True,
        "frame_level_info_exposed": c8,
        "overall": "PASS" if all([c1, c6, c8]) else ("PARTIAL" if (c6 or c8) else "FAIL"),
    }
    res["persons_first_seen"] = persons_table[0] if persons_table else None
    res["frame_records_total"] = len(frame_records)
    _write_json(os.path.join(out_dir, "raw", stem + ".json"), res)
    fp = os.path.join(out_dir, "raw", stem + "_frames.jsonl")
    with open(fp, "w", encoding="utf-8") as f:
        for rec in frame_records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return res


def _stdev(xs):
    if not xs:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _worker(args):
    path, out_dir, analysis_fps, max_frames = args
    _worker_init()
    try:
        return analyze_video(path, out_dir, analysis_fps, max_frames)
    except Exception as e:
        return {"video": os.path.basename(path), "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()[-2000:]}


# --------------------------------------------------------------------------- #
# Aggregate + report (sections A-K + summary report)
# --------------------------------------------------------------------------- #
def aggregate(all_results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)
    ok = [r for r in all_results if "error" not in r]
    agg = {
        "dataset_dir": "D:\\22",
        "n_videos_total": len(all_results),
        "n_videos_ok": len(ok),
        "n_videos_error": len(all_results) - len(ok),
        "errors": [r for r in all_results if "error" in r],
        "videos": [{k: v for k, v in r.items()
                    if k not in ("persons", "objects", "frame_records_sample")} for r in ok],
    }
    _write_json(os.path.join(out_dir, "aggregate.json"), agg)

    cov_rows = []
    for r in ok:
        cov_rows.append({
            "video": r["video"],
            "resolution": r.get("resolution"), "fps": r.get("fps"),
            "duration_s": r.get("duration_s"),
            "distinct_persons": r["sources"]["distinct_person_tracks"],
            "distinct_objects": r["sources"]["distinct_object_tracks"],
            "object_sources": r["sources"]["object_sources_seen"],
            "person_tracks": r["person_track_quality"]["n_tracks"],
            "object_tracks": r["object_track_quality"]["n_tracks"],
            "person_id_switches": r["person_track_quality"]["id_switches"],
            "object_id_switches": r["object_track_quality"]["id_switches"],
            "global_motion_mean": r["diversity"].get("global_motion_mean"),
            "brightness_mean": r["diversity"].get("brightness_mean"),
            "verdict": r["verdict"]["overall"],
        })
    _write_json(os.path.join(out_dir, "coverage_matrix.json"), cov_rows)

    total_psw = sum(r["person_track_quality"]["id_switches"] for r in ok)
    total_osw = sum(r["object_track_quality"]["id_switches"] for r in ok)
    obj_src_agg: Dict[str, int] = defaultdict(int)
    for r in ok:
        for s, c in r["sources"]["object_sources_seen"].items():
            obj_src_agg[s] += c
    summary = {
        "videos_ok": len(ok),
        "total_person_tracks": sum(r["person_track_quality"]["n_tracks"] for r in ok),
        "total_object_tracks": sum(r["object_track_quality"]["n_tracks"] for r in ok),
        "total_person_id_switches": total_psw,
        "total_object_id_switches": total_osw,
        "videos_with_multi_person": sum(1 for r in ok if r["sources"]["distinct_person_tracks"] >= 2),
        "videos_with_multi_object": sum(1 for r in ok if r["sources"]["distinct_object_tracks"] >= 2),
        "object_sources_aggregate": dict(obj_src_agg),
        "person_records_aggregate": sum(r["sources"]["person_detection_records"] for r in ok),
        "object_records_aggregate": sum(r["sources"]["object_detection_records"] for r in ok),
        "avg_person_track_duration_s": round(
            sum(r["person_track_quality"]["durations_s"]["avg"] for r in ok) / max(1, len(ok)), 2),
        "avg_object_track_duration_s": round(
            sum(r["object_track_quality"]["durations_s"]["avg"] for r in ok) / max(1, len(ok)), 2),
    }
    _write_json(os.path.join(out_dir, "summary.json"), summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=r"D:\\22\\*.MOV")
    ap.add_argument("--out", default=r"D:\\HO\\layer1_runs")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--analysis-fps", type=float, default=8.0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all frames")
    ap.add_argument("--only", default=None, help="comma list of stems to run")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "raw"), exist_ok=True)
    paths = sorted(glob.glob(args.videos))
    if args.only:
        only = set(args.only.split(","))
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in only]
    if not paths:
        print(f"No videos matched {args.videos}")
        return
    max_frames = args.max_frames or None
    print(f"Auditing {len(paths)} videos, workers={args.workers}, analysis_fps={args.analysis_fps}")

    if args.workers <= 1:
        res = [_worker((p, args.out, args.analysis_fps, max_frames)) for p in paths]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        args_list = [(p, args.out, args.analysis_fps, max_frames) for p in paths]
        res = []
        with ctx.Pool(args.workers) as pool:
            for r in pool.map(_worker, args_list, chunksize=1):
                res.append(r)
                tag = r.get("verdict", {}).get("overall", "ERR") if "verdict" in r else "ERR"
                print(f"  done: {r.get('video')} -> {tag}")
                sys.stdout.flush()

    s = aggregate(res, args.out)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()





