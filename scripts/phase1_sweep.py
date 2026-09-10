"""Phase-1 threshold sweep harness (audit-only, does not touch production behaviour).

WHY THIS DESIGN
---------------
The user requires before/after numbers on the 5 real videos for every candidate
MIN_STATIONARY_FRAMES value. Running YOLO+ByteTrack+MoveNet once per candidate
value would take hours on CPU and, worse, the heavy stack is not bit-exact
deterministic across long runs, so a threshold effect could be confused with
run-to-run noise.

So we split it in two stages:

  STAGE A (slow, once)  -- "capture"
      Run the EXACT production component chain
      (VideoFileSource -> YoloDetector.track -> BytetrackTracker -> MovenetPose
       -> build_tracks_real -> InferencePipeline.should_analyze throttle)
      and dump, for every *analysis tick*, the raw Track fields the event
      detector consumes. Nothing is synthesised. Cached to JSONL.

  STAGE B (fast, many)  -- "replay"
      Re-feed the cached ticks into a fresh LitteringEventDetector with a
      chosen EventDetectorConfig. Pure logic, milliseconds. Every candidate
      threshold therefore sees byte-identical inputs, so any difference in the
      outcome is attributable to the threshold alone.

Faithfulness note: the cached records reproduce exactly what
InferencePipeline._detector_person / _detector_bag hand to the detector,
INCLUDING the fact that the production adapter never populates
DetectorBag.timestamp (it stays 0.0). That is deliberate -- see the report.

Usage
-----
    .venv/Scripts/python.exe scripts/phase1_sweep.py capture           # stage A
    .venv/Scripts/python.exe scripts/phase1_sweep.py replay            # stage B
    .venv/Scripts/python.exe scripts/phase1_sweep.py both              # A then B
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

VIDEO_DIR = Path(r"D:\W")
CACHE_DIR = REPO / ".audit" / "phase1" / "tracks"
OUT_DIR = REPO / ".audit" / "phase1"
ANALYSIS_FPS = 8.0  # production value (backend/routers/analysis.py + run_production.py)

VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv"}


# --------------------------------------------------------------------------- #
# STAGE A -- capture real tracks, once per video
# --------------------------------------------------------------------------- #
def _kp_to_dict(kp):
    if kp is None:
        return None
    out = {}
    for name in ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder",
                 "torso_center", "nose"):
        v = getattr(kp, name, None)
        out[name] = [float(v[0]), float(v[1])] if v is not None else None
    out["nose_confidence"] = float(getattr(kp, "nose_confidence", 0.0) or 0.0)
    return out


def capture_video(path: Path) -> dict:
    """Run the production CV stack once and cache every analysis tick."""
    from inference.capture.camera_source import VideoFileSource
    from inference.detection.yolo_detector import YoloDetector
    from inference.pose.movenet_pose import MovenetPose
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from inference.pipeline import InferencePipeline, PipelineConfig
    from scripts.run_pipeline import build_tracks_real

    print(f"\n[capture] {path.name}", flush=True)
    source = VideoFileSource(str(path))
    if not source.open():
        return {"video": path.name, "error": "cannot open"}

    detector = YoloDetector()
    detector.load()
    detector.reset_tracking()
    tracker = BytetrackTracker()
    tracker.load()
    movenet = MovenetPose()
    movenet.load()

    cfg = PipelineConfig(buffer_seconds=8.0, analysis_fps=ANALYSIS_FPS,
                         camera_id="phase1", post_backend_url=None)
    pipe = InferencePipeline(cfg)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_fp = (CACHE_DIR / f"{path.stem}.jsonl").open("w", encoding="utf-8")
    meta_fp = (CACHE_DIR / f"{path.stem}.meta.json").open("w", encoding="utf-8")

    frame_idx = 0
    ticks = 0
    first_ts = None
    t0 = time.time()
    # Replicate InferencePipeline's analysis throttle EXACTLY.
    # should_analyze() reads self._last_analysis_ts, which is only advanced
    # inside process_frame() -- and we deliberately never call process_frame()
    # (it would run the event detector and we only want the raw tracks). So we
    # mirror that one piece of state here, same predicate, same ordering:
    #   should_analyze(ts) -> if pass, _last_analysis_ts = ts
    last_analysis_ts = 0.0
    period = 1.0 / ANALYSIS_FPS
    try:
        for pkt in source:
            if first_ts is None:
                first_ts = float(pkt.timestamp)
            tracked = detector.track(pkt.frame, persist=True)
            ts = float(pkt.timestamp)
            run_pose = (last_analysis_ts <= 0 or (ts - last_analysis_ts) >= period)
            if run_pose:
                last_analysis_ts = ts
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker, frame_idx, run_pose=run_pose)
            if run_pose:
                rec = {
                    "frame_index": int(frame_idx),
                    "t_rel": round(float(pkt.timestamp) - first_ts, 4),
                    "timestamp": float(pkt.timestamp),
                    "persons": [
                        {
                            "track_id": int(p.track_id),
                            "bbox": [float(v) for v in p.bbox],
                            "confidence": float(getattr(p, "confidence", 0.0) or 0.0),
                            "keypoints": _kp_to_dict(getattr(p, "keypoints", None)),
                        }
                        for p in persons
                    ],
                    "objects": [
                        {
                            "track_id": int(o.track_id),
                            "bbox": [float(v) for v in o.bbox],
                            "confidence": float(getattr(o, "confidence", 0.0) or 0.0),
                            "class_name": str(o.class_name),
                            "source": str(getattr(o, "source", "yolo") or "yolo"),
                        }
                        for o in objects
                    ],
                }
                out_fp.write(json.dumps(rec) + "\n")
                ticks += 1
            frame_idx += 1
            if frame_idx % 60 == 0:
                el = time.time() - t0
                print(f"    {frame_idx}/{source.total_frames}  {frame_idx/el:.2f} fps  ticks={ticks}",
                      flush=True)
    finally:
        out_fp.close()
        source.release()

    meta = {
        "video": path.name,
        "path": str(path),
        "fps": round(float(source.fps or 0), 3),
        "total_frames": int(source.total_frames),
        "duration_s": round(float(source.total_frames) / max(1e-6, float(source.fps or 30)), 3),
        "width": int(source.width),
        "height": int(source.height),
        "analysis_fps": ANALYSIS_FPS,
        "analysis_ticks": ticks,
        "capture_wall_s": round(time.time() - t0, 1),
    }
    meta_fp.write(json.dumps(meta, indent=2))
    meta_fp.close()
    print(f"    cached {ticks} analysis ticks in {meta['capture_wall_s']}s", flush=True)
    return meta


# --------------------------------------------------------------------------- #
# STAGE B -- replay cached ticks through the detector with a given config
# --------------------------------------------------------------------------- #
def load_ticks(stem: str):
    fp = CACHE_DIR / f"{stem}.jsonl"
    if not fp.exists():
        raise FileNotFoundError(fp)
    return [json.loads(line) for line in fp.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay(stem: str, overrides: dict, base_overrides: dict | None = None):
    """Re-run LitteringEventDetector on cached ticks with a modified config."""
    from littering_event_detector import (
        DetectorBag, DetectorKeypoints, DetectorPerson,
        EventDetectorConfig, LitteringEventDetector, load_event_config,
    )

    cfg = load_event_config()
    data = cfg.to_dict()
    data.update(base_overrides or {})
    data.update(overrides or {})
    cfg = EventDetectorConfig.from_dict(data)

    det = LitteringEventDetector(cfg)
    det.reset()
    for rec in load_ticks(stem):
        persons = []
        for p in rec["persons"]:
            kpd = p.get("keypoints")
            kp = None
            if kpd:
                kp = DetectorKeypoints(
                    left_wrist=tuple(kpd["left_wrist"]) if kpd.get("left_wrist") else None,
                    right_wrist=tuple(kpd["right_wrist"]) if kpd.get("right_wrist") else None,
                    torso_center=tuple(kpd["torso_center"]) if kpd.get("torso_center") else None,
                    left_shoulder=tuple(kpd["left_shoulder"]) if kpd.get("left_shoulder") else None,
                    right_shoulder=tuple(kpd["right_shoulder"]) if kpd.get("right_shoulder") else None,
                )
            persons.append(DetectorPerson(
                track_id=int(p["track_id"]),
                bbox=tuple(p["bbox"]),
                confidence=float(p["confidence"]),
                keypoints=kp,
            ))
        bags = []
        for o in rec["objects"]:
            src = o.get("source", "yolo")
            bags.append(DetectorBag(
                track_id=int(o["track_id"]),
                bbox=tuple(o["bbox"]),
                confidence=float(o["confidence"]),
                class_name=str(o["class_name"]),
                source=src,
                yolo_confirmed=(src == "yolo"),
            ))
        det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_index"]))
    det.finalize()
    return det, cfg


def summarise(det, video: str) -> dict:
    """One row per (person,bag) decision actually produced by the detector."""
    rows = []
    for ev in det.confirmed_events + det.rejected_events:
        rows.append({
            "video": video,
            "confirmed": bool(ev.confirmed),
            "reason": str(ev.reason),
            "confidence": round(float(ev.confidence), 4),
            "pair": f"P{ev.person_track_id}->B{ev.bag_track_id}",
            "person_id": int(ev.person_track_id),
            "bag_id": int(ev.bag_track_id),
            "bag_class": str(ev.bag_class),
            "evidence": ev.evidence,
            "details": ev.details,
            "frames": ev.frames,
        })
    return {
        "video": video,
        "confirmed": len(det.confirmed_events),
        "rejected": len(det.rejected_events),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    vids = sorted(p for p in VIDEO_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTS)

    if mode in ("capture", "both"):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        metas = []
        for v in vids:
            metas.append(capture_video(v))
        (OUT_DIR / "capture_meta.json").write_text(json.dumps(metas, indent=2), encoding="utf-8")
        print("\n=== CAPTURE SUMMARY ===")
        for m in metas:
            if "error" in m:
                print(f"{m['video']:<14} ERROR {m['error']}")
                continue
            print(f"{m['video']:<14} {m['duration_s']:>6.2f}s "
                  f"{m['width']}x{m['height']}  ticks={m['analysis_ticks']:>3}  "
                  f"capture={m['capture_wall_s']}s")

    if mode in ("replay", "both"):
        run_sweep()

    return 0


def run_sweep() -> None:
    """Sweep MIN_STATIONARY_FRAMES; report per-video outcome at each value."""
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    metas = json.loads((OUT_DIR / "capture_meta.json").read_text(encoding="utf-8"))
    ticks_by_video = {m["video"]: m for m in metas}
    ANALYSIS = ANALYSIS_FPS

    # Fine steps near the baseline (8 ticks = 1.00s) because IMG_5117 is only
    # 4.56 s long and confirms at t=4.54 s -- there is almost no headroom, so
    # the safe ceiling is likely only a few ticks above the current value.
    # Then coarse steps to show where the longer clips break.
    candidates = [8, 9, 10, 11, 12, 13, 14, 16, 20, 24, 32, 40]

    results = {}
    for n in candidates:
        results[n] = {}
        for stem in stems:
            det, cfg = replay(stem, {"min_stationary_frames": n})
            s = summarise(det, stem)
            results[n][stem] = s

    print("\n=== PHASE 1 SWEEP: MIN_STATIONARY_FRAMES (analysis_fps=%.1f, 8 ticks = 1.00s) ===" % ANALYSIS)
    hdr = f"{'frames':>7} {'sec':>6} | " + " | ".join(f"{s:<26}" for s in stems)
    print(hdr)
    print("-" * len(hdr))
    for n in candidates:
        sec = n / ANALYSIS
        cells = []
        for stem in stems:
            s = results[n][stem]
            if s["confirmed"]:
                cell = f"CONFIRMED x{s['confirmed']}"
            elif s["rejected"]:
                reasons = sorted({r["reason"] for r in s["rows"]})
                cell = "REJ:" + ",".join(r[:24] for r in reasons)
            else:
                cell = "no-decision"
            cells.append(f"{cell:<26}")
        print(f"{n:>7} {sec:>6.2f} | " + " | ".join(cells))

    (OUT_DIR / "sweep_stationary.json").write_text(json.dumps(
        {str(k): v for k, v in results.items()}, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'sweep_stationary.json'}")


if __name__ == "__main__":
    sys.exit(main())
