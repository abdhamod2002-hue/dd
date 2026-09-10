#!/usr/bin/env python3
"""
Phase 3 Step 12 (+Step 14) — real-video A/B validation (ANALYSIS ONLY).

Runs the production chain on D:\\22\\IMG_5306.MOV (src frames 0..2400, the
Phase-2 window) TWICE with identical inputs:
  BEFORE: PHASE3_DISABLE_DEDUP=1 (legacy detector output)
  AFTER:  dedup enabled (new code)

Chain per frame: YoloDetector.track(persist=True) -> BytetrackTracker.update
-> to_tracks() -> InferencePipeline.process_frame (auto_tune=False,
active_learning=False; pose=None by construction, novelty=None to isolate
the dedup variable — identical in both runs).

Records per run: per-frame person/object counts+boxes (tracker input),
confirmed events, detector wall/CPU time. Dedup micro-cost is measured
offline on realistic box counts (Step 14, no hidden cost).

Output: phase3_runs/video_validation/AB.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path(r"D:\22")
OUT_DIR = PROJECT_ROOT / "phase3_runs" / "video_validation"

VIDEO = os.environ.get("PHASE3_AB_VIDEO", "IMG_5306.MOV")
SRC_FIRST = int(os.environ.get("PHASE3_AB_FIRST", "0"))
SRC_LAST = int(os.environ.get("PHASE3_AB_LAST", "2400"))  # inclusive


def run_chain(disable: bool) -> dict:
    if disable:
        os.environ["PHASE3_DISABLE_DEDUP"] = "1"
    else:
        os.environ.pop("PHASE3_DISABLE_DEDUP", None)

    from inference.detection.yolo_detector import YoloDetector
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from inference.pipeline import InferencePipeline, PipelineConfig

    detector = YoloDetector()
    detector.load()
    detector.reset_tracking()
    tracker = BytetrackTracker()
    tracker.load()
    pipe = InferencePipeline(PipelineConfig(auto_tune=False,
                                            active_learning=False))

    cap = cv2.VideoCapture(str(DATA_ROOT / VIDEO))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, SRC_FIRST)

    n_frames = 0
    person_boxes_total = 0
    object_boxes_total = 0
    frames_with_dup_pattern = 0  # >1 object box sharing a near-identical area
    det_sec = 0.0
    t0 = time.time()
    for src in range(SRC_FIRST, SRC_LAST + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        ts = src / src_fps
        d0 = time.time()
        tracked = detector.track(frame, persist=True)
        det_sec += time.time() - d0
        tracker.update(tracked, n_frames)
        persons, objects = tracker.to_tracks(tracked)
        pipe.process_frame(frame, ts, persons, objects)
        n_frames += 1
        person_boxes_total += len(persons)
        object_boxes_total += len(objects)
        # Duplicate pattern at tracker-input level: two object boxes with
        # IoU>=0.5 or max-containment>=0.7 (same predicate as the fix).
        from inference.detection.yolo_detector import boxes_duplicate
        ob = [tuple(o.bbox) for o in objects]
        for i in range(len(ob)):
            for j in range(i + 1, len(ob)):
                if boxes_duplicate(ob[i], ob[j]):
                    frames_with_dup_pattern += 1
                    break
            else:
                continue
            break
    cap.release()
    wall = time.time() - t0
    return {
        "disabled": disable,
        "frames": n_frames,
        "person_boxes_total": person_boxes_total,
        "object_boxes_total": object_boxes_total,
        "frames_with_object_dup_pattern": frames_with_dup_pattern,
        "confirmed_events": len(pipe.events),
        "rejected": len(pipe.event_detector.rejected_events),
        "detector_cpu_sec": round(det_sec, 1),
        "wall_sec": round(wall, 1),
    }


def micro_bench() -> dict:
    """Honest dedup CPU cost on realistic per-frame box counts."""
    import random
    from inference.detection.yolo_detector import deduplicate_tracked, TrackedDetection
    random.seed(7)
    out = {}
    for n in (5, 10, 20):
        dets = []
        for i in range(n):
            x, y = random.uniform(0, 900), random.uniform(0, 1500)
            w, h = random.uniform(40, 300), random.uniform(60, 400)
            dets.append(TrackedDetection(
                i, "Garbage Bag", random.uniform(0.25, 0.9),
                (x, y, x + w, y + h), (x + w / 2, y + h / 2), False, "yolo"))
        t0 = time.perf_counter()
        reps = 200
        for _ in range(reps):
            deduplicate_tracked(dets)
        out[f"n={n}"] = round((time.perf_counter() - t0) / reps * 1000, 3)  # ms
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("PHASE3_AB_ONLY", "both")  # before|after|both
    print(f"video={VIDEO} src={SRC_FIRST}..{SRC_LAST} only={only}", flush=True)
    payload = {"video": VIDEO, "src_first": SRC_FIRST, "src_last": SRC_LAST}
    if only in ("before", "both"):
        payload["before"] = run_chain(disable=True)
        print("BEFORE: " + json.dumps(payload["before"]), flush=True)
        (OUT_DIR / "AB.json").write_text(json.dumps(payload, indent=2),
                                         encoding="utf-8")
    if only in ("after", "both"):
        payload["after"] = run_chain(disable=False)
        print("AFTER:  " + json.dumps(payload["after"]), flush=True)
        payload["dedup_micro_ms"] = micro_bench()
        print("DEDUP micro ms/frame: " + json.dumps(payload["dedup_micro_ms"]),
              flush=True)
        # Merge with any previously saved BEFORE arm.
        try:
            prev = json.loads((OUT_DIR / "AB.json").read_text(encoding="utf-8"))
            payload.setdefault("before", prev.get("before"))
        except Exception:
            pass
        (OUT_DIR / "AB.json").write_text(json.dumps(payload, indent=2),
                                         encoding="utf-8")
    print(f"wrote {OUT_DIR / 'AB.json'}", flush=True)


if __name__ == "__main__":
    main()
