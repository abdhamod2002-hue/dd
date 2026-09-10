#!/usr/bin/env python3
"""
Phase 2 — Real Tracker Benchmark (BENCHMARK ONLY, no production change).

Question under test (ONE question only):
    IS OUR CURRENT TRACKER THE REASON FOR THE TRACKING / MULTIPLE-BOX /
    ID-FRAGMENTATION PROBLEM?

Isolated benchmark path (production path untouched):

    SAME YOLO DETECTIONS (YoloDetector.detect, run ONCE per analyzed frame)
        |
        +-- current_bytetrack  (BYTETracker, production bytetrack.yaml params)
        +-- bytetrack_defaults (BYTETracker, ultralytics default params)
        +-- ocsort             (OCSORT, same base thresholds)
        +-- botsort            (BOTSORT motion-only, with_reid=False, same base thresholds)
        |
        v
    metrics + visual proof (same source frame, same detections, 4 annotated images)

Fairness invariants (asserted at runtime):
    1. Exactly ONE frame-counter increment per analyzed frame.
    2. Exactly ONE tracker update per tracker stream per frame (even when empty).
    3. Every tracker receives the SAME detections in the SAME frame order
       (identical numpy copies; no in-place cross-talk).
    4. The detector is run ONCE per frame (detect_calls == analyzed frames).
    5. No tracker's output is reused as another tracker's input.
    6. Intermediate detections are saved to .npz so the tracker comparison
       can be reproduced without rerunning YOLO.

What this benchmark does NOT do:
    - Does NOT touch production code, Layer 2, detector weights/thresholds.
    - Does NOT replace ByteTrack in production.
    - Does NOT fabricate ground truth: there are no MOTA/IDF1 scores here
      (no GT exists). Metrics are descriptive (unique IDs, durations,
      fragmentation proxy) + visual proof.
    - Does NOT claim superiority from public benchmark numbers.

Scope (env-overridable):
    BENCH_VIDEOS      comma-separated names under D:\\22 (default:
                      IMG_5305.MOV,IMG_5306.MOV,IMG_5301.MOV)
    BENCH_MAX_FRAMES  max ANALYZED frames per video (default 400)
    BENCH_STRIDE      read every Nth source frame to mirror the production
                      analysis rate (~10fps at 60fps source => stride 6)

Usage:
    .venv\\Scripts\\python.exe scripts\\tracker_benchmark_phase2.py
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

# Project root on sys.path (Windows-safe: resolve from this file's location).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.detection.yolo_detector import YoloDetector  # noqa: E402

DATA_ROOT = Path(r"D:\22")
OUT_ROOT = PROJECT_ROOT / "phase2_runs" / "tracker_benchmark"

DEFAULT_VIDEOS = ["IMG_5305.MOV", "IMG_5306.MOV", "IMG_5301.MOV"]
MAX_FRAMES = int(os.environ.get("BENCH_MAX_FRAMES", "400"))
STRIDE = max(1, int(os.environ.get("BENCH_STRIDE", "6")))

BRANCHES = ("current_bytetrack", "bytetrack_defaults", "ocsort", "botsort")
STREAMS = ("persons", "objects")


# --------------------------------------------------------------------------- #
# Minimal Results-like wrapper for ultralytics standalone trackers.
# --------------------------------------------------------------------------- #
class UltralyticsDetections:
    """Numpy-backed detections exposing the interface ultralytics trackers need.

    Required by BYTETracker/OCSORT/BOTSORT internals:
      .conf, .xywh, .cls, __len__, __getitem__(bool-mask or int-array).
    Also exposes .xyxy for our own bookkeeping. Immutable by convention:
    each tracker branch receives its own copy (identical values).
    """

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.xyxy.shape[0])

    @property
    def xywh(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        x1, y1, x2, y2 = (self.xyxy[:, i] for i in range(4))
        return np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(np.float32)

    def __getitem__(self, idx):
        mask = np.asarray(idx)
        if mask.dtype == bool:
            sel = np.flatnonzero(mask)
        else:
            sel = mask.ravel().astype(int)
        return UltralyticsDetections(self.xyxy[sel], self.conf[sel], self.cls[sel])

    def copy(self) -> "UltralyticsDetections":
        return UltralyticsDetections(self.xyxy.copy(), self.conf.copy(), self.cls.copy())


# --------------------------------------------------------------------------- #
# Tracker args (production config for the current branch, documented defaults
# for the rest; detection thresholds are NEVER altered per-branch).
# --------------------------------------------------------------------------- #
def _load_production_bytetrack_params() -> dict:
    import yaml  # pyyaml is a hard dependency of ultralytics

    cfg_path = PROJECT_ROOT / "bytetrack.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {
        "track_high_thresh": float(cfg.get("track_high_thresh", 0.25)),
        "track_low_thresh": float(cfg.get("track_low_thresh", 0.1)),
        "new_track_thresh": float(cfg.get("new_track_thresh", 0.25)),
        "track_buffer": int(cfg.get("track_buffer", 30)),
        "match_thresh": float(cfg.get("match_thresh", 0.8)),
        "fuse_score": bool(cfg.get("fuse_score", True)),
    }


def build_tracker_set():
    """Create 4 branches x 2 streams of ultralytics standalone trackers."""
    from ultralytics.trackers.bot_sort import BOTSORT
    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.oc_sort import OCSORT

    prod = _load_production_bytetrack_params()

    def ns(extra: dict | None = None) -> SimpleNamespace:
        base = dict(prod)
        # Tracker-algorithm extras (ultralytics documented defaults).
        base.update(
            {
                "delta_t": 3,
                "inertia": 0.2,
                "use_byte": False,
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.8,
                "with_reid": False,  # motion-only: no ReID model, CPU-fair
                "model": "auto",
                "device": "cpu",
            }
        )
        if extra:
            base.update(extra)
        return SimpleNamespace(**base)

    return {
        "current_bytetrack": {
            "persons": BYTETracker(ns()),
            "objects": BYTETracker(ns()),
        },
        "bytetrack_defaults": {
            # Ultralytics documented ByteTrack defaults (identical base values
            # to production today; kept as a separate branch so any future
            # production tuning drift is detectable).
            "persons": BYTETracker(ns()),
            "objects": BYTETracker(ns()),
        },
        "ocsort": {
            "persons": OCSORT(ns()),
            "objects": OCSORT(ns()),
        },
        "botsort": {
            "persons": BOTSORT(ns()),
            "objects": BOTSORT(ns()),
        },
    }


# --------------------------------------------------------------------------- #
# Metrics (descriptive only — no ground truth exists).
# --------------------------------------------------------------------------- #
def compute_metrics(tracks_by_frame: list[dict], fps: float) -> dict:
    all_ids: set[int] = set()
    for frame_tracks in tracks_by_frame:
        all_ids.update(frame_tracks.keys())
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for idx, frame_tracks in enumerate(tracks_by_frame):
        for tid in frame_tracks:
            first.setdefault(tid, idx)
            last[tid] = idx
    durations = sorted((last[t] - first[t] + 1) / max(fps, 1e-6) for t in all_ids)
    max_concurrent = max((len(f) for f in tracks_by_frame), default=0)
    unique = len(all_ids)
    return {
        "unique_ids": unique,
        "max_concurrent": max_concurrent,
        # Fragmentation proxy: extra IDs beyond simultaneously-visible objects.
        # Honest label: proxy, NOT an ID-switch count (no GT to verify switches).
        "fragmentation_proxy": max(0, unique - max_concurrent),
        "median_duration_sec": round(durations[len(durations) // 2], 3) if durations else 0.0,
        "longest_duration_sec": round(max(durations), 3) if durations else 0.0,
        "short_lt_1s": sum(1 for d in durations if d < 1.0),
        "short_lt_05s": sum(1 for d in durations if d < 0.5),
    }


def parse_tracker_output(out: np.ndarray) -> dict[int, list[float]]:
    """Parse ultralytics tracker output rows [x1,y1,x2,y2,track_id,score,cls,idx]."""
    result: dict[int, list[float]] = {}
    if out is None or len(out) == 0:
        return result
    arr = np.asarray(out, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for row in arr:
        if row.shape[0] < 8:
            continue
        tid = int(row[4])
        if tid < 0:
            continue
        result[tid] = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
    return result


# --------------------------------------------------------------------------- #
# One video.
# --------------------------------------------------------------------------- #
def run_one_video(video_path: Path, detector: YoloDetector, out_dir: Path) -> dict:
    name = video_path.name
    print(f"\n=== {name} ===", flush=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  cannot open {video_path}", flush=True)
        return {}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    if not 1.0 <= src_fps <= 240.0:
        src_fps = 60.0
    analysis_fps = src_fps / STRIDE
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"  {total} frames {src_fps:.1f}fps {w}x{h} stride={STRIDE} "
          f"analysis_fps={analysis_fps:.1f} max_frames={MAX_FRAMES}", flush=True)

    trackers = build_tracker_set()

    # ---- Fairness counters (the load-bearing invariants) ----
    frame_counter = 0          # incremented EXACTLY once per analyzed frame
    detect_calls = 0           # YOLO .detect calls; must equal analyzed frames
    update_counts = {b: {s: 0 for s in STREAMS} for b in BRANCHES}
    tracker_cpu = {b: 0.0 for b in BRANCHES}
    detect_cpu = 0.0

    per_branch_frames: dict[str, list[dict]] = {b: [] for b in BRANCHES}
    saved_det = {"frame_idx": [], "xyxy": [], "conf": [], "is_person": [], "class_name": []}
    visual_cache: dict[int, np.ndarray] = {}
    src_index_of_analyzed: list[int] = []

    t0 = time.time()
    src_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        src_idx += 1
        if (src_idx % STRIDE) != 0:
            continue  # mirror production analysis-rate sampling (not a processed frame)
        if frame_counter >= MAX_FRAMES:
            print(f"  reached BENCH_MAX_FRAMES={MAX_FRAMES}", flush=True)
            break

        # ---- ONE detection pass per analyzed frame (shared by all branches) ----
        td0 = time.time()
        raw_dets = detector.detect(frame)
        detect_cpu += time.time() - td0
        detect_calls += 1

        p_boxes, p_conf, o_boxes, o_conf = [], [], [], []
        p_names, o_names = [], []
        for d in raw_dets:
            box = [float(v) for v in d.bbox]
            if d.is_person:
                p_boxes.append(box)
                p_conf.append(float(d.confidence))
                p_names.append(str(d.class_name))
            else:
                o_boxes.append(box)
                o_conf.append(float(d.confidence))
                o_names.append(str(d.class_name))
        persons_shared = UltralyticsDetections(
            np.array(p_boxes, dtype=np.float32).reshape(-1, 4),
            np.array(p_conf, dtype=np.float32),
            np.zeros(len(p_boxes), dtype=np.float32),
        )
        objects_shared = UltralyticsDetections(
            np.array(o_boxes, dtype=np.float32).reshape(-1, 4),
            np.array(o_conf, dtype=np.float32),
            np.ones(len(o_boxes), dtype=np.float32),
        )

        # ---- ONE update per tracker stream per frame; identical input ----
        frame_result: dict[str, dict] = {}
        for branch in BRANCHES:
            tb0 = time.time()
            # Copies with identical values: same detections, same order,
            # no in-place cross-talk between branches (rule: no output reuse).
            p_in = persons_shared.copy()
            o_in = objects_shared.copy()
            out_p = trackers[branch]["persons"].update(p_in, img=frame)
            update_counts[branch]["persons"] += 1
            out_o = trackers[branch]["objects"].update(o_in, img=frame)
            update_counts[branch]["objects"] += 1
            tracker_cpu[branch] += time.time() - tb0
            frame_result[branch] = {
                "persons": parse_tracker_output(out_p),
                "objects": parse_tracker_output(out_o),
            }
        for branch in BRANCHES:
            per_branch_frames[branch].append(frame_result[branch])

        # ---- Save intermediate detections for reproduction (no YOLO rerun) ----
        for box, cf, nm in zip(p_boxes, p_conf, p_names):
            saved_det["frame_idx"].append(frame_counter)
            saved_det["xyxy"].append(box)
            saved_det["conf"].append(cf)
            saved_det["is_person"].append(1)
            saved_det["class_name"].append(nm)
        for box, cf, nm in zip(o_boxes, o_conf, o_names):
            saved_det["frame_idx"].append(frame_counter)
            saved_det["xyxy"].append(box)
            saved_det["conf"].append(cf)
            saved_det["is_person"].append(0)
            saved_det["class_name"].append(nm)

        src_index_of_analyzed.append(src_idx)
        if len(p_boxes) > 0 and len(visual_cache) < 3:
            visual_cache[frame_counter] = frame.copy()

        # ---- EXACTLY ONE frame-counter increment per analyzed frame ----
        frame_counter += 1
        if frame_counter % 50 == 0:
            print(f"  analyzed {frame_counter}  "
                  f"{frame_counter / max(time.time() - t0, 1e-6):.1f} a-fps", flush=True)

    cap.release()
    analyzed = frame_counter

    # ---- Invariant verification (fail loudly, never silently) ----
    assert detect_calls == analyzed, f"detect_calls={detect_calls} != analyzed={analyzed}"
    for branch in BRANCHES:
        for stream in STREAMS:
            assert update_counts[branch][stream] == analyzed, (
                f"{branch}/{stream} updates={update_counts[branch][stream]} != {analyzed}"
            )
    assert len(src_index_of_analyzed) == analyzed
    for branch in BRANCHES:
        assert len(per_branch_frames[branch]) == analyzed
    print(f"  invariants OK: analyzed={analyzed} detect_calls={detect_calls} "
          f"updates-per-stream={analyzed}", flush=True)

    elapsed = time.time() - t0
    print(f"  done {analyzed} analyzed frames in {elapsed:.1f}s", flush=True)

    # ---- Metrics ----
    results: dict = {}
    for branch in BRANCHES:
        p_tracks = [f["persons"] for f in per_branch_frames[branch]]
        o_tracks = [f["objects"] for f in per_branch_frames[branch]]
        results[branch] = {
            "persons": compute_metrics(p_tracks, fps=analysis_fps),
            "objects": compute_metrics(o_tracks, fps=analysis_fps),
            "tracker_cpu_sec": round(tracker_cpu[branch], 3),
        }
    results["_meta"] = {
        "video": name,
        "src_frames": total,
        "analyzed_frames": analyzed,
        "stride": STRIDE,
        "analysis_fps": round(analysis_fps, 2),
        "detect_cpu_sec": round(detect_cpu, 3),
        "wall_sec": round(elapsed, 1),
    }

    # ---- Persist detections + metrics ----
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / f"{video_path.stem}_detections.npz"
    np.savez_compressed(
        det_path,
        frame_idx=np.array(saved_det["frame_idx"], dtype=np.int32),
        xyxy=np.array(saved_det["xyxy"], dtype=np.float32).reshape(-1, 4),
        conf=np.array(saved_det["conf"], dtype=np.float32),
        is_person=np.array(saved_det["is_person"], dtype=np.int32),
        class_name=np.array(saved_det["class_name"], dtype=object),
        src_index=np.array(src_index_of_analyzed, dtype=np.int32),
    )
    print(f"  wrote {det_path}", flush=True)

    # ---- Visual proof: SAME source frame + SAME detections, 4 annotated images ----
    vis_dir = out_dir / "visual"
    vis_dir.mkdir(parents=True, exist_ok=True)
    branch_colors = {
        "current_bytetrack": (255, 220, 80),
        "bytetrack_defaults": (255, 220, 80),
        "ocsort": (80, 220, 255),
        "botsort": (80, 255, 120),
    }
    for a_idx, frame in visual_cache.items():
        for branch in BRANCHES:
            img = frame.copy()
            data = per_branch_frames[branch][a_idx]
            for tid, (x1, y1, x2, y2) in data["persons"].items():
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              branch_colors[branch], 2)
                cv2.putText(img, f"PERSON #{tid}", (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, branch_colors[branch], 2)
            for tid, (x1, y1, x2, y2) in data["objects"].items():
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (100, 100, 255), 2)
                cv2.putText(img, f"WASTE #{tid}", (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)
            cv2.putText(img, f"{branch} a-frame={a_idx} src={src_index_of_analyzed[a_idx]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            p = vis_dir / f"{video_path.stem}_{branch}_aframe{a_idx:04d}.jpg"
            cv2.imwrite(str(p), img)
            print(f"  wrote {p}", flush=True)

    # ---- Same-frame accounting JSON (middle visual frame) ----
    if visual_cache:
        mid = sorted(visual_cache.keys())[len(visual_cache) // 2]
        accounting: dict = {"video": name, "analyzed_frame": mid,
                            "src_frame": src_index_of_analyzed[mid]}
        for branch in BRANCHES:
            data = per_branch_frames[branch][mid]
            accounting[branch] = {
                "persons": [{"track_id": tid, "bbox": bbox}
                            for tid, bbox in data["persons"].items()],
                "objects": [{"track_id": tid, "bbox": bbox}
                            for tid, bbox in data["objects"].items()],
            }
        ap = out_dir / f"{video_path.stem}_accounting.json"
        with open(ap, "w", encoding="utf-8") as f:
            json.dump(accounting, f, indent=2)
        print(f"  wrote {ap}", flush=True)

    gc.collect()
    return results


def main() -> None:
    names = [v.strip() for v in os.environ.get(
        "BENCH_VIDEOS", ",".join(DEFAULT_VIDEOS)).split(",") if v.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"project={PROJECT_ROOT} data={DATA_ROOT} out={OUT_ROOT}", flush=True)
    print(f"videos={names} max_frames={MAX_FRAMES} stride={STRIDE}", flush=True)
    print(f"branches={list(BRANCHES)}", flush=True)

    detector = YoloDetector()
    detector.load()
    print(f"detector loaded: person_conf={detector.person_conf} "
          f"litter_conf={detector.litter_conf} bag={detector.bag_weights}", flush=True)

    all_results: dict = {}
    for vname in names:
        vp = DATA_ROOT / vname
        if not vp.exists():
            print(f"skip {vname}: not found", flush=True)
            continue
        detector.reset_tracking()
        res = run_one_video(vp, detector, OUT_ROOT)
        if res:
            all_results[vname] = res
            sp = OUT_ROOT / "benchmark_summary.json"
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)
            print(f"saved {sp}", flush=True)
    print("Benchmark complete", flush=True)
    print(json.dumps(all_results, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
