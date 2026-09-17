#!/usr/bin/env python3
"""Prove Path A (CFR+determinism) vs Path B (raw CLI-style) divergence.

Runs the SAME short clip twice in-process with identical models:
  Path A: ensure_cfr_source + configure_determinism(0) + AdaptiveEventDetector(deterministic=True)
  Path B: raw VideoFileSource + NO configure_determinism + AdaptiveEventDetector(deterministic=False)
           still freeze_learning_writes so learning.json is not mutated.

Records confirmed counts, rejection reasons, and effective tier-0 thresholds.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "project_audit" / "_comprehensive_plan_path_divergence.json"
VIDEO_CANDIDATES = [
    Path("/data/22/IMG_5117.MOV"),
    Path(r"D:\22\IMG_5117.MOV"),
]


def _find_video() -> Path:
    for p in VIDEO_CANDIDATES:
        if p.is_file():
            return p
    raise FileNotFoundError("IMG_5117.MOV not found under /data/22 or D:/22")


def _run(path: Path, *, cfr: bool, deterministic: bool, tag: str) -> dict:
    os.environ["TFHUB_CACHE_DIR"] = str(ROOT / "models" / "movenet")
    from backend.services.video_normalizer import ensure_cfr_source
    from inference.capture.camera_source import VideoFileSource
    from inference.detection.novelty_detector import NoveltyConfig, NoveltyDetector
    from inference.detection.yolo_detector import YoloDetector
    from inference.pipeline import InferencePipeline, PipelineConfig
    from inference.pose.movenet_pose import MovenetPose
    from inference.runtime_determinism import configure_determinism
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from scripts.run_pipeline import build_tracks_real

    if deterministic:
        configure_determinism(0)
        os.environ["MOTARED_DETERMINISTIC"] = "1"
    else:
        os.environ.pop("MOTARED_DETERMINISTIC", None)

    with tempfile.TemporaryDirectory(prefix=f"path_{tag}_") as td:
        src_path = path
        if cfr:
            src_path = Path(ensure_cfr_source(path, td, stem="source_CFR"))
        source = VideoFileSource(str(src_path))
        assert source.open(), f"cannot open {src_path}"
        detector = YoloDetector()
        detector.load()
        detector.reset_tracking()
        tracker = BytetrackTracker()
        tracker.load()
        nov_cfg = NoveltyConfig.from_yaml()
        nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
        movenet = MovenetPose()
        movenet.load()
        pipe = InferencePipeline(
            PipelineConfig(
                buffer_seconds=8.0,
                analysis_fps=8.0,
                camera_id=f"path-{tag}",
                post_backend_url=None,
                auto_tune=True,
                deterministic=bool(deterministic),
            )
        )
        pipe.event_detector.reset()
        t0 = time.time()
        n = 0
        last_ts = 0.0
        for pkt in source:
            n += 1
            last_ts = float(pkt.timestamp)
            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            persons, objects = build_tracks_real(
                pkt.frame,
                tracked,
                movenet,
                tracker,
                n - 1,
                run_pose=run_pose,
                nov=nov,
            )
            pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
        source.release()
        pipe.finalize(last_ts)
        elapsed = time.time() - t0
        ed = pipe.event_detector
        conf = list(ed.confirmed_events)
        rej = list(ed.rejected_events)
        cfg = ed.config
        times = []
        for e in conf:
            ts = (e.timestamps or {}).get("confirmed") or (e.timestamps or {}).get(
                "departure"
            )
            if ts is not None:
                times.append(float(ts))
        return {
            "tag": tag,
            "cfr": cfr,
            "deterministic": deterministic,
            "frames": n,
            "elapsed_sec": round(elapsed, 2),
            "n_confirmed": len(conf),
            "n_rejected": len(rej),
            "pred_times": times,
            "rejection_reasons": dict(Counter(getattr(e, "reason", None) for e in rej)),
            "thresholds": {
                "min_carried_frames": int(cfg.min_carried_frames),
                "release_distance_ratio": float(cfg.release_distance_ratio),
                "departure_motion_ratio": float(cfg.departure_motion_ratio),
                "smoothing_window": int(cfg.smoothing_window),
            },
        }


def main() -> None:
    video = _find_video()
    a = _run(video, cfr=True, deterministic=True, tag="A_cfr_det")
    b = _run(video, cfr=False, deterministic=False, tag="B_raw_cli")
    out = {
        "video": str(video),
        "path_A": a,
        "path_B": b,
        "decision_differs": a["n_confirmed"] != b["n_confirmed"]
        or a["pred_times"] != b["pred_times"],
        "threshold_differs": a["thresholds"] != b["thresholds"],
        "note": (
            "Path B here still constructs AdaptiveEventDetector which READS "
            "learning.json; difference isolates CFR+determinism+deterministic flag, "
            "not the learning read path (that is §26.2)."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
