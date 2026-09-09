#!/usr/bin/env python3
"""
Evaluate the pipeline against the test dataset.

Usage:
    python scripts/evaluate.py --dataset path/to/dataset --report evaluation_report.json

Runs each clip through the pipeline (via VideoFileSource), records
ClipResults, and writes a per-scenario + aggregate evaluation report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

from evaluation.metrics import ClipResult, evaluate
from inference.capture.camera_source import VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pipeline import InferencePipeline, PipelineConfig
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from scripts.run_pipeline import build_tracks_real


def run_clip(path: str, pipe: InferencePipeline, detector, movenet, tracker_ns) -> tuple:
    """Run one clip; return (confirmed_bool, latency, fps)."""
    src = VideoFileSource(path)
    if not src.open():
        return False, 0.0, 0.0
    t0 = time.time()
    n = 0
    confirmed = False
    latency = 0.0
    confirm_ts = None
    for pkt in src:
        # REAL ByteTrack path (stable ids across the clip)
        tracked = detector.track(pkt.frame, persist=True)
        run_pose = pipe.should_analyze(pkt.timestamp)
        persons, objects = build_tracks_real(pkt.frame, tracked, movenet, tracker_ns, n, run_pose=run_pose)
        events = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
        if events and not confirmed:
            confirmed = True
            confirm_ts = pkt.timestamp
            latency = pkt.timestamp - events[0].event_timestamp
        n += 1
    # Flush any detector state that only confirms at end-of-clip.
    final_events = pipe.finalize()
    if final_events and not confirmed:
        confirmed = True
        latency = 0.0
    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else 0.0
    src.release()
    return confirmed, latency, fps


def main():
    ap = argparse.ArgumentParser(description="Evaluate pipeline on test dataset")
    ap.add_argument("--dataset", required=True, help="dataset root (with littering/ and normal/)")
    ap.add_argument("--report", default="evaluation_report.json")
    args = ap.parse_args()

    detector = YoloDetector(); detector.load()
    movenet = MovenetPose(); movenet.load()
    tracker_ns = BytetrackTracker(); tracker_ns.load()

    results = []
    for split in ("littering", "normal"):
        split_dir = os.path.join(args.dataset, split)
        if not os.path.isdir(split_dir):
            continue
        for fname in sorted(os.listdir(split_dir)):
            if not fname.endswith(".mp4"):
                continue
            clip_id = os.path.splitext(fname)[0]
            ann_path = os.path.join(split_dir, clip_id + ".json")
            if not os.path.exists(ann_path):
                continue
            with open(ann_path) as f:
                ann = json.load(f)
            pipe = InferencePipeline(PipelineConfig(camera_id="eval"))
            confirmed, latency, fps = run_clip(
                os.path.join(split_dir, fname), pipe, detector, movenet, tracker_ns)
            results.append(ClipResult(
                clip_id=clip_id,
                scenario=ann["scenario"],
                ground_truth=ann["ground_truth"],
                predicted=confirmed,
                latency_seconds=latency,
                fps=fps,
            ))
            print(f"{clip_id}: GT={ann['ground_truth']} pred={confirmed} "
                  f"scenario={ann['scenario']} fps={fps:.1f}")

    report = evaluate(results)
    print()
    print(report.summary_str())
    report.to_json(args.report)
    print(f"\nReport written to {args.report}")


if __name__ == "__main__":
    main()
