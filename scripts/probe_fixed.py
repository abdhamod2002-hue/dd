#!/usr/bin/env python3
"""Probe: 20 identical-detections frames through 4 trackers. No YOLO, synthetic dets."""
import os, sys, time
import cv2, numpy as np
import supervision as sv
import trackers

sys.path.insert(0, "/app")
from inference.tracking.bytetrack_tracker import BytetrackTracker

VIDEO = "/tmp/IMG_5302.MOV"
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("cannot open " + VIDEO)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"VIDEO {total} frames {fps:.2f} fps {w}x{h}")

# Init all four trackers
tp = trackers.ByteTrackTracker()
to = trackers.OCSORTTracker()
tb = trackers.BoTSORTTracker()
prod = BytetrackTracker()
prod.load()

def make_det():
    # One fake person bbox at a fixed location, stable id=1
    xyxy = np.array([[400, 300, 500, 600]], dtype=np.float32)
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        tracker_id=np.array([1], dtype=np.int32),
    )

# Warm-up: skip 3 frames, run trackers
for _ in range(3):
    ok, frame = cap.read()
    if not ok:
        break
    for t in (tp, to, tb, prod):
        try:
            if isinstance(t, BytetrackTracker):
                t.update(make_det(), frame_index=0)
            else:
                t.update(make_det())
        except Exception as e:
            print(f"warm err {type(t).__name__}: {e!r}")

# Timed run: 20 frames
t0 = time.perf_counter()
n = 0
for frame in iter(lambda: cap.read()[1], None):
    if frame is None:
        break
    n += 1
    d = make_det()
    for t in (tp, to, tb, prod):
        try:
            if isinstance(t, BytetrackTracker):
                t.update(d, frame_index=n)
            else:
                t.update(d)
        except Exception as e:
            print(f"run err {type(t).__name__}: {e!r}")
    if n >= 20:
        break

cap.release()
elapsed = time.perf_counter() - t0
print(f"done {n} frames, wall={elapsed:.3f}s, fps={n/elapsed:.2f}" if elapsed else f"done {n} frames")
