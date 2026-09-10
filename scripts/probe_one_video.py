#!/usr/bin/env python3
import os, sys, time
import cv2, numpy as np
import supervision as sv
import trackers

VIDEO = '/tmp/IMG_5302.MOV'
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit('cannot open ' + VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames', total, 'fps', round(cap.get(cv2.CAP_PROP_FPS),2),
      'size', int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'x', int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

tik_p = trackers.ByteTrackTracker()
tik_o = trackers.OCSORTTracker()
tik_b = trackers.BoTSORTTracker()

sys.path.insert(0, '/app')
from inference.tracking.bytetrack_tracker import BytetrackTracker
tk_prod = BytetrackTracker()
tk_prod.load()

def fresh():
    return sv.Detections(
        xyxy=np.array([[400, 300, 500, 600]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        tracker_id=np.array([1], dtype=np.int32),
    )

# warm
for _ in range(3):
    ok, f = cap.read()
    if not ok:
        break
    for t in (tik_p, tik_o, tik_b, tk_prod):
        try:
            t.update(fresh())
        except Exception as e:
            print('warm err', type(t).__name__, repr(e))
            break

t0 = time.perf_counter()
n = 0
for f in iter(lambda: cap.read()[1], None):
    if f is None:
        break
    n += 1
    for t in (tik_p, tik_o, tik_b, tk_prod):
        try:
            t.update(fresh())
        except Exception as e:
            print('run err', type(t).__name__, repr(e))
    if n >= 20:
        break

elapsed = time.perf_counter() - t0
cap.release()
print('frames', n, 'wall_sec', round(elapsed, 3),
      'fps', round(n / elapsed if elapsed else 0, 2))
