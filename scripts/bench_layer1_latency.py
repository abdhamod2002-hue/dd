"""Latency benchmark: Layer-1 per-tick cost on real D:\22 frames (CPU).

Measures person model, semantic bag model, and HSV fallback separately
(average + p95) on sampled real frames from IMG_5305.
"""
import os
import sys
import time

sys.path.insert(0, r"D:\HO")
os.chdir(r"D:\HO")

import cv2
from ultralytics import YOLO
from inference.detection.color_bag_detector import ColorBagTracker

person = YOLO("yolov8n.pt")
bag = YOLO(r"D:\HO\inference\detection\weights\garbage_bag_v2.pt")

cap = cv2.VideoCapture(r"D:\22\IMG_5305.MOV")
frames = []
fi = -1
while len(frames) < 30:
    ok, frame = cap.read()
    fi += 1
    if not ok:
        break
    if fi % 150 == 0:
        frames.append(frame)
cap.release()
print("frames:", len(frames))

def bench(model, imgsz, name):
    ts = []
    for f in frames:
        t0 = time.perf_counter()
        model(f, imgsz=imgsz, device="cpu", conf=0.25, verbose=False)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    print(f"{name}: avg={sum(ts)/len(ts):.1f}ms p50={ts[len(ts)//2]:.1f}ms p95={ts[int(len(ts)*0.95)]:.1f}ms")

bench(person, 640, "person yolov8n-640")
bench(bag, 640, "bag garbage_bag_v2-640")

trk = ColorBagTracker()
ts = []
for f in frames:
    t0 = time.perf_counter()
    trk._detect_raw(f)
    ts.append((time.perf_counter() - t0) * 1000)
ts.sort()
print(f"hsv _detect_raw: avg={sum(ts)/len(ts):.1f}ms p95={ts[int(len(ts)*0.95)]:.1f}ms")
print("BENCH_DONE")
