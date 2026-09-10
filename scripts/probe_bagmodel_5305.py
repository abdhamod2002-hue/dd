"""Probe garbage_bag_v2 directly on IMG_5305 red-bag frames."""
import os
import sys

sys.path.insert(0, r"D:\HO")
os.chdir(r"D:\HO")

import cv2
from ultralytics import YOLO

M = YOLO(r"D:\HO\inference\detection\weights\garbage_bag_v2.pt")
cap = cv2.VideoCapture(r"D:\22\IMG_5305.MOV")
fps = cap.get(cv2.CAP_PROP_FPS) or 30

for imgsz in (640, 960):
    fi = -1
    best = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frame = cap.read()
        fi += 1
        if not ok:
            break
        if not (4380 <= fi <= 4560) or fi % 5:
            continue
        r = M(frame, imgsz=imgsz, device="cpu", conf=0.15, verbose=False)[0]
        for b in r.boxes:
            x1, y1, x2, y2 = [round(float(v)) for v in b.xyxy[0]]
            best.append((fi, x1, y1, x2, y2, round(float(b.conf[0]), 2)))
    best.sort(key=lambda t: -t[5])
    print(f"imgsz={imgsz}: top12 conf boxes:", best[:12])
cap.release()
print("PROBE_DONE")