"""Save annotated frames from IMG_5305 for visual verification."""
import os
import sys

sys.path.insert(0, r"D:\HO")
os.chdir(r"D:\HO")

import cv2
from ultralytics import YOLO

M = YOLO(r"D:\HO\inference\detection\weights\garbage_bag_v2.pt")
cap = cv2.VideoCapture(r"D:\22\IMG_5305.MOV")
targets = {4420, 4430, 4840, 4860}
os.makedirs(r"D:\HO\_l1_5305_reg", exist_ok=True)
fi = -1
while True:
    ok, frame = cap.read()
    fi += 1
    if not ok:
        break
    if fi not in targets:
        continue
    r = M(frame, imgsz=960, device="cpu", conf=0.25, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = [int(float(v)) for v in b.xyxy[0]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)
        cv2.putText(frame, f"bag {float(b.conf[0]):.2f}", (x1, max(40, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 255), 5)
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 3, h // 3))
    cv2.imwrite(rf"D:\HO\_l1_5305_reg\f{fi}.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
cap.release()
print("SAVED")