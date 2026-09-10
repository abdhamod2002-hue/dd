import sys, cv2, json
sys.path.insert(0, r"D:\HO")
import os
os.chdir(r"D:\HO")
from collections import Counter
from inference.detection.color_bag_detector import ColorBagTracker, ColorBagConfig

VID = r"D:\22\IMG_5305.MOV"
cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("video", VID, "frames", total, "fps", fps)

tracker = ColorBagTracker()
counts = Counter()
frames_by_class = {}
fi = -1
while True:
    ok, frame = cap.read()
    fi += 1
    if not ok:
        break
    if fi % 10:
        continue
    for centroid, bbox, conf, cls in tracker._detect_raw(frame):
        counts[cls] += 1
        frames_by_class.setdefault(cls, []).append((fi, round(fi/fps, 1), [round(v) for v in bbox], round(conf, 2)))

print(json.dumps({"raw_candidates_by_class": dict(counts)}, indent=1))
for cls in ("color_candidate_red", "color_candidate_yellow"):
    seq = frames_by_class.get(cls, [])
    print(cls, "first frames:", seq[:5], "... last:", seq[-3:] if seq else [])
