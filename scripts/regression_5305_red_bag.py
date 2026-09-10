"""Step 25 regression: IMG_5305 red bag must get a semantic Layer-1 track."""
import os
import sys
from collections import defaultdict

sys.path.insert(0, r"D:\HO")
os.chdir(r"D:\HO")

from inference.detection.yolo_detector import YoloDetector
from inference.capture.camera_source import VideoFileSource

det = YoloDetector()
det.load()
src = VideoFileSource(r"D:\22\IMG_5305.MOV")
assert src.open()

semantic = defaultdict(list)
fi_seq = 0
while True:
    pkt = src.read()
    if pkt is None:
        break
    fi_seq += 1
    if fi_seq % 10:
        continue
    for t in det.track(pkt.frame, persist=True):
        if not t.is_person and t.source == "yolo":
            semantic[t.track_id].append((fi_seq, [round(v) for v in t.bbox], round(t.confidence, 2)))

src.release()
print("semantic Garbage-Bag tracks:", dict((k, len(v)) for k, v in semantic.items()))
for tid, seq in sorted(semantic.items()):
    spans = [(seq[0][0], seq[-1][0])]
    x = min(b[0] for _, b, _ in seq); y = min(b[1] for _, b, _ in seq)
    x2 = max(b[2] for _, b, _ in seq); y2 = max(b[3] for _, b, _ in seq)
    confs = [c for _, _, c in seq]
    print(f"track {tid}: n={len(seq)} frames={spans} bbox_range=({x:.0f},{y:.0f},{x2:.0f},{y2:.0f}) conf[{min(confs)},{max(confs)}]")
print("REGRESSION_DONE")