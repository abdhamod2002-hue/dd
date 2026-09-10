"""Validate garbage_bag_v2 (semantic waste detector) on held-out real data.

1. Report ultralytics test-split metrics (precision/recall/mAP50/mAP50-95).
2. Copy best weights -> inference/detection/weights/garbage_bag_v2.pt (never
   overwrites best.pt, never touches the rejected waste_bag_real_v1.pt).
3. Record model provenance (absolute path, classes, sha256).
"""
import hashlib
import json
import os
import shutil
import sys

sys.path.insert(0, r"D:\HO")
os.chdir(r"D:\HO")

from ultralytics import YOLO

RUN_DIR = r"D:\HO\runs\garbage_bag_v2_640"
BEST = os.path.join(RUN_DIR, "weights", "best.pt")
DST = r"D:\HO\inference\detection\weights\garbage_bag_v2.pt"

model = YOLO(BEST)
metrics = model.val(data=r"datasets/roboflow_colored_bags/Trash.v1i.yolov8/data.yaml",
                    split="test", imgsz=416, device="cpu", verbose=False,
                    project=r"D:\HO\runs", name="garbage_bag_v2_eval", exist_ok=True)

report = {
    "model": DST,
    "source_run": RUN_DIR,
    "classes": list(model.names.values()),
    "test_split_metrics": {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    },
}

shutil.copy(BEST, DST)
h = hashlib.sha256()
with open(DST, "rb") as f:
    h.update(f.read())
report["sha256"] = h.hexdigest()
report["bytes"] = os.path.getsize(DST)

with open(r"D:\HO\GARBAGE_BAG_V2_EVAL.json", "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
print("EVAL_DONE")
