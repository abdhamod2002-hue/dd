import sys
sys.path.insert(0, "/app")
from scripts.benchmark_trackers import run_one_video
from inference.detection.yolo_detector import YoloDetector

det = YoloDetector()
det.load()
print("Detector loaded")
res = run_one_video("/tmp/IMG_5302.MOV", det)
print("RESULT:", res)
import json, os
print(json.dumps(res, indent=2))
