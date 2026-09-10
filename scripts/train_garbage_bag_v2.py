import os
import sys

sys.path.insert(0, r"D:\HO")
os.chdir(r"D:\HO")

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data=r"datasets/roboflow_colored_bags/Trash.v1i.yolov8/data.yaml",
    epochs=15,
    imgsz=640,
    batch=8,
    device="cpu",
    name="garbage_bag_v2_640",
    project=r"D:\HO\runs",
    exist_ok=True,
    workers=2,
    verbose=True,
)
print("TRAIN_DONE")
