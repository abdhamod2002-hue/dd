"""Runtime verification for the Dockerized backend container.

Checks (inside the backend container, run via `docker compose exec backend python /app/verify_runtime.py`):
  1. Key Python packages import.
  2. Model weight files exist (yolov8n.pt, best.pt, MoveNet cache).
  3. D:\\W is bind-mounted at /data/W and real files are readable.
  4. A real D:\\W image opens via OpenCV.
  5. A real D:\\W MOV opens via OpenCV (frame count + first frame).
Reports OK / FAIL per check. Never fakes success.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def check(name: str, fn) -> None:
    try:
        fn()
        print(f"[OK]   {name}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")


def imports_() -> None:
    import cv2  # noqa: F401
    import numpy  # noqa: F401
    import tensorflow as tf  # noqa: F401
    import tensorflow_hub  # noqa: F401
    import torch  # noqa: F401
    from ultralytics import YOLO  # noqa: F401
    from inference.pose.movenet_pose import MovenetPose  # noqa: F401
    from inference.detection.yolo_detector import YoloDetector  # noqa: F401
    from inference.capture.camera_source import VideoFileSource  # noqa: F401


def torches_cpu() -> None:
    import torch
    print(f"      torch={torch.__version__} cuda_available={torch.cuda.is_available()}")


def weights_() -> None:
    for p in ["yolov8n.pt", "inference/detection/weights/best.pt"]:
        fp = Path(p)
        if not fp.exists():
            raise FileNotFoundError(f"{p} missing")
        if fp.stat().st_size <= 0:
            raise RuntimeError(f"{p} is empty")


def movenet_() -> None:
    from inference.pose.movenet_pose import MovenetPose
    m = MovenetPose()
    m.load()
    if not m.is_loaded:
        raise RuntimeError("MoveNet is_loaded False")


def mount_() -> None:
    d = Path("/data/W")
    if not d.is_dir():
        raise FileNotFoundError("/data/W not mounted")
    files = sorted(d.iterdir())
    print(f"      /data/W contains {len(files)} entries: {[f.name for f in files]}")
    if not files:
        raise RuntimeError("/data/W is empty")


def open_image_() -> None:
    import cv2
    caps = list(Path("/data/W").glob("*.JPG")) + list(Path("/data/W").glob("*.jpg"))
    if not caps:
        raise FileNotFoundError("no JPG in /data/W")
    img = cv2.imread(str(caps[0]))
    if img is None:
        raise RuntimeError(f"cv2.imread failed for {caps[0].name}")
    print(f"      opened {caps[0].name} shape={img.shape}")


def open_video_() -> None:
    import cv2
    vids = [p for p in Path("/data/W").iterdir() if p.suffix.lower() == ".mov"]
    if not vids:
        raise FileNotFoundError("no MOV in /data/W")
    cap = cv2.VideoCapture(str(vids[0]))
    if not cap.isOpened():
        raise RuntimeError(f"cv2.VideoCapture could not open {vids[0].name}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("could not read first frame")
    print(f"      opened {vids[0].name} frames={n} first_frame_shape={frame.shape}")


def main() -> None:
    print("=== Runtime verification ===")
    check("imports (cv2/np/tf/tfhub/torch/ultralytics)", imports_)
    check("torch reports CPU", torches_cpu)
    check("model weights present", weights_)
    check("MoveNet loads from cache", movenet_)
    check("/data/W mounted", mount_)
    check("real image opens", open_image_)
    check("real MOV opens", open_video_)
    print("=== done ===")


if __name__ == "__main__":
    main()