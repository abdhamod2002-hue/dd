"""Phase 2 — cost context.

Phase 2 adds NO model inference: MoveNet already runs on every analysis tick
(``run_pose = pipe.should_analyze(...)`` in scripts/run_pipeline.py), and the
person track already carries keypoints into the pipeline. What Phase 2 adds is
pure Python arithmetic inside the event-decision layer.

So the honest way to state the cost is as a fraction of the REAL per-analysis
tick budget. This script measures that budget on a real clip.

Run with the project venv:  .venv\\Scripts\\python.exe scripts/phase2_cost.py
"""
from __future__ import annotations

import os
import statistics as st
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

REPO = Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

VIDEO = Path(r"D:\W\IMG_5115.MOV")
N_SAMPLE = 12


def main() -> int:
    import cv2  # noqa: E402
    from ultralytics import YOLO  # noqa: E402
    from inference.pose.movenet_pose import MovenetPose  # noqa: E402

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"cannot open {VIDEO}")
        return 1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"{VIDEO.name}: {total} frames")

    idxs = [int(i * max(1, total - 1) / max(1, N_SAMPLE - 1)) for i in range(N_SAMPLE)]
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            frames.append((i, f))
    cap.release()
    print(f"sampled {len(frames)} frames: {[i for i, _ in frames]}\n")

    print("loading YOLOv8n ...")
    yolo = YOLO(str(REPO / "yolov8n.pt"))
    print("loading MoveNet ...")
    pose = MovenetPose()
    pose.load()

    # warm up (first inference always carries lazy-init cost)
    _ = yolo.predict(frames[0][1], verbose=False, device="cpu", classes=[0])
    dummy = [(0, 0, 100, 100)]
    try:
        pose.estimate(frames[0][1], dummy)
    except Exception:
        pass

    yolo_ms, pose_ms, n_persons = [], [], []
    for i, f in frames:
        t0 = time.perf_counter()
        res = yolo.predict(f, verbose=False, device="cpu", classes=[0])[0]
        yolo_ms.append((time.perf_counter() - t0) * 1000)

        boxes = []
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes.xyxy.cpu().numpy():
                boxes.append(tuple(int(v) for v in b[:4]))
        n_persons.append(len(boxes))
        if boxes:
            t0 = time.perf_counter()
            pose.estimate(f, boxes)
            pose_ms.append((time.perf_counter() - t0) * 1000)

    def s(xs, unit="ms"):
        if not xs:
            return f"n/a (0 samples)"
        return (f"mean={st.mean(xs):.1f}{unit}  median={st.median(xs):.1f}{unit}  "
                f"min={min(xs):.1f}  max={max(xs):.1f}")

    print("\n" + "=" * 78)
    print("PER-ANALYSIS-TICK COST (real frames, CPU, IMG_5115 1920x1080-ish)")
    print("=" * 78)
    print(f"  YOLOv8n (person class only) : {s(yolo_ms)}")
    print(f"  MoveNet (all persons)       : {s(pose_ms)}  "
          f"(persons/frame: mean={st.mean(n_persons):.1f}, max={max(n_persons)})")

    per_tick = st.mean(yolo_ms) + (st.mean(pose_ms) if pose_ms else 0.0)
    print(f"  ---- measured subtotal      : {per_tick:.1f} ms/tick")
    print("  (ByteTrack + HSV fallback + buffering are NOT included;")
    print("   so this subtotal UNDERSTATES the true per-tick cost.)")

    # Phase-2 delta, measured by scripts/phase2_final.py:
    #   +7.1 ms total across 269 analysis ticks (all 5 videos)
    delta_total_ms = 7.1
    n_ticks = 269
    delta_per_tick = delta_total_ms / n_ticks
    print(f"\n  Phase-2 added cost          : {delta_total_ms:.1f} ms / {n_ticks} ticks "
          f"= {delta_per_tick:.4f} ms/tick")
    print(f"  as a share of the subtotal  : {100*delta_per_tick/per_tick:.3f}%")
    print(f"  as a share of one 8fps tick : {100*delta_per_tick/125.0:.4f}%  "
          f"(analysis_fps=8 -> 125 ms budget/tick)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
