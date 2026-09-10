#!/usr/bin/env python3
"""
Phase 3 Step 2 — Capture RAW per-model detections (BENCHMARK/ANALYSIS ONLY).

For each sampled frame, runs every YOLO model call that YoloDetector itself
makes (same weights, same conf, same imgsz, same device) and records each raw
box with model/class/confidence/bbox/source — BEFORE any dedup or tracking.

Also records the aggregated YoloDetector.detect() output for comparison.

No production code is modified. Read-only use of the loaded models.

Output: phase3_runs/raw_detections/<VIDEO>_raw.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.detection.yolo_detector import YoloDetector  # noqa: E402

DATA_ROOT = Path(r"D:\22")
OUT_DIR = PROJECT_ROOT / "phase3_runs" / "raw_detections"

VIDEOS = [v.strip() for v in os.environ.get(
    "PHASE3_VIDEOS", "IMG_5306.MOV,IMG_5305.MOV,IMG_5301.MOV").split(",") if v.strip()]
FRAMES_PER_VIDEO = int(os.environ.get("PHASE3_FRAMES", "40"))
# Explicit frames to always include (Phase-2 accounting/visual frames, src idx).
EXTRA_FRAMES = {"IMG_5306.MOV": [6], "IMG_5305.MOV": [168, 2166], "IMG_5301.MOV": [0]}


def raw_model_boxes(model, frame, conf, device, imgsz, model_tag):
    """Replicate exactly the per-model inference loop in YoloDetector."""
    out = []
    if model is None:
        return out
    for r in model(frame, conf=conf, device=device, imgsz=imgsz, verbose=False):
        for box in r.boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            c = float(box.conf[0])
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            out.append({
                "model": model_tag,
                "class": name,
                "confidence": round(c, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "centroid": [round(cx, 1), round(cy, 1)],
                "is_person": name.lower() == "person",
            })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    det = YoloDetector()
    det.load()
    print(f"models: person={det.person_weights} litter={det.litter_weights} "
          f"bag={det.bag_weights}", flush=True)
    print(f"conf: person={det.person_conf} litter={det.litter_conf} bag={det.bag_conf} "
          f"imgsz={det.imgsz} device={det.device}", flush=True)

    for vname in VIDEOS:
        vp = DATA_ROOT / vname
        if not vp.exists():
            print(f"skip {vname}: not found", flush=True)
            continue
        cap = cv2.VideoCapture(str(vp))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, total // FRAMES_PER_VIDEO)
        wanted = set(range(0, total, stride))
        wanted.update(EXTRA_FRAMES.get(vname, []))
        wanted = sorted(i for i in wanted if i < total)[: FRAMES_PER_VIDEO + 8]
        print(f"\n=== {vname}: {total} frames, sampling {len(wanted)} "
              f"(stride {stride}) ===", flush=True)

        frames = []
        t0 = time.time()
        for fi in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rec = {"src_frame": fi, "sources": {}}
            rec["sources"]["person_model"] = raw_model_boxes(
                det._person_model, frame, det.person_conf, det.device, det.imgsz,
                "person_model(yolov8n)")
            if det._litter_model is not None:
                rec["sources"]["litter_model"] = raw_model_boxes(
                    det._litter_model, frame, det.litter_conf, det.device,
                    det.imgsz, "litter_model(best.pt)")
            else:
                rec["sources"]["litter_model"] = []
            if det._bag_model is not None:
                rec["sources"]["bag_model"] = raw_model_boxes(
                    det._bag_model, frame, det.bag_conf, det.device, det.imgsz,
                    "bag_model(garbage_bag_v2.pt)")
            else:
                rec["sources"]["bag_model"] = []
            # Aggregated detect() output for comparison (same frame).
            agg = det.detect(frame)
            rec["aggregated_detect"] = [{
                "class": d.class_name, "confidence": round(d.confidence, 4),
                "bbox": [round(v, 1) for v in d.bbox], "is_person": d.is_person,
            } for d in agg]
            n_raw = sum(len(v) for v in rec["sources"].values())
            print(f"  src={fi}: raw={n_raw} "
                  f"(person={len(rec['sources']['person_model'])} "
                  f"litter={len(rec['sources']['litter_model'])} "
                  f"bag={len(rec['sources']['bag_model'])}) "
                  f"aggregated={len(agg)}", flush=True)
            frames.append(rec)
        cap.release()

        payload = {
            "video": vname,
            "total_src_frames": total,
            "sampled_frames": len(frames),
            "detector_config": {
                "person_weights": det.person_weights, "person_conf": det.person_conf,
                "litter_weights": det.litter_weights, "litter_conf": det.litter_conf,
                "bag_weights": det.bag_weights, "bag_conf": det.bag_conf,
                "imgsz": det.imgsz, "device": det.device,
            },
            "frames": frames,
            "wall_sec": round(time.time() - t0, 1),
        }
        op = OUT_DIR / f"{Path(vname).stem}_raw.json"
        with open(op, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)
        print(f"wrote {op}", flush=True)


if __name__ == "__main__":
    main()
