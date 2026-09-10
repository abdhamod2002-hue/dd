#!/usr/bin/env python3
"""LAYER 1 EVALUATION HARNESS (measurement only — no production changes).

Runs every detection candidate on the SAME sampled real frames from D:\\22
and records, per detection: class, confidence, bbox, person-IoU, area
fraction. Also measures per-model latency (warm).

Candidates:
  person      : yolov8n.pt COCO (shared person boxes for person-IoU)
  best        : inference/detection/weights/best.pt     (legacy litter, documented broken)
  taco        : inference/detection/weights/taco_transfer_v1.pt (TACO transfer v1)
  world       : yolov8s-world.pt + waste text prompts (open-vocabulary, zero-training)
  hsv         : production ColorBagTracker (current fallback source)

Windows are chosen from the Phase-1 identity audits:
  positives  : IMG_5306 confirmed-event windows (yellow-bag actions on record)
  red-bag    : IMG_5305 4100-4360 (real red bag put on the ground ~f4235-4252;
               also the window where HSV anchored the arc to an off-white shirt)
  negatives  : IMG_5295/IMG_5304 (no people), plus sparse background samples

Output: layer1_eval/eval_frames.jsonl (raw), and a printed summary.
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")

OUT_DIR = "layer1_eval"
os.makedirs(OUT_DIR, exist_ok=True)
RAW = os.path.join(OUT_DIR, "eval_frames.jsonl")

WINDOWS = {
    # video: list of (start, end, stride)
    "IMG_5305": [(4100, 4360, 8), (0, 4000, 80)],
    "IMG_5306": [(380, 480, 8), (640, 740, 8), (2400, 2520, 8), (3980, 4180, 8), (4720, 4830, 8)],
    "IMG_5299": [(0, 1909, 40)],
    "IMG_5302": [(1200, 1400, 8), (1680, 1796, 8)],
    "IMG_5295": [(0, 3200, 60)],
    "IMG_5304": [(0, 2900, 60)],
}

CONF = {"best": 0.25, "taco": 0.25, "world": 0.20, "person": 0.40}
WORLD_PROMPTS = ["plastic bag", "garbage bag", "trash bag", "bottle", "cup", "can"]

PERSON_IOU_WRAP = 0.40  # >= this => box wraps a person (localization unusable for association)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> None:
    from ultralytics import YOLO
    from inference.detection.color_bag_detector import ColorBagTracker

    t0 = time.time()
    person_model = YOLO("yolov8n.pt")
    best_model = YOLO("inference/detection/weights/best.pt")
    taco_model = YOLO("inference/detection/weights/taco_transfer_v1.pt")
    world_model = YOLO("yolov8s-world.pt")
    world_model.set_classes(WORLD_PROMPTS)
    hsv = ColorBagTracker()
    print(f"models loaded in {time.time() - t0:.1f}s", flush=True)

    # warmup (JIT/graph init) — excluded from latency stats
    dummy = np.full((1920, 1080, 3), 128, np.uint8)
    for m in (person_model, best_model, taco_model, world_model):
        m.predict(dummy, conf=0.01, imgsz=640, verbose=False, device="cpu")
    hsv.update(dummy, frame_index=1, person_boxes=[])

    raw_fp = open(RAW, "w", encoding="utf-8")
    lat = {k: [] for k in ("person", "best", "taco", "world", "hsv")}

    def predict(model, frame, conf, tag, video, fidx, person_boxes, class_map=None):
        t = time.time()
        res = model.predict(frame, conf=conf, imgsz=640, verbose=False, device="cpu")[0]
        lat[tag].append((time.time() - t) * 1000.0)
        rows = []
        for i in range(len(res.boxes)):
            x1, y1, x2, y2 = map(float, res.boxes.xyxy[i].tolist())
            c = float(res.boxes.conf[i])
            cid = int(res.boxes.cls[i])
            name = (class_map or res.names).get(cid, str(cid)) if isinstance(res.names, dict) else res.names[cid]
            pi = max((iou((x1, y1, x2, y2), pb) for pb in person_boxes), default=0.0)
            area = (x2 - x1) * (y2 - y1)
            rows.append({
                "video": video, "frame": fidx, "model": tag, "class": str(name),
                "conf": round(c, 4), "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                "person_iou": round(pi, 3),
                "wraps_person": bool(pi >= PERSON_IOU_WRAP),
                "area_frac": round(area / (1080.0 * 1920.0), 5),
            })
        return rows

    for video, windows in WINDOWS.items():
        path = rf"D:\22\{video}.MOV"
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"!! cannot open {video}")
            continue
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        want = set()
        for a, b, s in windows:
            want.update(range(a, min(b, total), s))
        hsv.reset()
        print(f"== {video}: {len(want)} sampled frames of {total}", flush=True)
        fidx = -1
        while True:
            ok, frame = cap.read()
            fidx += 1
            if not ok or fidx > max(w[1] for w in windows):
                break
            if fidx not in want:
                continue
            # persons shared across candidates (production person source)
            t = time.time()
            pres = person_model.predict(frame, conf=CONF["person"], imgsz=640, verbose=False, device="cpu")[0]
            lat["person"].append((time.time() - t) * 1000.0)
            person_boxes = [tuple(map(float, pres.boxes.xyxy[i].tolist())) for i in range(len(pres.boxes))]

            rows = []
            rows += predict(person_model, frame, CONF["person"], "person", video, fidx, person_boxes) if False else []
            # best.pt
            rows += predict(best_model, frame, CONF["best"], "best", video, fidx, person_boxes)
            # taco
            rows += predict(taco_model, frame, CONF["taco"], "taco", video, fidx, person_boxes)
            # world (class ids are prompt indices after set_classes)
            world_names = {i: p for i, p in enumerate(WORLD_PROMPTS)}
            rows += predict(world_model, frame, CONF["world"], "world", video, fidx, person_boxes, class_map=world_names)
            # hsv (temporal tracker; feed person boxes like production)
            t = time.time()
            hdets = hsv.update(frame, frame_index=fidx, person_boxes=person_boxes)
            lat["hsv"].append((time.time() - t) * 1000.0)
            for d in hdets:
                x1, y1, x2, y2 = d.bbox
                pi = max((iou(d.bbox, pb) for pb in person_boxes), default=0.0)
                rows.append({
                    "video": video, "frame": fidx, "model": "hsv", "class": d.class_name,
                    "conf": round(float(d.confidence), 4),
                    "bbox": [round(v, 1) for v in d.bbox],
                    "person_iou": round(pi, 3), "wraps_person": bool(pi >= PERSON_IOU_WRAP),
                    "area_frac": round((x2 - x1) * (y2 - y1) / (1080.0 * 1920.0), 5),
                })
            for r in rows:
                raw_fp.write(json.dumps(r) + "\n")
        raw_fp.flush()
        cap.release()

    raw_fp.close()

    # ---------------- summary ----------------
    import collections
    per = collections.defaultdict(lambda: {"n": 0, "wrap": 0})
    per_vid_class = collections.defaultdict(lambda: collections.Counter())
    with open(RAW, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["class"])
            per[key]["n"] += 1
            per[key]["wrap"] += int(r["wraps_person"])
            per_vid_class[(r["video"], r["model"])][r["class"]] += 1

    print("\n=== detections by model/class (person-IoU>=0.4 counts as wraps-person) ===")
    for (m, c), v in sorted(per.items()):
        print(f"{m:<7} {c:<20} n={v['n']:<5} wraps_person={v['wrap']:<5} ({v['wrap'] / max(1, v['n']) * 100:.0f}%)")

    print("\n=== per-video per-model counts ===")
    for (vid, m), cnt in sorted(per_vid_class.items()):
        print(f"{vid:<9} {m:<7} {dict(cnt)}")

    print("\n=== latency (ms/frame, warm, CPU) ===")
    for k, v in lat.items():
        if v:
            a = np.array(v)
            print(f"{k:<7} mean={a.mean():7.1f}  p95={np.percentile(a, 95):7.1f}  n={len(a)}")
    print(f"\ntotal wall: {time.time() - t0:.0f}s -> {RAW}")


if __name__ == "__main__":
    main()
