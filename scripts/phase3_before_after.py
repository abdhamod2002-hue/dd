#!/usr/bin/env python3
"""
Phase 3 Step 9 — real BEFORE/AFTER counts + visual proof (ANALYSIS ONLY).

Uses the saved raw captures (no YOLO rerun) and the REAL new dedup code
(inference.detection.yolo_detector.deduplicate_detections) to compute,
per frame: raw pooled detections, duplicate candidates, removed count,
tracker-input-after count.

Saves before/after JPGs for selected frames:
  before_raw.jpg / after_raw.jpg (detector output boxes)
  before_tracker_input.jpg / after_tracker_input.jpg (what the tracker sees)

Output: phase3_runs/before_after/*.jpg + BEFORE_AFTER.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference.detection.yolo_detector import Detection, deduplicate_detections  # noqa: E402

DATA_ROOT = Path(r"D:\22")
RAW_DIR = PROJECT_ROOT / "phase3_runs" / "raw_detections"
OUT_DIR = PROJECT_ROOT / "phase3_runs" / "before_after"

# (video, src_frame) selections: one proven duplicate frame + one
# legitimate-overlap frame per video where available.
SELECTED = {
    "IMG_5306.MOV": [4816, 6],
    "IMG_5305.MOV": [847, 4235],
    "IMG_5301.MOV": [1408, 1320],
}

MODEL_COLOR = {"person_model": (255, 220, 80), "litter_model": (80, 255, 120),
               "bag_model": (100, 100, 255)}


def pooled_detections(fr):
    """Mirror YoloDetector.detect() filtering (drop non-person COCO from the
    person model when the litter model exists), keeping model attribution."""
    out = []
    for src_name, boxes in fr["sources"].items():
        for b in boxes:
            if src_name == "person_model" and not b["is_person"]:
                continue
            cx, cy = b["centroid"]
            out.append((src_name, Detection(
                b["class"], b["confidence"], tuple(b["bbox"]), (cx, cy))))
    return out


def draw_boxes(img, items, title):
    canvas = img.copy()
    for src_name, d in items:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        color = MODEL_COLOR.get(src_name, (255, 255, 255))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{d.class_name} {d.confidence:.2f} [{src_name}]",
                    (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2)
    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255, 255, 255), 2)
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for rp in sorted(RAW_DIR.glob("*_raw.json")):
        payload = json.loads(rp.read_text(encoding="utf-8"))
        vname = payload["video"]
        tot_raw = tot_removed = tot_after = 0
        tot_persons = tot_waste = 0
        frames_with_dup = 0
        by_src = {fr["src_frame"]: fr for fr in payload["frames"]}
        for fr in payload["frames"]:
            pooled = pooled_detections(fr)
            dets = [d for _, d in pooled]
            after = deduplicate_detections(dets)
            removed = len(dets) - len(after)
            tot_raw += len(dets)
            tot_removed += removed
            tot_after += len(after)
            tot_persons += sum(1 for d in after if d.is_person)
            tot_waste += sum(1 for d in after if not d.is_person)
            if removed:
                frames_with_dup += 1
        table.append({"video": vname, "frames": len(payload["frames"]),
                      "raw": tot_raw, "removed": tot_removed,
                      "after": tot_after, "frames_with_dup": frames_with_dup,
                      "persons_after": tot_persons, "waste_after": tot_waste})

        # Visual proof on selected frames.
        cap = cv2.VideoCapture(str(DATA_ROOT / vname))
        for src in SELECTED.get(vname, []):
            if src not in by_src:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, src)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            pooled = pooled_detections(by_src[src])
            dets = [d for _, d in pooled]
            after = deduplicate_detections(dets)
            after_set = set(map(id, after))
            after_items = [(s, d) for s, d in pooled if id(d) in after_set]
            stem = Path(vname).stem
            cv2.imwrite(str(OUT_DIR / f"{stem}_src{src}_before_raw.jpg"),
                        draw_boxes(frame, pooled,
                                   f"BEFORE raw src={src} n={len(pooled)}"))
            cv2.imwrite(str(OUT_DIR / f"{stem}_src{src}_after_raw.jpg"),
                        draw_boxes(frame, after_items,
                                   f"AFTER dedup src={src} n={len(after)}"))
            # Tracker-input views: same boxes, tracker-style labels.
            cv2.imwrite(
                str(OUT_DIR / f"{stem}_src{src}_before_tracker_input.jpg"),
                draw_boxes(frame, pooled,
                           f"TRACKER INPUT BEFORE src={src} n={len(pooled)}"))
            cv2.imwrite(
                str(OUT_DIR / f"{stem}_src{src}_after_tracker_input.jpg"),
                draw_boxes(frame, after_items,
                           f"TRACKER INPUT AFTER src={src} n={len(after)}"))
            print(f"  {vname} src={src}: raw={len(pooled)} "
                  f"removed={len(pooled) - len(after)} "
                  f"after={len(after)}", flush=True)
        cap.release()
    (OUT_DIR / "BEFORE_AFTER.json").write_text(json.dumps(table, indent=2),
                                               encoding="utf-8")
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
