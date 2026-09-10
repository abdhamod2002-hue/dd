#!/usr/bin/env python3
"""
Phase 3 Step 3 (+Step 4 input) — pairwise duplicate-pattern analysis.

Reads phase3_runs/raw_detections/*_raw.json, pools ALL raw boxes per frame
(all models, as the tracker input effectively does), and for every pair
computes: IoU, containment ratios, center distance, area ratio, class
relationship, model sources. Classifies each pair:

  SAME_PHYSICAL_OBJECT : same-model, IoU>=0.5 or containment>=0.7
  LIKELY_DUPLICATE     : cross-model same-group, IoU>=0.5 or containment>=0.7
  LEGITIMATE_OVERLAP   : overlapping but below thresholds, or person-vs-object
  DISTINCT             : no overlap

Also simulates the CURRENT _dedup_contained() twice:
  (a) per-model (what production track() actually does), and
  (b) pooled cross-model (what it would do if it saw everything),
to demonstrate the Step-4 failure mode with real numbers.

Output: phase3_runs/dup_analysis/<VIDEO>_pairs.json + SUMMARY.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "phase3_runs" / "raw_detections"
OUT_DIR = PROJECT_ROOT / "phase3_runs" / "dup_analysis"

IOU_DUP = 0.5
CONTAIN_DUP = 0.7


def geom(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    ab = max(1e-6, (bx2 - bx1) * (by2 - by1))
    union = aa + ab - inter
    iou = inter / union if union > 0 else 0.0
    return {
        "iou": round(iou, 4),
        "contain_a_in_b": round(inter / aa, 4),
        "contain_b_in_a": round(inter / ab, 4),
        "area_ratio": round(min(aa, ab) / max(aa, ab), 4),
        "center_dist": round(
            (((ax1 + ax2) / 2 - (bx1 + bx2) / 2) ** 2
             + ((ay1 + ay2) / 2 - (by1 + by2) / 2) ** 2) ** 0.5, 1),
    }


def current_dedup_contained(dets, contain_thresh=0.7):
    """Faithful re-implementation of YoloDetector._dedup_contained for simulation."""
    kept = []
    for d in sorted(dets, key=lambda t: t["confidence"], reverse=True):
        x1, y1, x2, y2 = d["bbox"]
        area = max(1e-6, (x2 - x1) * (y2 - y1))
        contained = False
        for k in kept:
            if k["is_person"] != d["is_person"]:
                continue
            kx1, ky1, kx2, ky2 = k["bbox"]
            ix = max(0.0, min(x2, kx2) - max(x1, kx1))
            iy = max(0.0, min(y2, ky2) - max(y1, ky1))
            if (ix * iy) / area >= contain_thresh:
                contained = True
                break
        if not contained:
            kept.append(d)
    return kept


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    grand = {"frames": 0, "pairs": 0, "by_class": {},
             "per_model_removed": 0, "pooled_would_remove": 0,
             "cross_model_dup_pairs": 0, "same_model_dup_pairs": 0}
    for rp in sorted(RAW_DIR.glob("*_raw.json")):
        payload = json.loads(rp.read_text(encoding="utf-8"))
        video_out = {"video": payload["video"], "frames": []}
        for fr in payload["frames"]:
            pooled = []
            for src_name, boxes in fr["sources"].items():
                for b in boxes:
                    # Mirror detect(): person model contributes persons only when
                    # the litter model exists (non-person COCO dropped by design).
                    if src_name == "person_model" and not b["is_person"]:
                        continue
                    pooled.append({**b, "src": src_name})
            pairs = []
            for i, j in combinations(range(len(pooled)), 2):
                A, B = pooled[i], pooled[j]
                g = geom(A["bbox"], B["bbox"])
                overlap = g["iou"] > 0.02 or max(
                    g["contain_a_in_b"], g["contain_b_in_a"]) > 0.05
                if A["is_person"] != B["is_person"]:
                    cls = "LEGITIMATE_OVERLAP"  # person vs object: never duplicates
                elif not overlap:
                    cls = "DISTINCT"
                elif g["iou"] >= IOU_DUP or max(
                        g["contain_a_in_b"], g["contain_b_in_a"]) >= CONTAIN_DUP:
                    if A["src"] == B["src"]:
                        cls = "SAME_PHYSICAL_OBJECT"
                        grand["same_model_dup_pairs"] += 1
                    else:
                        cls = "LIKELY_DUPLICATE"
                        grand["cross_model_dup_pairs"] += 1
                else:
                    cls = "LEGITIMATE_OVERLAP"
                grand["pairs"] += 1
                grand["by_class"][cls] = grand["by_class"].get(cls, 0) + 1
                if cls in ("SAME_PHYSICAL_OBJECT", "LIKELY_DUPLICATE"):
                    pairs.append({
                        "a": {k: A[k] for k in ("src", "model", "class", "confidence", "bbox")},
                        "b": {k: B[k] for k in ("src", "model", "class", "confidence", "bbox")},
                        "metrics": g, "verdict": cls,
                    })
            # Current-dedup simulation: per-model vs pooled.
            removed_per_model, removed_pooled = 0, 0
            for src_name, boxes in fr["sources"].items():
                sub = [b for b in pooled if b["src"] == src_name]
                removed_per_model += len(sub) - len(current_dedup_contained(sub))
            removed_pooled = len(pooled) - len(current_dedup_contained(pooled))
            grand["per_model_removed"] += removed_per_model
            grand["pooled_would_remove"] += removed_pooled
            grand["frames"] += 1
            video_out["frames"].append({
                "src_frame": fr["src_frame"],
                "pooled_count": len(pooled),
                "duplicate_pairs": pairs,
                "current_dedup_removed_per_model": removed_per_model,
                "current_dedup_would_remove_pooled": removed_pooled,
            })
        op = OUT_DIR / f"{Path(payload['video']).stem}_pairs.json"
        op.write_text(json.dumps(video_out, indent=1), encoding="utf-8")
        print(f"wrote {op}", flush=True)
    sp = OUT_DIR / "SUMMARY.json"
    sp.write_text(json.dumps(grand, indent=2), encoding="utf-8")
    print(json.dumps(grand, indent=2), flush=True)
    print(f"wrote {sp}", flush=True)


if __name__ == "__main__":
    main()
