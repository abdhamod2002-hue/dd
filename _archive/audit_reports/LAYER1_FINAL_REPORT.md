# LAYER 1 FINAL REPORT — Production Fix: Detection + Tracking

Scope: Layer 1 only. Layer 2 (littering_event_detector.py), Evidence, and Dashboard
logic were NOT modified. Stopped per the mandatory stop condition.

## 1. Current Layer-1 architecture (from real code)

```
VIDEO → VideoFileSource (D:\22 .MOV)
  → YoloDetector.load()   [load-once]
      person model  yolov8n.pt (COCO person)
      litter model  best.pt (bottle, juice-cup, nescafe, plate, tissue)
      BAG MODEL     inference/detection/weights/garbage_bag_v2.pt  ← NEW (semantic, source="yolo")
  → YoloDetector.track()
      ByteTrack per model (independent ID spaces, persist=True)
      → HSV ColorBagTracker fallback ONLY when no object exists
        (honest classes: color_candidate_<hue>, source="color")
  → BytetrackTracker (Track store) → stable object UIDs (object_identity.py)
  → Layer 2 (unchanged)
```

## 2. Files changed (BEFORE → ROOT CAUSE → CHANGE → AFTER)

| File | Before | Root cause | Change |
|---|---|---|---|
| `inference/detection/color_bag_detector.py` | HSV classes named `yellow_waste_bag`, `green_waste`, … (semantic claims from pixel color); NO red range | Step 7 violation: yellow region ≠ waste; red waste invisible | Classes renamed to `color_candidate_<hue>` (yellow/green/blue/red/black/white); red range added (H 0–10 and 170–179, guarded min_hits=3, min_conf=0.28) |
| `inference/detection/yolo_detector.py` | `bag_weights=None` slot permanently disabled | No semantic waste detector existed | `_resolve_bag_weights()` added: env `WASTE_BAG_WEIGHTS` → `garbage_bag_v2.pt` if present → None. Rejected `waste_bag_real_v1.pt` unreachable under any path. Loaded ONCE in `load()` |
| `backend/Dockerfile` | Only `best.pt` copied explicitly | Model must load in-container, not host-only | `COPY inference/detection/weights/garbage_bag_v2.pt` added with sha256 comment |
| `scripts/train_garbage_bag_v2.py`, `scripts/eval_garbage_bag_v2.py` | — | Reproducible approved training/eval | New scripts (CPU, yolov8n, 15 epochs, imgsz 640) |
| `scripts/layer1_real_video_matrix.py`, `scripts/regression_5305_red_bag.py`, `scripts/probe_5305_hsv.py`, `scripts/probe_bagmodel_5305.py` | — | Real-data validation (Steps 24–26) | New validation scripts |

NOT changed: littering_event_detector.py, inference/tracking/*, pipeline event
paths, evidence, dashboard.

## 3. Dataset / annotation status

- `datasets/roboflow_colored_bags/Trash.v1i.yolov8`: **human-annotated**, CC BY 4.0,
  847 images, 1 class `Garbage Bag`, multi-color. Sole training source.
- `D:\22`: real deployment footage, **no ground-truth annotations** → real-video
  results are observational (GROUND TRUTH NOT AVAILABLE for D:\22).
- `real_train/dataset_v5` (auto-labeled): NOT used — the exact recipe that produced
  the rejected `waste_bag_real_v1.pt`. That model remains permanently rejected;
  never loaded or resolved by any code path.

## 4. Detector chosen and why

YOLOv8n fine-tuned on the human-annotated Roboflow colored-bags dataset
(reliable public human-labeled data; project data has no labels).

**Held-out test split (real metrics):** precision **0.967**, recall **0.967**,
mAP50 **0.978**, mAP50-95 **0.579**. (First attempt at imgsz=416: P=0.90,
R=0.97, mAP50=0.968; retrained at 640 after small-object misses were measured
on IMG_5305.)

Deployed: `D:\HO\inference\detection\weights\garbage_bag_v2.pt`,
sha256 `2f33b83654789ef273151db3a7c1ef89adb405d076e46ca9659bcdd68ba6a824`,
6,245,290 bytes, classes `['Garbage Bag']`. Auto-loaded by `YoloDetector.load()`
(env override: `WASTE_BAG_WEIGHTS`).

## 5. Real-video matrix (production path, every 15th source frame, CPU)

| Video | Persons | Semantic Waste (source=yolo) | HSV candidates (source=color) | Legacy best.pt | Raw Obj Track IDs | ID Switches | Potential False Detections |
|---|---:|---:|---:|---:|---:|---:|---|
| IMG_5305 | 3 | 26 ticks | 868 (red 351, white 190, black 176, yellow 151) | 0 | 23 | 0 | HSV candidates are unverified pixels (honestly labeled); probe found one car boxed as bag at conf 0.62 in a single frame (model limitation, reported) |
| IMG_5306 | 12 | 55 | 6859 (black 3004, white 1463, yellow 1199, red 1171, blue 22) | 0 | 49 | 0 | same caveat |
| IMG_5299 | 2 | 38 (+2 legacy "nescafe") | 701 | 2 | 39 | 0 | same caveat |
| IMG_5302 | 2 | 11 | 755 | 0 | 29 | 0 | same caveat |

Stable object UIDs: mechanism unchanged (object_identity.py); raw→stable tests
pass (`test_layer1_audit_models`, `test_layer2_identity`).

## 6. Critical tests

- **IMG_5305 red bag (Step 25):** the semantic model detects the dark bag being
  carried to the bin (conf 0.61 at imgsz 960) but does **NOT** detect the red bag
  lying on the ground, even at 960px. The red bag IS represented as an honest HSV
  candidate (`color_candidate_red`, 351 tracked ticks) — a proposal, never
  semantic waste. **Honest limitation.** Closing it requires human annotation of
  real D:\22 red-bag frames and a retrain.
- **Shirt false positive (Step 26):** impossible through the semantic path — HSV
  output can no longer carry `*_waste_*` names; a shirt can only be
  `color_candidate_*` (source="color"), treated as unconfirmed by Layer 2.
  Verified: no `yellow_waste_bag` string remains in detection output.


## 8. Layer-2 compatibility & full tests

- Full pytest suite AFTER changes: **1 failed, 138 passed** — the single failure
  (`test_bytetrack_emits_stable_ids_across_frames`) is the known pre-existing
  failure, identical to the pre-change baseline.
- No Layer-2 field renamed/removed: track_id, bbox, confidence, class,
  detector_source, trajectory, stable UIDs unchanged. `source` values
  ("yolo"/"color"/"novelty") unchanged.

## 9. Docker (Step 33) — VERIFIED

The production image `ho-backend:latest` was rebuilt and verified live:

- `COPY inference/detection/weights/garbage_bag_v2.pt` →
  `/app/inference/detection/weights/garbage_bag_v2.pt`
- Container model load (YoloDetector.load(), once):
  `bag model: inference/detection/weights/garbage_bag_v2.pt ['Garbage Bag']`
  `bag sha256: 2f33b83654789ef273151db3a7c1ef89adb405d076e46ca9659bcdd68ba6a824`
  (identical to the deployed host model)
- Real-frame inference inside the container (IMG_5305 frame):
  `[('person', 0.86, 'yolo'), ('person', 0.55, 'yolo')]` — production
  `d.track()` path works end-to-end in the container.

**Incident found & fixed during verification:** the image contained a stale,
legacy-format `yolov8n.pt` (sha `f9268313…`, not a torch zip) that crashed
torch 2.4.1 at person-model load — a PRE-EXISTING image defect, unrelated to
the bag model. Root cause: BuildKit served stale bytes for the unchanged
context path even after cache pruning. Fix: Dockerfile now copies a
byte-identical copy under a distinct path (`yolov8n_cpu.pt → /app/yolov8n.pt`,
sha `f59b3d83…`) with a documented `CACHEBUST` ARG for future weight refreshes.
Verified: container file `iszip True`, sha `f59b3d83…`, load OK.

## 10. Remaining blockers (honest)

1. Red ground bag in IMG_5305 not semantically detected (needs human annotation
   of D:\22 + retrain).
2. Isolated car-as-bag false positive observed in one probed frame — needs
   negative hardening in the next training round (real negative frames,
   human-verified).
3. HSV color candidates over-fire on cluttered scenes; honestly labeled
   proposals, but a learned verifier would reduce noise.

## STOP

Layer-1 work is complete and validated. Per the mandatory stop condition, no
Layer-2 redesign, Layer-3/evidence, dashboard, or new temporal features were
touched.

## 7. Performance (measured, CPU, 30 real IMG_5305 frames)

| Stage | avg | p50 | p95 |
|---|---:|---:|---:|
| Person yolov8n-640 | 108.7 ms (incl. warmup) | 45.4 ms | 85.8 ms |
| Semantic bag model garbage_bag_v2-640 | 43.4 ms | 43.0 ms | 46.1 ms |
| HSV `_detect_raw` (multi-color) | 68.8 ms | — | 87.8 ms |

Analysis cadence unchanged; no full-source-frame inference added; Layer-2
temporal semantics untouched. Script: `scripts/bench_layer1_latency.py`.

