# NEW PRODUCTION ROADMAP — Vision/AI Consolidation + Enterprise UI Overhaul

**Status:** Planning document only. No application code was changed to produce this file — everything below is grounded in a fast, non-exhaustive metadata/label audit of `D:\22` and direct reads of current source (file:line cited throughout). Follow-up sessions should execute this phase by phase using the prompts in Section 5, each scoped as narrowly as the P0-1 fix earlier in this project.

---

## 0. Executive Summary

Two things are true at once, and both matter for scoping this roadmap correctly:

1. **The "single-object/yellow-bag bias" is real, but it is not a detection-quality problem — it's a taxonomy-fragmentation problem.** Three separate, disjoint, narrow class vocabularies already exist in this repo, trained independently, never merged: `garbage_bag_v2.pt` (1 class: `"Garbage Bag"`), `best.pt` (5 classes: `bottle, juice-cup, nescafe, plate, tissue` — no can, no bag), and an unused `taco_yolo` dataset (4 classes: `bag, bottle, cup, paper` — 928 images, already YOLO-formatted, never trained on). A **unified 8-class canonical schema already exists on paper** (`scripts/datasets/unified_class_map.py`: `person, bottle, cup, can, bag, paper, crumpled_tissue, other_litter`, with a TACO mapping already written) — it was designed, documented in `ANNOTATION_PLAN.md`, and never executed. Part A of this roadmap is mostly about **finishing work that was already started**, not starting from zero.

2. **The bin-vs-ground gate (P1-5) is further along than the last audit reported.** `littering_event_detector.py` already has a full `BinZone` dataclass, a `bin_zones` config field, and live per-pair zone-containment logic (`in_bin_zone`, `bin_zone_frames`, a dedicated `BIN_ZONE_DEPOSIT` rejection reason) — this is real, working engine code, not a proxy heuristic. What's actually missing is **operator-facing configuration**: today a zone can only be added by hand-editing a commented-out YAML block (`config/events.yaml:184-190`) with normalized polygon coordinates and restarting the backend. There is no UI, no API, no per-camera database storage. Part A's bin-zone work is a config/UI/persistence task, not an ML task.

Part B (UI overhaul) targets a codebase that already has a legitimate dark, command-center-style design system in place (`dashboard/src/index.css` — navy/teal palette, JetBrains Mono, glow/pulse animations) and one page (`EventDetail.tsx`, after the last session's P1-3 fix) that already has the right *structural* pattern (primary evidence panel, demoted debug panel). The overhaul is about **applying that pattern consistently and raising its production values** (a real sequence strip, a tighter tri-crop forensic layout, a collapsible debug tab instead of an always-visible second panel) — not a rebuild from a blank canvas.

---

## 1. Part A — Vision & AI Stack Upgrade

### 1.1 D:\22 Fast Metadata & Label Audit (VERIFIED FROM RUNTIME — `ffprobe`, file listing; no frames decoded)

| Fact | Value |
|---|---|
| Real video files | **19** (`IMG_5287.MOV` … `IMG_5307.MOV`, non-contiguous numbering — some numbers were never captured or were discarded before delivery) |
| Sidecar clutter | 19 macOS `._IMG_*.MOV` AppleDouble metadata files (4KB each, not video — ignore, do not feed to any pipeline) |
| Resolution | **1920×1080, uniform across all 19 files** |
| Frame rate | **60 fps, uniform across all 19 files** |
| Duration range | 5.4s (`IMG_5303`) to 115.1s (`IMG_5306`); total ≈ 752s (~12.5 minutes) of footage across all 19 |
| Ground-truth / label / manifest files | **None found** (`find` for `.json/.csv/.xml/.txt/.yaml/.yml` under `D:\22` → zero results). This is 100% raw, unlabeled footage. |

This matches `adaptive_tuner.py`'s own docstring ("Measured on the 19 real videos in D:\\22") — confirms this is the exact, already-known reference set the adaptive tier ladder and its `learning/learning.json` history were tuned against, not a new/unfamiliar dataset.

### 1.2 Root Cause of the Class-Coverage Gap (VERIFIED FROM CURRENT SOURCE)

| Source | Classes | Images | Status |
|---|---|---|---|
| `garbage_bag_v2.pt` (production bag model) | `["Garbage Bag"]` — **1 class**, no color/type distinction | 847 (`datasets/roboflow_colored_bags/`, misleadingly named — it is single-class despite the folder name) | Trained, deployed, 96.7% precision / 97.8% mAP50 **on its own narrow eval set** (`GARBAGE_BAG_V2_EVAL.json`) |
| `best.pt` (litter model) | `bottle, juice-cup, nescafe, plate, tissue` — **5 classes, disjoint from the bag model** | unknown (external/reference model, not one of this repo's prepared datasets) | Trained, deployed |
| `taco_yolo` (prepared, unused) | `bag, bottle, cup, paper` — **4 classes, canonical-ish** | 928, already YOLO-formatted (`datasets/taco_yolo/data.yaml`) | **Prepared, never trained on, never deployed** |
| `scripts/datasets/unified_class_map.py` (designed, unused) | `person, bottle, cup, can, bag, paper, crumpled_tissue, other_litter` — **8-class canonical schema**, with a full TACO→canonical mapping table already written | N/A (a mapping script, not a dataset) | **Designed and documented (`ANNOTATION_PLAN.md`), never executed end-to-end** |
| `datasets/active_learning/cam-01/` | Real production crops auto-collected on every carry/confirm event (`adaptive.py: dump_pair_crops`, wired at `inference/pipeline.py:_collect_active_learning`) with a `labels.jsonl` | small, growing | **Already running in production** — a real feedback loop exists and has already accumulated genuine site footage crops; this is the best starting seed for site-specific retraining, better than any external transfer set |

**No class for `can` exists in any currently-deployed weight file.** No class for a "personal, non-litter, carried item" (hat, phone, umbrella) exists or should exist as a *litter* class — see §1.4.

### 1.3 Target: One Unified Model, Not Three

**Decision:** consolidate `garbage_bag_v2.pt` + `best.pt` into a single fine-tuned YOLO model trained on the already-designed 8-class canonical schema (`person, bottle, cup, can, bag, paper, crumpled_tissue, other_litter`). This directly satisfies the user's ask (white bags, black bags, tissue boxes, cans, bottles all covered as first-class detector outputs, not fallback proposals) and is also the **highest-leverage performance fix** already identified in `MASTER_REPAIR_PLAN.md` §17/§21 (P1-9) — replacing 2-3 unconditional per-frame YOLO passes with 1.

Why `bag` stays a single class (not split by color): this decision was already made and justified in `ANNOTATION_PLAN.md` §1 — color-specific classes (`yellow_bag`, `white_bag`, `black_bag`) would fragment an already-small site dataset and teach the model "color" instead of "bag shape," which fails on any unseen color. **Keep this decision.** Color remains available as an *attribute*, not a class, if ever needed for evidence text (e.g., dominant-color sampled from the confirmed bbox, purely descriptive, never gating the detection).

Data sourcing plan, in priority order (cheapest/most relevant first):
1. **`datasets/active_learning/cam-01/`** — real site crops already collected in production. Smallest volume but zero domain-gap; always include.
2. **`datasets/taco_yolo/`** (928 images, already in canonical-compatible format) — immediately usable for `bag/bottle/cup/paper`, needs relabeling only for the class-index remap (2→2 cup, 1→1 bottle line up; `paper` maps to canonical index 5, currently taco_yolo uses index 3 — needs a straight re-index, not re-annotation).
3. **`datasets/roboflow_colored_bags/`** (847 images, single-class) — remap `"Garbage Bag"` → canonical `bag` (index 4). Immediately usable.
4. **New annotation pass on `D:\22`'s 19 videos** — this is the only step that requires new human/tool labeling effort, and it's the only source that can supply `can` and `crumpled_tissue`/`other_litter` examples in the actual site environment. Extract frames at a fixed low stride (e.g., 1 fps) from all 19 videos → ~750 candidate frames total — small enough for either manual annotation (Roboflow/CVAT, per `ANNOTATION_PLAN.md` §2) or bootstrapped auto-labeling.
5. **`scripts/datasets/auto_label_grounding_dino.py`** — already exists and is exactly the right tool to bootstrap step 4: it takes a video and a list of open-vocabulary text cues (currently defaults to `["crumpled tissue paper", "small white litter", "plastic bottle"]`) and produces preliminary YOLO-format boxes via Grounding DINO. **Extend its `--cues` list** to include `"tissue box", "aluminum can", "soda can", "white plastic bag", "black garbage bag"` and run it across the 19 `D:\22` videos — this converts an otherwise fully-manual annotation task into a bootstrap-then-review-and-correct workflow, which is dramatically cheaper. Every auto-labeled frame still needs a human pass to fix false positives/negatives before it enters training, per the tool's own purpose ("preliminary bounding boxes on FAILING video frames").

### 1.4 Complex Human Interactions (hat-in-one-hand, bag-in-the-other) — this is an ASSOCIATION fix, not a detection fix

Do **not** add a `hat`/`personal_item` detection class. A hat is invisible to the litter pipeline today (it's simply not a trained class) and should stay that way — the moment you'd need to name it is if the system started false-flagging it, and there is no evidence of that in the current gate design (`_is_semantic_waste()` only ever admits `source=="yolo"` boxes for classes the litter/bag models actually predict; a hat, not being a predicted class at all, can never enter the pipeline).

The real problem the user is describing is: **when a person is simultaneously associated with two real, class-recognized objects (e.g., a bag AND a bottle) and releases only one, does the system correctly attribute the event to the released object and not the retained one?** This is `MASTER_REPAIR_PLAN.md` P1-4, already diagnosed precisely: `_select_primary_associations()` (`littering_event_detector.py:1440-1471`) hard-enforces **one litter object per person per tick** — "Littering is a single-actor / single-object interaction," by design. Today, if a person carries two real litter-class objects, only the higher-scoring one is ever tracked into a pair; the other is marked `ambiguous=True` and dropped from that tick entirely. A correct two-object implementation (wrist-slot bookkeeping, occupied-hand tracking) already exists in the **deleted** `PersonObjectAssociator` class (removed in the last session's Section-16 cleanup, archived at `_archive/dead_code_2026-09-09/`) but was never wired into the live FSM.

**Recommended fix (do this before or alongside the model consolidation, since it's independent of detector class coverage):** port the wrist-slot/occupied-hand bookkeeping concept from the archived `PersonObjectAssociator` into `_select_primary_associations()`, so a person can carry two tracked objects simultaneously (one per hand, keyed by nearest wrist keypoint from MoveNet) and each gets its own independent `_PairMemory` and its own independent FSM progression. This directly resolves Cases D and E from `MASTER_REPAIR_PLAN.md` §26 (currently both "Not supported by production FSM"), and is exactly what "carrying a bag and a hat/bottle, drop only the bag" requires: the retained item's pair simply never advances past `BAG_CARRIED`, so it can never confirm, while the released item's pair proceeds normally. No new label class needed — this is purely `littering_event_detector.py` + `inference/pose/movenet_pose.py` wrist-assignment logic.

### 1.5 Bin Zone Implementation (P1-5) — engine logic exists; build the missing configuration layer

**Already done, verified in current source (do not re-implement):**
- `BinZone` frozen dataclass (`littering_event_detector.py:87-119`) — normalized-polygon, per-camera, accepts `bbox` shorthand.
- `EventDetectorConfig.bin_zones: Tuple[BinZone, ...]` (`littering_event_detector.py:282`) and `require_ground_confirmation` gate (`:264`, defaults `True`).
- Live per-pair containment check (`_active_bin_zones`, `in_bin_zone`, `bin_zone_frames`, `littering_event_detector.py:965-966, 1462-1470, 1540`) and the dedicated `BIN_ZONE_DEPOSIT` rejection reason (`:84`).

**Missing (the actual work):**
1. **Persistence:** add a `bin_zones_json` (JSONB/text) column to `Camera` (`backend/models.py:27-40`) instead of the current static, commented-out YAML block (`config/events.yaml:184-190`) — so zones are per-camera, editable at runtime, and don't require a backend restart to change.
2. **API:** `GET/PUT /api/cameras/{id}/bin-zones` — read/write a camera's zone list (list of `{name, polygon: [[x,y],...]}` in normalized coordinates).
3. **Config wiring:** at analysis-job start (`backend/routers/analysis.py`) and live-pipeline start (`scripts/run_pipeline.py`), load the target camera's zones from the DB and pass them into `EventDetectorConfig(bin_zones=...)` instead of only reading the static YAML default.
4. **Operator UI (Part B territory, listed here for completeness):** a simple polygon-drawing tool on a representative frame (e.g., the first frame of the most recent job for that camera) — click points, see the polygon overlaid, save. This does not need to be fancy; a `<canvas>` over a static `<img>` with click-to-add-point and drag-to-adjust is sufficient. Belongs on a per-camera settings view (see §2.6).
5. **Validation test:** a new integration test that places a bag's resting centroid inside a configured zone and asserts `BIN_ZONE_DEPOSIT` (not a confirmed violation) — extending the existing `tests/test_bin_zone.py` (already created by the last session, confirm it currently only tests the YAML-static-config path).

### 1.6 Phased Execution — Part A

| Phase | Deliverable | Primary files touched |
|---|---|---|
| A1 | Class-index remap scripts for `taco_yolo` and `roboflow_colored_bags` into the canonical 8-class schema; merge into one training-ready directory | `scripts/datasets/unified_class_map.py`, `scripts/datasets/prepare_datasets.py` |
| A2 | Extend `auto_label_grounding_dino.py` cues; run against all 19 `D:\22` videos at 1fps stride; human review pass on the output | `scripts/datasets/auto_label_grounding_dino.py` |
| A3 | Train the unified single-class-schema YOLO model (`scripts/train_yolo.py --make-yaml`, already supports this workflow) on merged data + active-learning crops; evaluate against held-out site clips | `scripts/train_yolo.py`, new `inference/detection/weights/litter_unified_v1.pt` |
| A4 | Swap `yolo_detector.py` from 2-3 model instances to 1 unified model; re-run the full `tests/` suite + the existing 19-video validation matrix; re-tune `adaptive_tuner.py` tier thresholds if detection behavior shifts materially | `inference/detection/yolo_detector.py`, `adaptive_tuner.py`, `tests/` |
| A5 | Two-object-per-person association fix (port wrist-slot logic from the archived `PersonObjectAssociator`) | `littering_event_detector.py`, `inference/pose/movenet_pose.py`, new tests for Cases D/E |
| A6 | Bin-zone persistence + API + config wiring (UI is Part B, §2.6) | `backend/models.py`, new `backend/routers/cameras.py` endpoints, `backend/routers/analysis.py`, `scripts/run_pipeline.py`, `tests/test_bin_zone.py` |

Phases A1-A3 are independent of A4-A6 and can run in parallel across sessions. A4 (the actual swap-in) must run **after** A3 produces a validated model, and should itself be validated against the 19-video reference set before touching production thresholds.

---

## 2. Part B — Enterprise Surveillance UI/UX Overhaul

### 2.1 What's already there (build on this, don't replace it)

`dashboard/src/index.css` already defines a genuine command-center dark palette: `--bg-base:#0a1020`, `--bg-panel:#111c2e`, teal `--accent:#00e5b8`, `--danger:#ff4757`, JetBrains Mono for data, live-pulse/alert-flash keyframe animations, glow utilities. `EventDetail.tsx` (post P1-3 fix) already separates a primary evidence panel from a secondary "Technical Review — Full Video (Debug)" panel. **The visual language is not the problem — density, consistency, and the forensic sequence narrative are.**

### 2.2 Tri-Forensic Asset Display — concrete layout spec

Replace `EventDetail.tsx`'s current evidence grid (`:216-267`, a `sm:grid-cols-3` flat tile grid mixing snapshot/carry/release/ground/person/waste/face indiscriminately) with a structured, top-to-bottom forensic narrative:

```
┌─────────────────────────────────────────────────────────────┐
│  EVENT CLIP (6-8s, carry→departure window)         [PRIMARY] │
│  ▶ centered, 16:9, autoplay-on-hover thumbnail scrub bar     │
└─────────────────────────────────────────────────────────────┘
┌───────────────┬───────────────────┬─────────────────────────┐
│ PERSON ACTOR   │  FACE CROP        │  GROUND WASTE EVIDENCE  │
│ (full-body,    │  (sensitive —     │  (object crop at the    │
│  stable UID    │  human-review     │   confirmed ground/     │
│  badge)        │  badge)           │   abandonment frame)    │
└───────────────┴───────────────────┴─────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│  SEQUENCE STRIP:  ①Approach → ②Carry → ③Release → ④Ground →  │
│                    ⑤Departure   (existing carry/release/     │
│                    ground images from P1-2, restyled as a    │
│                    horizontal filmstrip with frame timestamps)│
└─────────────────────────────────────────────────────────────┘
```

This uses data that already exists end-to-end — `clip_path` (event clip), `person_image_path`, `face_image_path`, `waste_image_path`/`ground_image_path`, and the `carry_image_path`/`release_image_path`/`ground_image_path` trio from the last session's P1-2 fix — this is a **layout/component reorganization**, not a new backend capability. The only genuinely new backend need is trimming the event clip to a tight 6-8s window: `evidence_package.py`'s current `pre_seconds=5.0, post_seconds=5.0` (`backend/routers/analysis.py:734-735`) produces a variable-length clip (carry-to-departure span + 10s padding, which can run well past 8s on a slow walk-away); tightening this is a one-line config change plus a decision on whether to trim padding or trim the carry→departure span itself when it's long (recommend: cap total clip length at ~8s by trimming pre/post padding first, never the carry→release→ground→departure span itself, since that's the actual evidentiary content).

### 2.3 Component structure (new/changed files)

| Component | Change |
|---|---|
| `dashboard/src/components/ForensicAssetPanel.tsx` (new) | The tri-crop + clip layout above, parameterized by an `Evidence` + `Event` pair. Used by both `EventDetail.tsx` and `VideoAnalysisPage.tsx` so the two pages render identical primary evidence, not two divergent implementations (this is the root cause of the earlier P1-3 inconsistency — two separate hand-built layouts drifting apart). |
| `dashboard/src/components/SequenceStrip.tsx` (new) | The 5-step approach→carry→release→ground→departure filmstrip, replacing the current ad-hoc `sm:grid-cols-3` figure tiles. |
| `dashboard/src/components/DebugReviewPanel.tsx` (new) | Wraps the existing "Technical Review — Full Video (Debug)" content in a **collapsed-by-default** `<details>`/disclosure component (not just visually demoted — actually collapsed, per the user's ask), with the box-overlay video, timeline markers, and confidence-score breakdown inside. |
| `dashboard/src/pages/EventDetail.tsx` | Replace the current inline JSX (`:160-321`) with `<ForensicAssetPanel>` + `<SequenceStrip>` + `<DebugReviewPanel>`. |
| `dashboard/src/pages/VideoAnalysisPage.tsx` | Replace its own separately-implemented primary-evidence block (`:463-526` per the earlier audit) with the same `<ForensicAssetPanel>`, so upload-result and event-review pages are visually and structurally identical. |
| `dashboard/src/components/CameraZoneEditor.tsx` (new, ties to §1.5) | Click-to-draw polygon tool over a static reference frame, for bin-zone configuration. Lives on a per-camera settings sub-page, not on `EventDetail`. |

### 2.4 Multi-event timelines / camera switching / export sheets

- **Multi-event timeline:** `VideoAnalysisPage.tsx` currently shows only `confirmed_violations.slice(0,1)` (MASTER_REPAIR_PLAN P2-11) — extend this to render one `<ForensicAssetPanel>` per confirmed event in that job, in a horizontally-scrollable or tabbed strip, so a video with multiple events doesn't hide all but the first.
- **Camera switching:** `Dashboard.tsx`/`LiveMonitoring.tsx` already fetch `/api/cameras` — add a persistent camera selector in the app shell (`App.tsx`) rather than per-page, so switching cameras preserves context across Live Monitoring, Violations, and Dashboard views.
- **Verified violation export sheets:** a new `GET /api/events/{id}/export` (PDF or a print-formatted HTML page) containing the same tri-forensic layout plus a signature/reviewer-name field and disposition (confirmed/dismissed) — this is genuinely new backend surface, scoped as its own phase, not bundled into the visual redesign.

### 2.5 What NOT to do

- Do not introduce a new CSS framework or design token system — extend `index.css`'s existing `--var` tokens.
- Do not remove the full analyzed video entirely — collapse it, don't delete it; it remains the only tool for genuinely debugging a wrong detection.
- Do not couple the UI redesign to the Part A model swap — they're independent; the new `ForensicAssetPanel` works identically whether evidence came from the old 2-3-model stack or the new unified model.

### 2.6 Phased Execution — Part B

| Phase | Deliverable | Primary files touched |
|---|---|---|
| B1 | Build `ForensicAssetPanel` + `SequenceStrip` + `DebugReviewPanel` as isolated, reusable components against real existing evidence data (no backend changes) | `dashboard/src/components/{ForensicAssetPanel,SequenceStrip,DebugReviewPanel}.tsx` |
| B2 | Wire them into `EventDetail.tsx`, verify visually against 2-3 real historical events (jobs 87/88 with their double-event pairs are a good regression case, per the earlier audit) | `dashboard/src/pages/EventDetail.tsx` |
| B3 | Wire the same components into `VideoAnalysisPage.tsx`; fix the `slice(0,1)` limitation to show all confirmed events per job | `dashboard/src/pages/VideoAnalysisPage.tsx` |
| B4 | Tighten event-clip duration to 6-8s (backend) | `backend/routers/analysis.py`, `inference/visualization/evidence_package.py` |
| B5 | Camera zone editor UI + wiring to the §1.5 bin-zone API | `dashboard/src/components/CameraZoneEditor.tsx`, a new per-camera settings page |
| B6 | Persistent camera selector in the app shell | `dashboard/src/App.tsx` |
| B7 | Violation export sheet (new backend endpoint + print-styled page) | `backend/routers/events.py`, new `dashboard/src/pages/EventExport.tsx` |

---

## 3. Dependency Order Across Both Parts

```
A1 (data remap) ──┬── A3 (train unified model) ── A4 (swap detector) ── re-validate vs D:\22
A2 (auto-label)  ─┘

A5 (two-object association fix) ── independent, can run anytime, needs its own new tests

A6 (bin-zone persistence/API) ── independent, needed before B5 (zone editor UI) can be wired end-to-end

B1 (new components, no backend dep) ── B2 ── B3 (both pages consistent)
B4 (clip trim) ── independent, small
B5 (zone editor) ── depends on A6
B6, B7 ── independent, lowest priority
```

Recommended overall sequence: **A1→A2→A3→A4** and **B1→B2→B3→B4** can run as two parallel tracks (data/model track, UI track) since neither blocks the other. **A5, A6→B5, B6, B7** are lower-priority follow-ups once the two main tracks land.

---

## 4. Risk Notes (carried forward from this session's near-miss)

- Every phase below must end with: full `pytest` suite green, a `git commit` of the actual diff (not left uncommitted), and — for anything touching backend/inference code — a `docker compose up -d --force-recreate backend` plus a live confirmation that the running container's `StartedAt` postdates every changed file's mtime. The last session shipped correct code that the live container wasn't even running yet; don't repeat that.
- Retraining (A3) changes detector output statistics — after A4, expect `adaptive_tuner.py`'s tier thresholds and `learning/learning.json`'s accumulated history to no longer be well-calibrated for the new model. Plan to reset or re-derive the learning store, not silently carry old learned relaxations forward against a different detector's confidence distribution.
- The association fix (A5) touches the same `_select_primary_associations` function flagged as the highest-risk "do not touch without regression tests" area in `MASTER_REPAIR_PLAN.md` §21/§27. Write the Case D/E tests *before* changing the function, not after.

---

## 5. Exact Prompt Sequences (ready to paste into follow-up sessions, one phase at a time)

**Prompt — A1 (dataset remap):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase A1 strictly. Write a script (or extend `scripts/datasets/prepare_datasets.py`) that remaps `datasets/taco_yolo/` (4-class) and `datasets/roboflow_colored_bags/Trash.v1i.yolov8/` (1-class) label files into the canonical 8-class schema defined in `scripts/datasets/unified_class_map.py`, merging both into one training-ready directory tree under `datasets/unified_v1/`. Do not touch model weights, do not run training. Report the merged class-distribution counts and STOP.

**Prompt — A2 (auto-labeling D:\22):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase A2 strictly. Extend `scripts/datasets/auto_label_grounding_dino.py`'s default `--cues` list to include tissue box, aluminum/soda can, white plastic bag, and black garbage bag prompts. Run it against all 19 real videos in D:\22 at a 1fps frame stride, writing preliminary YOLO labels to `datasets/d22_autolabel/`. Do not modify D:\22 itself. Report per-video detection counts per class and STOP — do not merge into the training set yet (that needs a human review pass first).

**Prompt — A3 (train + evaluate unified model):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase A3. Using `scripts/train_yolo.py --make-yaml`, train a unified single-model YOLO on `datasets/unified_v1/` (from A1) plus the reviewed `datasets/d22_autolabel/` output (from A2) plus `datasets/active_learning/cam-01/`. Evaluate against a held-out split of the 19 D:\22 videos. Do NOT swap it into `yolo_detector.py` yet. Report precision/recall/mAP per class vs. the current `garbage_bag_v2.pt`/`best.pt` baselines and STOP.

**Prompt — A4 (swap detector stack):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase A4. Only proceed if A3's validated model exists. Replace the 2-3 separate YOLO model instances in `inference/detection/yolo_detector.py` with the single unified model. Run the full pytest suite and the existing 19-video validation matrix. Flag any `adaptive_tuner.py` tier threshold or `learning/learning.json` history that appears miscalibrated against the new model's confidence distribution, but do not silently reset it without asking. Report test results and STOP.

**Prompt — A5 (two-object-per-person association):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase A5 and MASTER_REPAIR_PLAN.md P1-4/Cases D&E. Write new failing tests first (a person carrying two independently-tracked litter-class objects, releasing only one) against the current `_select_primary_associations()` one-object-per-person limitation in `littering_event_detector.py`. Then port the wrist-slot/occupied-hand bookkeeping concept from the archived `PersonObjectAssociator` (`_archive/dead_code_2026-09-09/`) to make those tests pass, using MoveNet wrist keypoints for hand assignment. Run the full suite. Report results and STOP.

**Prompt — A6 (bin-zone persistence + API):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase A6. Add a `bin_zones_json` column to `Camera` (`backend/models.py`), add `GET`/`PUT /api/cameras/{id}/bin-zones` endpoints, and wire zone loading into `EventDetectorConfig(bin_zones=...)` at the start of both the video-upload job (`backend/routers/analysis.py`) and the live pipeline (`scripts/run_pipeline.py`), replacing the static YAML-only path. Extend `tests/test_bin_zone.py` to cover the DB-backed path. Report test results and STOP.

**Prompt — B1 (forensic components):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase B1 and the layout spec in §2.2. Build `ForensicAssetPanel.tsx`, `SequenceStrip.tsx`, and `DebugReviewPanel.tsx` as isolated components under `dashboard/src/components/`, styled with the existing `index.css` design tokens (do not add a new CSS framework). Do not wire them into any page yet. Show me a screenshot or describe the rendered layout against a real event's evidence data (e.g. event 28) and STOP.

**Prompt — B2/B3 (wire into pages):**
> Follow NEW_PRODUCTION_ROADMAP.md Phases B2 and B3. Replace `EventDetail.tsx`'s and `VideoAnalysisPage.tsx`'s separate hand-built evidence layouts with the shared `ForensicAssetPanel`/`SequenceStrip`/`DebugReviewPanel` components from B1. Fix VideoAnalysisPage's `confirmed_violations.slice(0,1)` limitation so all confirmed events in a job are shown. Verify against real historical double-event jobs (88, 87, 84). Run `npm run build`/`npx tsc --noEmit`. Report and STOP.

**Prompt — B4 (clip trim):**
> Follow NEW_PRODUCTION_ROADMAP.md Phase B4. Adjust event-clip generation so total clip length is capped at 6-8 seconds, trimming pre/post padding first and only the actual carry→departure span if it alone exceeds the cap. Do not trim carry/release/ground content. Report and STOP.

Each prompt should be run in its own session/turn, verified (tests green, container fresh, git committed) before the next one starts, per Section 4.

---

## 6. Summary

- **Part A's real problem is fragmentation, not absence** — three narrow class vocabularies and one unused canonical 8-class schema already exist; consolidate them into one model rather than designing a new taxonomy.
- **Bin-zone detection logic is already built** (`BinZone`, live containment checks, `BIN_ZONE_DEPOSIT`) — only persistence, API, and a UI editor are missing.
- **The hat/bag disambiguation is an association-layer fix** (port the archived `PersonObjectAssociator`'s wrist-slot logic), not a new detection class.
- **Part B's foundation (dark theme, primary/debug split) already exists** — the work is componentizing it once (`ForensicAssetPanel`) and using it everywhere, plus tightening clip length and adding zone-editing, multi-event, camera-switching, and export capabilities.
- Seven ready-to-paste prompts (Section 5) sequence the work; two independent tracks (data/model, UI) can proceed in parallel.
