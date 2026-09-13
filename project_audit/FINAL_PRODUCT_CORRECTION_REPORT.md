# Final Product Correction Report — Ground vs Bin + Primary Clip

**Date:** 2026-09-13  
**Scope:** Job #31 / `IMG_5306.MOV` product failures (wrong actor, bin vs ground confusion, full-video primary UI)  
**Constraint:** General FSM/UI rules only — no video-specific hardcoding, no model retrain, no Novita spend

---

## 1. What failed (Job #31)

| Expected | Actual (pre-fix) |
|----------|-------------------|
| Confirm Person 1 (black t-shirt) dropping red bag on asphalt ~f6420 | Confirmed Person UID #5 / Track #95 (plaid shirt by van) ~f4025 |
| Reject dumpster disposal as bin | Bin path partially OK (`BIN_DISPOSAL` ~f6097); hip-attached color blob still confirmed as ground litter |
| Primary media = ~10s event clip | `AnalysisDetail` exposed Event Clip / AI Tracks / Raw Source switches |

**Root causes**

1. **Clothing / static color latch** — Adaptive tiers 1–2 confirmed a body-adjacent color blob that never physically moved as a discarded bag; person walking away inflated `norm_distance`.
2. **Relaxed-tier override** — `_dedup_confirmed` / `_flush_pending` could surface relaxed confirmations that tier-0 rejected as `NOT_ENOUGH_CARRIED_FRAMES`.
3. **Primary UI** — `AnalysisDetail` / `EventDetail` passed full analyzed/original URLs into `ForensicAssetPanel`.

---

## 2. Fixes shipped

### Detector ([littering_event_detector.py](littering_event_detector.py))

- New rejection reason `NO_PHYSICAL_SEPARATION`.
- Confirm gate requires (when `ever_contained`):
  - sustained post-release separation (`max_post_release_norm_distance` + `separated_frames`), **and**
  - real bag motion (`max_bag_displacement_px` vs person height).
- Alt-release paths (`crit_feet`, `crit_held_static`, …) blocked while `still_attached` (high containment + tiny separation).
- Fast-drop: classic distance release allowed at `max(2, min_carried_frames - 2)`.
- YOLO-confirmed bags preferred over color/novelty when one person has multiple associations.

### Adaptive wrapper ([adaptive_tuner.py](adaptive_tuner.py))

- Relaxed confirmations must pass `_event_has_physical_separation` (sep + bag displacement).
- `_primary_rejected_weak_carry` blocks clothing latches when tier-0 already rejected for separation / weak carry without real motion.
- `_dedup_confirmed` only includes higher-tier events that were `_accept`-ed (no silent report inflation).

### Dashboard

- [ForensicAssetPanel.tsx](dashboard/src/components/ForensicAssetPanel.tsx) — primary stream = `evidence.clip_path` only; stream switcher removed.
- [AnalysisDetail.tsx](dashboard/src/pages/AnalysisDetail.tsx) / [EventDetail.tsx](dashboard/src/pages/EventDetail.tsx) — no longer pass full video URLs into primary panel.
- Full analyzed/original remain under collapsed `DebugReviewPanel` (`defaultOpen={false}`).

---

## 3. Validation evidence

### Unit tests

```
tests/test_critical_product_corrections.py
tests/test_adaptive_tuner.py
tests/test_correctness_audit.py
→ 24 passed
```

New coverage:

- Hip/static latch → `NO_PHYSICAL_SEPARATION`
- Real separated discard → confirm allowed
- Fast-drop below full `min_carried_frames` with separation → confirm allowed
- Adaptive flush suppresses unseparated relaxed confirms

### Job #31 frames.jsonl replay (AdaptiveEventDetector)

| Metric | Before | After |
|--------|--------|-------|
| Accepted confirmations | 1 (Track #95 / bag 60070 @ ~f4025) | **0** |
| Clothing latch emitted | Yes (tier 1/2) | **No** |

Replay command path: load `evidence_store/analysis/31/frames.jsonl` through `AdaptiveEventDetector(enable_learning=False)`.

### Residual (honest)

On the **same** Job #31 detection archive, the true ground litter (~f6420, person track 303 + YOLO `Garbage Bag`) still does **not** confirm: pairs stay at `BAG_NEAR_PERSON` / reject `NOT_ENOUGH_CARRIED_FRAMES` because carry latching against the ground-plane band remains weak in that archive.  

A **fresh full pipeline re-run** of `IMG_5306.MOV` is required to re-emit detections under the new gates and to verify the true event end-to-end. Helper:

```bash
docker exec littering-backend python /app/project_audit/_rerun_img5306.py
```

---

## 4. Acceptance checklist (post re-run)

- [ ] Exactly **1** confirmed ground-litter event ≈ black-t-shirt actor ~f6420  
- [ ] f4025 clothing latch **not** confirmed  
- [ ] Dumpster deposit ~f6097 rejected as `BIN_DISPOSAL` / non-violation  
- [ ] Dashboard primary player shows **~10–12s event clip only** (full video only under Technical Review)

---

## 5. Files touched

- `littering_event_detector.py`
- `adaptive_tuner.py`
- `dashboard/src/components/ForensicAssetPanel.tsx`
- `dashboard/src/pages/AnalysisDetail.tsx`
- `dashboard/src/pages/EventDetail.tsx`
- `dashboard/src/pages/VideoAnalysisPage.tsx`
- `tests/test_critical_product_corrections.py`
- `tests/test_correctness_audit.py`
- `project_audit/_rerun_img5306.py`
- `project_audit/FINAL_PRODUCT_CORRECTION_REPORT.md` (this file)
