# VIDEO ORACLE COMPARISON — IMG_5290.MOV

**Date:** 2026-09-12  
**Real video:** `D:\22\IMG_5290.MOV` (full 19.008s / 1140 frames @ ~59.97 fps)  
**Independent reference:** external video-understanding JSON (not ground truth)  
**Rule followed:** project pipeline on the **entire** video; agent did not substitute visual judgment for detection.

---

## EXPECTED REFERENCE

| Field | Value |
|---|---|
| Label | `ground_littering` |
| Window | ~13.0–19.0 s |
| Violation | `true` |
| Description | At ~13.0s, man releases a small dark bag/trash from right hand; it falls near the black dumpster |

---

## ACTUAL PROJECT RESULT

### A) Unmodified baseline (required first run)

Artifact: `project_audit/_oracle_IMG_5290_baseline.json`

| Metric | Value |
|---|---|
| Frames processed | 1140 (full video) |
| Detector | `AdaptiveEventDetector` |
| Confirmed | **0** |
| Rejected | **0** |
| Candidates | **0** |

**Verdict vs reference: FALSE NEGATIVE**

### B) After fixes — production path (Docker + Postgres)

| Metric | Value |
|---|---|
| Job | **#17** (`oracle_IMG_5290.MOV`) |
| Status | `completed` |
| Events persisted | **1** (Event **#3**) |
| Confidence | **0.8755** (~88%) |
| Actor | person UID **1** (track 3) |
| Object | waste UID **100002** (track 60005) |
| FSM timeline | CARRY **12.54s** → RELEASE **14.54s** → GROUND **14.81s** → DEPARTURE/EVENT **16.27s** |
| Location | ground litter (abandonment/departure after put-down) |

Host proof (clean + polluted learning) also confirmed the same carry→release→ground arc; see `_oracle_IMG_5290_postfix5.json`, `_oracle_IMG_5290_polluted.json`.

**Verdict vs reference: MATCH on violation + window (action inside 13–19s)**  
Class string quirk: API/DB `object_type` / `bag_class` sometimes reads `yellow_waste_bag` while evidence crops are labeled **BLACK_WASTE_BAG** and show the handheld dark bag — see limitations.

---

## FIRST DIVERGENCE (baseline)

**Stage: DETECTION → SEMANTIC ADMISSION / OWNERSHIP (before CARRIED)**

| Item | Detail |
|---|---|
| File | `inference/detection/yolo_detector.py` (`_color_fallback_track`) + gate in `littering_event_detector.py` (`_is_semantic_waste`) |
| Function | HSV color fallback emits `color_candidate_*`; semantic gate rejects them |
| Frame/time | Throughout video; critical near carry (~12s+) and put-down (~13–15s) |
| Input | Person detected; dumpster-scale black proposals; sparse wrong YOLO `Garbage Bag` |
| Actual | Only `color_candidate_*` / novelty proposals → **no semantic waste** → pairs stay `NO_BAG` → **0 candidates** |
| Expected | Small dark handheld bag admitted as semantic waste → ownership → CARRIED |
| Reason | P0-02 “admit color” was dormant: production HSV never renamed `color_candidate_*` to a waste class, so admission never engaged. Dumpster black proposals stayed proposal-only (correct) but the handheld bag never entered the FSM. |

Downstream stages (TRACKING → … → DASHBOARD) never ran for a real event on baseline because the chain never left DETECTION/admission.

---

## ROOT CAUSE

1. **Primary (FN):** Handheld dark bag invisible to FSM — color proposals not promoted when in the person carry band.  
2. **Secondary (Docker miss after promote):** Under polluted `learning.json` (`release_distance_ratio: 0.12`), early RELEASE→GROUND then **false regrab** (`PICKED_BACK_UP`) via walking/wrist association on 1–2 analysis frames (timeline: ground 14.81 → CARRIED 15.07 on jobs 14–16).

---

## FIX

### 1. Handheld color promotion
- **File:** `inference/detection/yolo_detector.py` — `promote_handheld_color_class`
- Promotes `color_candidate_{hue}` → `{hue}_waste_bag` only for person-carry-band, handheld-scale boxes (not dumpster).
- Never promotes white (shirt ownership theft).
- Tests: `tests/test_handheld_color_promote.py`

### 2. Missing-bag abandonment after put-down
- **File:** `littering_event_detector.py`
- After RELEASE/ON_GROUND, if semantic bag drops out of the set (promotion stops outside carry band), continue settle/abandonment so ground litter can confirm.

### 3. Post-ground regrab hysteresis (Docker PICKED_BACK_UP)
- **File:** `littering_event_detector.py`
- Before ground: only **strong lift** may reclaim.
- After ground: require **sustained** `regrab_lift_streak >= max(2, smoothing_window)` of strong lift **and** association; do not accept one-frame `walking_reclaim`.
- Persist `ground_bag_centroid`; measure lift from deeper of release/ground pose.
- Pause abandonment while a lift streak is building.
- Tests: regrab unit + pipeline integration still pass.

---

## TEST RESULT

| Suite | Result |
|---|---|
| `tests/test_handheld_color_promote.py` | PASS |
| `tests/test_littering_event_detector.py` (incl. regrab) | PASS |
| `tests/test_pipeline_integration.py::test_put_down_then_regrab_does_not_confirm` | PASS |

Automated tests alone are **not** treated as proof of the oracle.

---

## REAL VIDEO RESULT

| Run | Result |
|---|---|
| Baseline full video | FN — 0 events |
| Host POSTFIX5 (clean learning) | CONFIRMED `black_waste_bag`, UIDs 1 / 100002, release ~17.9s (tier-2) |
| Host polluted learning | CONFIRMED same UIDs / late release path |
| Docker jobs 13–16 (pre final regrab gate) | completed, **events=0**, `PICKED_BACK_UP` |
| Docker job **17** (full video, production worker) | **events=1**, timeline carry 12.54 → release 14.54 → ground 14.81 → event 16.27 |

Full video always used (no crop to 13–19s; no injected events from the independent JSON).

---

## DATABASE RESULT

| Field | Proven value |
|---|---|
| Job | `VideoAnalysisJob` id **17**, status `completed`, `events_count=1` |
| Event | `Event` id **3**, `analysis_job_id=17`, conf **0.8755** |
| Actor UID | **1** (non-null) |
| Object UID | **100002** (non-null) |
| Evidence rows | clip, person, waste, face, ground paths under `evidence_store/3/` |

---

## API RESULT

| Endpoint | Result |
|---|---|
| `GET /api/analysis/jobs/17` | `events_count=1`, `confirmed_events=1`, `persisted_event_ids=[3]` |
| `GET /api/analysis/jobs/17/events` | Event #3 with UIDs |
| `GET /api/events/3` | same |
| `GET /api/events/3/evidence` | paths to clip/person/waste/face/ground |
| Evidence files HTTP | `person.jpg` / `waste.jpg` / `face_evidence.jpg` / `event_clip.mp4` → **200** |

---

## DASHBOARD RESULT

| Check | Result |
|---|---|
| URL | http://localhost:5173/analysis/17 |
| Event exists | **Yes** — “VIOLATION DETECTED”, Event #3 |
| Timestamp / sequence | CARRY 12.54 · RELEASE 14.54 · GROUND 14.81 · DEPARTURE/EVENT 16.27 |
| Actor | PERSON UID #1 |
| Waste | WASTE UID #100002 |
| Event detail | http://localhost:5173/violations/3 — Confirmed, behavior checks CARRY/RELEASE/GROUND/DEPARTURE/NO REGRAB |
| Event clip | Present (“EVENT CLIP — CARRY TO DEPARTURE”) |
| Person / face / waste images | Files on disk + API 200; dashboard Primary Evidence shows actor/object UIDs (crop tiles depend on browser media load) |
| Full analyzed video | Under Technical Review (`analysis/17/analyzed.mp4`) — secondary |
| Unrelated person/object selected | **No** — single person UID 1; object UID 100002 |
| Box flood | Not used as success criterion here; prior P0-05 keeps color≠proposal flood |

Dashboard agrees with stored Event #3 (not backend-only claim).

---

## COMPARISON SUMMARY

| Dimension | Reference | Project (job 17) | Classification |
|---|---|---|---|
| Violation | true | confirmed event | **MATCH** |
| Time window | 13–19s | release 14.54 / confirm 16.27 | **MATCH** (inside window; reference start ~13s is approximate) |
| Actor | man with bag | UID 1 | **MATCH** (single person scene) |
| Object | small dark bag | UID 100002; evidence shows dark handheld bag | **MATCH** visually; **API class string** may say `yellow_waste_bag` |
| Evidence | actor + waste | person/waste/face/ground/clip | **PROVEN** |

---

## REMAINING LIMITATIONS

1. **`bag_class` / `object_type` string can disagree with evidence overlays** (API `yellow_waste_bag` vs crop label `BLACK_WASTE_BAG`) — finalize may keep a late color flicker; UID/track is the reliable identity.  
2. **Polluted global learning** still aggressively lowers release geometry; regrab hysteresis mitigates false `PICKED_BACK_UP` but learning hygiene remains operational debt.  
3. **Host vs Docker track numerics differ** (CPU/GPU); both can confirm after the regrab fix, but absolute release frame is not bit-identical.  
4. **Reference start 13.0s vs project carry ~12.5 / release ~14.5** — material enough to note, not a FN. Exact hand-open moment is not frame-aligned to the external JSON.  
5. **Evidence crop quality** remains coarse (small dark bag on black pants).  
6. Dashboard media tiles may render late/empty in some browser sessions even when API file routes return 200 — treat file + API as authoritative for media existence.

---

## ARTIFACTS

- `project_audit/_oracle_IMG_5290_baseline.json` — unmodified FN  
- `project_audit/_oracle_IMG_5290_diag.json` — admission diagnosis  
- `project_audit/_oracle_IMG_5290_postfix*.json` / `_postfix5.json` / `_polluted.json` — host re-runs  
- `project_audit/_oracle_job13.json` … `_oracle_job17.json` — Docker progression  
- `evidence_store/3/` — Event #3 package  
- `project_audit/D22_ACCEPTANCE.json` — IMG_5290 marked `LITTER`

---

## FINAL VERDICT

| Claim | Status |
|---|---|
| Baseline FN established before code change | **PROVEN** |
| First divergence = semantic color admission | **PROVEN** |
| Full-video production confirm after fix | **PROVEN** (job 17) |
| Database + API + Dashboard show same event | **PROVEN** |
| Independent reference agreement (violation in 13–19s, correct actor/dark bag evidence) | **PROVEN** with class-string caveat |
| Perfect class label / frame-exact 13.0s open-hand | **NOT PROVEN** / limited as above |
