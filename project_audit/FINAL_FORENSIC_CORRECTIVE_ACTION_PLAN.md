# MOTARED — FINAL FORENSIC CORRECTIVE ACTION PLAN

**Status:** AUDIT COMPLETE — NO CODE CHANGED BY THIS DOCUMENT
**Date:** 2026-09-16
**Branch audited:** `work/job57-fix` @ `712133f` (working tree, uncommitted changes included)
**Scope:** whole semantic decision path — video → detection → tracking → identity → FSM → event → evidence → dashboard → learning

**Evidence tags used throughout:**
`[CODE]` verified by reading the current source · `[RUNTIME]` verified from run artifacts in this repo · `[VIDEO]` verified from extracted frames (prior forensic job, re-checked against code here) · `[INFERRED]` reasoned, not directly measured · `[UNVERIFIED]` hypothesis needing a measurement

---

## 1. EXECUTIVE SUMMARY

The system's recall problem is largely solved. Its **precision problem is architectural, not numeric.**

Three independent structural defects, each sufficient on its own to produce `VIOLATION_CONFIRMED` for a person who did not litter, are live in production right now:

1. **The accused object is not guaranteed to be the object the person handled.** On `M.MOV` the confirmed violation object was the **green dumpster**, detected as `Garbage Bag` by the bag model, while the real yellow bag — which the person lifted *into* that dumpster — never entered the FSM at all (it stayed a non-semantic `color_candidate_yellow` proposal). `[RUNTIME]` `[VIDEO]` The FSM has no concept of "this object must have physically moved under human agency"; its anti-furniture guards are all satisfiable by a *static object seen through a churning tracker*. `[CODE]`

2. **The online learning store is a one-way recall ratchet that has been running in production for 95 videos.** Every video producing a rejection reason advances that reason's relaxation by one bounded step — including videos where the rejection was **correct**. There is no evidence path by which a false positive can ever tighten a threshold. `[CODE]` The currently pinned production thresholds are the *maximum* relaxation for 5 of 6 learnable reasons: `release_distance_ratio 0.35→0.12`, `min_carried_frames 6→3`, `smoothing_window 5→3`, `departure_distance_ratio 1.25→0.85`, `departure_motion_ratio 0.65→0.45`, `min_abandonment_frames 8→5`, `feet_release_frames 3→2`. `[RUNTIME]` (`learning/inference_pin.json`)

3. **The final decision is a logical OR across three detectors of decreasing strictness.** `AdaptiveEventDetector` runs tier 0 / 1 / 2 concurrently; a confirmation from *any* tier can reach the dashboard. System precision is therefore bounded by the **loosest** tier, not by the production config. The suppression list that lets tier 0 veto a relaxed rescue covers only 5 rejection reasons — `PICKED_BACK_UP`, `ASSOCIATION_AMBIGUOUS`, `OTHER_PERSON_CLOSER`, `PERSON_DID_NOT_DEPART` and `EVENT_CONFIDENCE_TOO_LOW` are **not** vetoable. A relaxed tier can therefore overturn tier 0's correct "the person picked it back up" and "we cannot tell which person did it" decisions. `[CODE]`

Measured consequence on the frozen set (11 clips, run `frozen_eval_20260915T112844Z`): **clip-level P = 0.89 / R = 1.00**, but **temporal P = 0.286** — 8 true event matches against **20 temporal false positives**, with the single bin-disposal hard negative (`IMG_5305`) confirmed as a violation and `IMG_5306` emitting **9 confirmations for 1 real event**. `[RUNTIME]`

The corrective direction is not "tune it down." It is:

- make **object agency** (self-motion under a hand) a precondition of `CARRIED`, not an inference from proximity;
- make **VALID_DISPOSAL** and **RECOVERED** first-class terminal outcomes rather than absences of evidence;
- make **GROUND provisional** until an observation horizon closes;
- make the decision a **single arbitrated verdict per physical incident**, not an OR over detectors and not one row per tracker churn;
- make learning **evidence-symmetric and offline-gated**, so a false positive can tighten as readily as a miss can loosen.

Everything currently working — detector recall, deterministic inference, stable-UID plumbing, evidence generation, the event-first dashboard, job control — is preserved. Nothing in this plan removes a detection source.

---

## 2. CURRENT SYSTEM STATUS

| Layer | Module | State |
|---|---|---|
| Detection | `inference/detection/yolo_detector.py` | 3 models pooled: `yolov8n.pt` (person), `best.pt` (litter classes), `garbage_bag_v2.pt` (bags). Cross-model NMS-style dedup. HSV `ColorBagTracker` + novelty detector emit **proposals**. `[CODE]` |
| Promotion | `promote_handheld_color_class()` | HSV `color_candidate_*` → `<hue>_waste_bag` only when inside the person's carry band and area ratio ∈ [0.003, 0.12]. `[CODE]` |
| Tracking | `inference/tracking/bytetrack_tracker.py` | ByteTrack over pooled detections. |
| Object identity | `ObjectIdentityManager` | Greedy nearest-neighbour, 220 px gate, class-family tokens, per-frame reservation, monotonic uids from 100001. `[CODE]` |
| Person identity | `PersonIdentityManager` | raw-id map → spatial/temporal re-association (200 px, 45 frames) + bbox/pose geometry, one-uid-per-frame claim. `[CODE]` |
| Semantics | `_is_semantic_waste()` | `source ∈ {yolo, color}` and class not `color_candidate_*` / `detected_object`. Novelty and unpromoted colour never enter the FSM. `[CODE]` |
| Decision | `LitteringEventDetector` (3433 lines) | 8-state FSM per `(person_uid, object_uid)` pair. `[CODE]` |
| Adaptation | `AdaptiveEventDetector` | 3 concurrent tiers + persistent `LearningStore`. **Always on in production** (both entry points hardcode `auto_tune=True`). `[CODE]` |
| Persistence | `backend/routers/analysis.py` | Cluster dedup by `(actor_uid, object_uid)` / time, then one `Event` row, `status="confirmed"`. `[CODE]` |
| Dashboard | `dashboard/src/pages/Violations.tsx` | Lists every `Event` row; all rows are violations. `[CODE]` |
| Offline learning | `learning_offline/` | Verdict → dataset → fine-tune → frozen eval → `F1↑ & FPR ≤ 1.1×` gate → versioned weight promotion. **Sound.** `[CODE]` |
| Determinism | `inference/runtime_determinism.py`, `deterministic=True` on upload | Seeds pinned, learning writes frozen, reads from `inference_pin.json`. `[CODE]` |

---

## 3. LATEST REAL-WORLD PERFORMANCE SUMMARY

**Frozen set** (`evaluation/frozen_test_set.v1.json` v1.0 — 8 LITTER, 3 NO_EVENT, 1 documentation duplicate).

Full run `frozen_eval_20260915T112844Z.json` (n = 11) `[RUNTIME]`:

| Metric | Value |
|---|---|
| Clip precision / recall / F1 | 0.889 / 1.000 / 0.941 |
| **Temporal precision / recall / F1** | **0.286** / 1.000 / 0.444 |
| Temporal TP / FP / FN | 8 / **20** / 0 |
| Hard negatives passed | `IMG_5115` ✅, `IMG_5303` ✅, **`IMG_5305` ❌ (2 confirmations)** |
| Confirmations per clip | `IMG_5306`: **9** (1 real event) · `IMG_5290`: 5 · `A`: 4 · `IMG_5117`: 2 · `IMG_5118`: 2 · `M`: 2 |

Partial run `frozen_eval_20260915T123621Z.json` (n = 5) shows `IMG_5118` (multi-person selective) flipping to a **miss** — recall on the hardest positive is not stable across runs of a nominally deterministic pipeline. `[RUNTIME]` `[UNVERIFIED — needs a controlled 3× replay to separate config drift from true non-determinism]`

**Two operator-recorded phone videos** (prior forensic job, re-verified against code here):

| | `M.MOV` (job 57) | `IMG_5613` (job 50) |
|---|---|---|
| Real-world truth `[VIDEO]` | Carries yellow bag → **lifts it into the green dumpster** → walks away empty-handed. Ground trash is pre-existing from t = 1.0 s. **VALID DISPOSAL** | Passer-by walks past an already-overflowing dumpster holding a water bottle. **NO ACTION** |
| System verdict `[RUNTIME]` | `VIOLATION_CONFIRMED`, conf 0.924, ×2 at the same instant | `VIOLATION_CONFIRMED`, conf 0.94 |
| Accused object `[RUNTIME]` | **The dumpster** (`Garbage Bag` tracks 10006/10011/10014, bbox ≈ the green container region) | Container / pre-existing sack |
| Real object | `color_candidate_yellow` f181–f213, **rising** y 594→522 — never became semantic, never entered the FSM | none |
| Frozen-set label | `LITTER @ 5.34 s` — **incorrect**; 5.34 s is the bag at its *highest* point, entering the bin | not in set |

`M.MOV`'s file hash matches the frozen reference exactly, so this is a label error, not a stale file. `[RUNTIME]`

---

## 4. KNOWN FAILURES

**F1 — Static container confirmed as discarded litter.** `M.MOV`, `IMG_5613`. `[RUNTIME]`
**F2 — Valid bin disposal confirmed as ground littering.** `M.MOV`, `IMG_5305`. `[RUNTIME]`
**F3 — Duplicate confirmations for one physical incident.** `IMG_5306` ×9, `M.MOV` ×2 simultaneous, `A` ×4. `[RUNTIME]`
**F4 — Wrong-object attribution.** The confirmed object ≠ the handled object in both phone videos. `[RUNTIME]`
**F5 — Relaxed tier overturns a correct strict rejection.** `PICKED_BACK_UP` and `ASSOCIATION_AMBIGUOUS` are not in the veto set. `[CODE]`
**F6 — Learned thresholds saturated at maximum permissiveness** after 95 videos, applied to every production run via the pin. `[RUNTIME]`
**F7 — Stale evidence survives release invalidation.** `_invalidate_stale_release()` leaves `ground_evidence_frames`, `bin_zone_frames`, `separated_frames`, `max_post_release_norm_distance`, `abandonment_frames`, `departure_frame/ts` intact. `[CODE]`
**F8 — `release_invalid` is never cleared**, so a pair that once lost continuity can never confirm a genuine later littering act unless `live_release_ground_ok` had already latched. `[CODE]` (recall risk, not precision)
**F9 — Non-causal timestamps possible.** Because F7 preserves `departure_frame/ts` across invalidation, a re-established release can carry `departure_ts < release_ts`. `[CODE]` `[UNVERIFIED — not yet observed in a stored event]`
**F10 — Multi-person selective positive unstable** (`IMG_5118` confirmed in one run, missed in another). `[RUNTIME]`
**F11 — No UNCERTAIN lane.** Every persisted `Event` is `status="confirmed"` and appears on the violations page. `[CODE]`

---

## 5. CONFIRMED ROOT CAUSES

### RC-1 — Motion and stationarity are computed on RAW track ids, not stable object uids `[CODE]`

`_update_bag_history()` and `_is_stationary()` both key on `bag.track_id`:

```
littering_event_detector.py:1666-1674   self._bag_history[bag.track_id]
littering_event_detector.py:1695-1722   hist = self._bag_history.get(bag.track_id)
```

while pair identity, ownership and all evidence key on `bag_uid`. When the tracker churns ids — exactly what happens on a large static container producing differently-cropped boxes — the freshly-minted history has `len < 2`, so:

- `_is_stationary()` returns **False** for an object that has not moved at all;
- `b_step` is `None`, and the `moves_with_person` branch is entered on absent evidence;
- with `stationary == False` the `zone_carry` suppressor at `:1820` (`if stationary and not wrist_near: zone_carry = False`) **does not fire**.

**This is the single mechanism that lets a static dumpster become `BAG_CARRIED`.** Every downstream anti-static guard is defeated by the same fact.

### RC-2 — `max_bag_displacement_px` measures detection-box wander, not object motion `[CODE]`

`mem.max_bag_displacement_px = max(..., _dist(mem.first_bag_centroid, info.bag_centroid))` accumulates across *rebinds to different raw tracks of the same uid*. The dumpster's successive crops — centroids ≈ (533,929) → (500,853) → (654,990) `[RUNTIME]` — produce ~170 px of apparent displacement. The `bag_actually_moved` guard in `_rejection_reason()` and `_event_has_physical_separation()` in the tuner (`bag_disp >= 50.0`) are both satisfied by a perfectly stationary object.

### RC-3 — `moves_with_person` is a *correlation* test with no agency requirement `[CODE]`

A person walking parallel to a static object, under RC-1's broken stationarity, yields `moves_with_person = True` → `ever_moved_with_person = True` → the carry-origin link is satisfied *without pose*, because the fallback is `(ever_wrist_near or ever_moved_with_person)`. In the no-pose branch a static object is indistinguishable from a carried one.

### RC-4 — VALID_DISPOSAL is representable only as an absence `[CODE]`

`EventState` has no `VALID_DISPOSAL`. The only container semantics are:

- `bin_zones`, operator-drawn polygons, **empty by default** — `in_bin_zone` was `False` for every failing job `[RUNTIME]`;
- `BIN_DISPOSAL`, awarded when `ground_evidence_frames <= 0`, i.e. purely negative evidence.

There is no container detector, no trajectory-into-container reasoning, no disappearance-behind-container reasoning. A bin deposit is recognised only by *failing to see the ground* — and `_release_pose_on_ground()` manufactures synthetic ground credit on missing-object ticks, which directly overrides that failure. Worse, `_rejection_reason()` **writes** `mem.ground_evidence_frames = max(..., 1)` for short AIDM arcs, converting "we never saw the ground" into "we saw the ground."

### RC-5 — The learning ratchet is evidence-asymmetric `[CODE]` `[RUNTIME]`

`LearningStore.record()` advances `step_index[reason] += 1` for every video where `count > 0`, unconditionally. Rejections are treated as *misses by definition*. There is no signal — not operator verdicts, not frozen-eval FPR — that can decrement a step. `LEARNING_STEPS` contains only relaxations. `videos_analyzed: 95` with 87 recorded `NO_RELEASE_TRANSITION` occurrences has driven release sensitivity to its floor.

### RC-6 — Tier arbitration is a disjunction `[CODE]`

`AdaptiveEventDetector.update()`: tier 0 confirms → emit; tiers 1–2 confirm → buffer 3 s, then emit unless a duplicate, unless `_event_has_physical_separation()` fails (defeated by RC-2), unless `_primary_rejected_weak_carry()` matches one of `{NOT_ENOUGH_CARRIED_FRAMES, NO_RELEASE_TRANSITION, NO_PHYSICAL_SEPARATION, BIN_DISPOSAL, BIN_ZONE_DEPOSIT}`. Everything else tier 0 rejected is rescuable.

### RC-7 — Event identity keys on a raw track id `[CODE]`

`AdaptiveEventDetector._is_duplicate()` and `_accept()` use `ev.person_track_id` — the **raw** id, not `event_actor_person_uid`. A person-track switch produces two "distinct" events for one incident. Backend clustering partly repairs this for uploads; the live-camera HTTP path has no equivalent.

### RC-8 — Ground is terminal-by-timer, not provisional-by-horizon `[CODE]`

`BAG_ON_GROUND` accumulates `abandonment_frames` and confirms at `min_abandonment_frames` (**learned down to 5 ticks ≈ 0.6 s @ 8 fps**). Confirmation is irreversible: `mem.emitted = True`, and `PERSON_DEPARTED` explicitly refuses all further reclaim. Recovery evidence arriving after the timer cannot reach the decision. The regrab path additionally requires a **strong lift** (`≥ 0.25 · person_height`) sustained for ≥ 2 ticks *plus* AIDM grip — a demanding bar, while abandonment needs only a timer.

### RC-9 — Release invalidation is partial `[CODE]`

See F7. The Job-57 hardening clears the *pose* state but not the *accumulated counters*, so a fresh arc inherits the stale arc's ground, separation and departure evidence.

---

## 6. UNCONFIRMED HYPOTHESES

| # | Hypothesis | How to settle |
|---|---|---|
| H1 | `garbage_bag_v2.pt` systematically classifies dumpsters/skips as `Garbage Bag` at high confidence | Run the detector alone over 20 container-containing frames; log class + conf + area for every container. No FSM involved. |
| H2 | `IMG_5118`'s run-to-run flip is genuine non-determinism, not config drift | 3× replay under `deterministic=True` with a hash of the effective config recorded per run; compare `frames.jsonl`. |
| H3 | `_release_pose_on_ground()`'s synthetic ground credit is the dominant confirm path on the FP clips | Count confirmations with `aidm_synthetic_ground=True` vs `live_ground_ticks > 0` across the frozen set. The field is already emitted. |
| H4 | Person-uid fragmentation contributes to F3 duplication | For each frozen clip, log `len(set(person_uid))` vs the number of distinct people visible. |
| H5 | The 3 s `RELAXED_EMIT_GRACE_SEC` is too short for tier 0 to win on long carries | Log per-confirmation `adaptive_tier`; if tier ≥ 1 dominates, tier 0 is structurally losing the race. |
| H6 | Pre-existing ground litter creates pairs that later receive carry-origin credit from a passer-by | Instrument `_create_pair` with `first_seen_stationary` and count. |

---

## 7. ROOT-CAUSE MATRIX

Legend for **Regression risk**: ▲ high (may cut true positives) · ◆ medium · ▼ low.

### RCM-01 · Static container attributed as carried litter
- **Root cause:** RC-1 + RC-2 + RC-3
- **Trigger:** large static object classified as a waste class, tracker id churn, person walking within `association_radius_ratio` (1.0 person-heights)
- **Current behaviour:** `NO_BAG → BAG_NEAR_PERSON → BAG_CARRIED` on an object that never moved
- **Why the system misinterprets it:** stationarity is computed on a churning key and returns False; displacement is measured on detection-box wander; the no-pose carry-origin fallback accepts motion correlation as agency
- **Desired behaviour:** an object may become `CARRYING` only with positive agency evidence — a wrist/hand association, **or** self-motion of the *same stable uid* exceeding scene noise while the object is off the ground plane
- **Files:** `littering_event_detector.py` `_update_bag_history`, `_is_stationary`, `_evaluate_pair`, `_advance_pair` carry gate; `inference/tracking/object_identity.py`
- **Current safety:** size gate (`bag_area > 0.40 · person_area`, `bh > 0.80 · ph`, `area > 65000`); `min_carry_origin_confidence`; static-ground-clutter gate at pair creation
- **Missing safety:** uid-keyed motion history; self-motion (agency) requirement; container-scale/static veto at the detection layer
- **Repair:** P0-1, P0-2
- **Risk:** ◆ — could suppress genuine carries where pose is absent and the object is small/low-contrast
- **Validation:** `M.MOV` and `IMG_5613` must not confirm; all 8 frozen LITTER clips must still confirm

### RCM-02 · Valid bin disposal confirmed as ground littering
- **Root cause:** RC-4
- **Trigger:** person deposits into a container; object leaves the hand and disappears into/behind it
- **Current behaviour:** absence of ground evidence is either overridden by `_release_pose_on_ground()` synthetic credit or reaches `BIN_DISPOSAL` only by luck
- **Desired behaviour:** a positive `VALID_DISPOSAL` terminal state with its own evidence bundle, evaluated **before** any violation verdict
- **Files:** `EventState`, `RejectionReason`, `_advance_pair`, `_rejection_reason`, `_make_event`, `config/events.yaml`, detector class map
- **Current safety:** `require_ground_confirmation`, `bin_zones` (unconfigured), `BIN_DISPOSAL`
- **Missing safety:** container detection; trajectory-into-container; disappearance-at-container; upward-then-vanish motion signature; the synthetic-ground override must not be able to defeat a container hypothesis
- **Repair:** P1-1
- **Risk:** ▲ — a too-eager container hypothesis suppresses real litter dropped near a bin (`IMG_5290` is precisely this case and must keep confirming)
- **Validation:** `IMG_5305` → NO violation; `M.MOV` → `VALID_DISPOSAL`; `IMG_5290`, `IMG_5295`, `IMG_5306` → still `VIOLATION`

### RCM-03 · Temporary ground contact confirmed before recovery
- **Root cause:** RC-8
- **Trigger:** put-down then pick-up within the abandonment window; learned `min_abandonment_frames = 5` ≈ 0.6 s
- **Current behaviour:** irreversible confirm; `PERSON_DEPARTED` refuses reclaim outright
- **Desired behaviour:** `ABANDONED` is provisional; a decision horizon must elapse with the actor observable and no recovery before `FINAL_VIOLATION`
- **Files:** `_advance_pair` (ON_GROUND branch), `_evaluate_confirmation`, `_finalize_pair`
- **Current safety:** `regrab_lift_streak`, `PICKED_BACK_UP` reclassification
- **Missing safety:** two-phase commit (provisional verdict → horizon → final verdict); symmetric evidence bars for abandon vs recover
- **Repair:** P1-2, P1-3
- **Risk:** ◆ — adds latency; short clips may end inside the horizon (handled explicitly, §14)
- **Validation:** a recovery clip must reject; `IMG_5117` (3.47 s event, short clip) must still confirm

### RCM-04 · Pre-existing litter attributed to a passer-by
- **Root cause:** RC-3 + weak pair-creation gate
- **Trigger:** object already resting when first seen; person walks within the association radius
- **Current behaviour:** the static-clutter gate only blocks pairs when `bag_confidence < 0.40` **and** not near **and** not carried; a confident pre-existing sack passes
- **Desired behaviour:** an object stationary from first sighting and never wrist-associated can never become an event object
- **Files:** `update()` pair-creation gate, `_bag_was_handled`
- **Missing safety:** a `first_seen_static` flag on the stable uid, independent of confidence
- **Repair:** P0-2
- **Risk:** ▼
- **Validation:** `IMG_5613`, `IMG_5303` → no confirmations

### RCM-05 · Stale separation / stale release evidence
- **Root cause:** RC-9
- **Trigger:** object lost, then a different blob rebinds
- **Current behaviour:** `_invalidate_stale_release()` clears pose but preserves ground/separation/departure counters; `release_invalid` never clears
- **Desired behaviour:** invalidation resets the **entire** post-release evidence bundle; a fresh valid release clears `release_invalid`
- **Files:** `_invalidate_stale_release`
- **Repair:** P0-3
- **Risk:** ▼ (precision) / ◆ (P0-3b clearing `release_invalid` restores recall — verify it doesn't reopen the Job-57 class)
- **Validation:** `tests/test_job57_stale_tracking.py` A–G must all still pass

### RCM-06 · Object UID substitution
- **Root cause:** `ObjectIdentityManager` matches on 220 px + class-family only — no size continuity, no appearance
- **Trigger:** two same-family objects within 220 px; or a container fragment near a real bag
- **Current behaviour:** greedy largest-first claim can hand a real bag's uid to a container crop
- **Desired behaviour:** add a size-ratio gate and a per-uid velocity plausibility gate; below the bar, mint a fresh uid and mark the arc `UNCERTAIN`
- **Files:** `inference/tracking/object_identity.py`
- **Repair:** P0-4
- **Risk:** ◆ — over-constraining fragments tracks and costs recall on occlusion
- **Validation:** uid-count-per-clip telemetry vs visible object count

### RCM-07 · Person identity fragmentation / merge
- **Root cause:** re-association score bar of 0.35 is low; 200 px / 45-frame window is generous for a 1920-px frame
- **Current behaviour:** the one-uid-per-frame claim prevents same-frame merges; cross-frame merges are possible
- **Desired behaviour:** report an identity confidence with each association; below the bar the pair is `UNCERTAIN` and cannot confirm
- **Files:** `person_identity.py`
- **Repair:** P2-1
- **Risk:** ◆
- **Validation:** `IMG_5118` (multi-person selective) — correct actor, every run

### RCM-08 · Premature / irreversible confirmation
- **Root cause:** RC-8; `mem.emitted = True` at confirm
- **Repair:** P1-2 · **Risk:** ◆ · **Validation:** all frozen positives keep confirming within `Δt = 3 s`

### RCM-09 · Duplicate events
- **Root cause:** RC-7 + tier OR + per-pair emission
- **Current behaviour:** up to 9 confirmations per real incident `[RUNTIME]`
- **Desired behaviour:** one incident key = `(actor_uid, object_uid, temporal episode)`; one verdict
- **Files:** `adaptive_tuner.py` `_is_duplicate`/`_accept`, `backend/routers/analysis.py` clustering, live-camera POST path
- **Repair:** P2-2
- **Risk:** ▼
- **Validation:** temporal FP count on the frozen set must fall from 20 toward ≤ 2

### RCM-10 · Timestamp causality violations
- **Root cause:** RC-9 (departure survives invalidation); `or`-fallback assignment of `ground_ts`/`departure_ts` on missing ticks
- **Desired behaviour:** `carry ≤ release ≤ ground ≤ confirmation`, and `ground ≤ recovery ≤ disposal`, asserted **before** persistence
- **Repair:** P0-3, P2-3
- **Risk:** ▼
- **Validation:** an invariant assertion over every event in `evidence_store/analysis/*/`

### RCM-11 · Learning-state contamination
- **Root cause:** RC-5
- **Current behaviour:** production runs on thresholds saturated toward recall by 95 uncurated videos
- **Desired behaviour:** offline, verdict-driven, symmetric, frozen-eval-gated, immutably pinned
- **Files:** `adaptive_tuner.py` `LearningStore`/`LEARNING_STEPS`, `learning/*.json`
- **Repair:** P0-5 (freeze now), P3-1 (rebuild)
- **Risk:** ▲ — reverting to YAML defaults **will** cost recall on brittle clips; must be measured, not assumed
- **Validation:** frozen eval at pinned vs YAML thresholds, side by side, before choosing the baseline

### RCM-12 · Relaxed tier overturns a correct strict rejection
- **Root cause:** RC-6
- **Repair:** P0-6 · **Risk:** ◆ · **Validation:** per-confirmation `adaptive_tier` telemetry; measure recall loss if tiers are disabled

### RCM-13 · Dashboard / evidence mismatch
- **Root cause:** F11 (no UNCERTAIN lane) + the evidence clip is built from the accused object's track
- **Current behaviour:** the operator sees a violation card whose evidence crop is a dumpster
- **Desired behaviour:** only `FINAL_VIOLATION` reaches the violations page; `VALID_DISPOSAL` / `RECOVERED` / `UNCERTAIN` go to a review lane
- **Repair:** P2-4 · **Risk:** ▼

### RCM-14 · Detector misses / true false negatives
- **Root cause:** `best.pt` + `garbage_bag_v2.pt` class coverage; HSV promotion band `[0.003, 0.12]` of person area
- **Note:** on `M.MOV` the real bag **was** detected by HSV but never promoted (outside the carry band while being *raised*) — a promotion-geometry gap, not a detection gap `[RUNTIME]`
- **Repair:** P4-1 (training only if H1 and this gap are measured first)

---

## 8. CURRENT ARCHITECTURE (as built)

```
frame
 └─ YOLO ×3 (person / best.pt / garbage_bag_v2.pt)  +  HSV colour  +  novelty
     └─ pooled dedup (IoU 0.5 | containment 0.7, source rank yolo<color<novelty)
         └─ promote_handheld_color_class()      ← colour → semantic, carry-band + area band
             └─ ByteTrack  → raw track ids  (CHURN)
                 ├─ PersonIdentityManager   raw → person_uid
                 └─ ObjectIdentityManager   det  → object_uid      ← 220px + class family
                     └─ _is_semantic_waste()  ← novelty + unpromoted colour dropped
                         └─ _evaluate_pair()  (per person × per semantic bag)
                             │   size veto · near · carried · stationary* · ground plane
                             │   *stationary + motion sync read RAW-ID history  ← RC-1
                             └─ _select_primary_associations()  ownership, 1 bag/person
                                 └─ pair key (person_uid, object_uid) → _PairMemory
                                     └─ _advance_pair()  8-state FSM
                                         └─ _evaluate_confirmation() → _rejection_reason()
                                             └─ LitteringEvent
                                                 └─ ×3 tiers → OR → dedup(raw pid) ← RC-6/7
                                                     └─ EvidenceManager (snapshot+clip)
                                                         └─ backend cluster dedup
                                                             └─ Event(status="confirmed")
                                                                 └─ Violations page
```

**States:** `NO_BAG → BAG_NEAR_PERSON → BAG_CARRIED → BAG_RELEASED → BAG_ON_GROUND → PERSON_DEPARTED → VIOLATION_CONFIRMED`, with `PICKED_BACK_UP` as a finalization-time reclassification only.

**Release** is a disjunction of **six** geometric criteria (`crit_distance`, `crit_static`, `crit_feet`, `crit_ground`, `crit_standing_ground`, `crit_desync`, `crit_held_static`) OR an AIDM wrist-separation latch. Each was added to recover a specific clip. Their union has never been validated against the negatives as a whole. `[CODE]`

---

## 9. REQUIRED SEMANTIC ARCHITECTURE

### 9.1 The target state model

```
                    ┌──────────────────┐
                    │ NO_INTERACTION   │
                    └────────┬─────────┘
                             │ object near actor
                    ┌────────▼─────────┐
                    │ NEAR_OBJECT      │
                    └────────┬─────────┘
                             │ AGENCY PROVEN (hand link, or self-motion off-ground)
                    ┌────────▼─────────┐
              ┌────►│ CARRYING         │◄──────────┐
              │     └────────┬─────────┘           │ recovery proven
              │              │ hand/object separation cue
              │     ┌────────▼─────────┐           │
              │     │ POSSIBLE_RELEASE │           │
              │     └────────┬─────────┘           │
              │              │ sustained over release window
              │     ┌────────▼─────────┐           │
              │     │ RELEASE_CONFIRMED│           │
              │     └────┬────────┬────┘           │
              │          │        │                │
              │  container hypothesis   ground hypothesis
              │          │        │                │
              │  ┌───────▼──┐ ┌───▼───────────────────────┐
              │  │DISPOSAL_ │ │ POSSIBLE_GROUND_CONTACT   │
              │  │PENDING   │ └───┬───────────────────┬───┘
              │  └───┬──────┘     │                   │
              │      │            │ settled           │ recovered
              │      │      ┌─────▼───────┐     ┌─────▼─────┐
              │      │      │ ABANDONED   │     │ RECOVERED ├──┘
              │      │      │(PROVISIONAL)│     └───────────┘
              │      │      └─────┬───────┘
              │      │            │ actor leaves / horizon
              │      │      ┌─────▼──────────┐
              │      │      │ PERSON_DEPARTED│
              │      │      └─────┬──────────┘
              │      │            │  ═══ DECISION HORIZON ═══
              │ ┌────▼─────────┐  │
              └─┤VALID_DISPOSAL│  │
                └──────────────┘  │
                          ┌───────▼────────┐
                          │ FINAL_VIOLATION│   │ REJECTED │   │ UNCERTAIN │
                          └────────────────┘
```

### 9.2 Current FSM vs required — what changes, what does not

| Current | Required | Change |
|---|---|---|
| `NO_BAG` | `NO_INTERACTION` | **rename only** |
| `BAG_NEAR_PERSON` | `NEAR_OBJECT` | **rename only** |
| `BAG_CARRIED` | `CARRYING` | **entry gate changes**: agency required (RCM-01). Everything else kept. |
| — | `POSSIBLE_RELEASE` | **new**: the six release criteria produce a *candidate*; only a sustained candidate becomes `RELEASE_CONFIRMED`. Turns a disjunction-of-heuristics into a debounced hypothesis. |
| `BAG_RELEASED` | `RELEASE_CONFIRMED` | kept |
| — | `DISPOSAL_PENDING` | **new** |
| `BAG_ON_GROUND` | `POSSIBLE_GROUND_CONTACT` → `ABANDONED` | **split**: contact ≠ abandonment. Contact requires resting-on-ground observation; abandonment requires contact **plus** a non-recovery window. |
| `PICKED_BACK_UP` (finalize-only) | `RECOVERED` (real state) | **promote to a live terminal state** |
| — | `VALID_DISPOSAL` | **new terminal** |
| `PERSON_DEPARTED` | `PERSON_DEPARTED` | kept, but no longer implies confirmation |
| `VIOLATION_CONFIRMED` | `FINAL_VIOLATION` | **gated by the decision horizon**, not by a timer |
| (rejection reasons) | `REJECTED` / `UNCERTAIN` | `UNCERTAIN` becomes a *first-class outcome*, not an unpersisted rejection |

**Explicitly unchanged:** pair keying by `(person_uid, object_uid)`; ownership lock after carry; one-bag-per-person; the six release criteria themselves; the AIDM wrist gate; auto-calibration; evidence generation; the ground-plane margin geometry; determinism plumbing.

---

## 10. IDENTITY STRATEGY

**Hard invariants.**

- **I1** An object uid must never be reassigned to a physically different object. Enforcement: class-family (exists) **+ size-ratio continuity** (new, `min(a,b)/max(a,b) ≥ 0.5` on bbox area) **+ velocity plausibility** (new, jump ≤ `max_object_jump_ratio · person_height` — the check exists in `_rebind_continuity_ok` but only on the pair-rebind path, not inside `ObjectIdentityManager`).
- **I2** An actor uid must not fragment across a littering arc, nor merge two people. Enforcement: the existing one-uid-per-frame claim (keep) + raise the re-association acceptance bar from 0.35 and record the achieved score as `actor_identity_confidence` on the event.
- **I3** **All temporal cues must read the stable uid, never the raw track id.** This is the single highest-value identity change (RC-1): `_bag_history` must be keyed by `object_uid`, with the raw id kept only for rendering.
- **I4** Identity uncertainty is an outcome, not a silent default. When I1 or I2 is violated mid-arc, the arc becomes `UNCERTAIN` and cannot produce a violation — but it also must not be silently discarded; it is persisted to the review lane.

**Allowed gaps — start from measurement, not guesswork.** `max_track_gap_frames` is currently 6 ticks (0.75 s @ 8 fps) and `max_object_jump_ratio` is 1.5 person-heights. Neither appears in `config/events.yaml`, so neither has been tuned against data. **Measure** the distribution of real track gaps and inter-tick jumps across the 11 frozen clips before setting them (§18.3, M-1).

**Do not over-constrain.** Occlusion behind a passing vehicle routinely exceeds 0.75 s. The correct response to a long gap is `UNCERTAIN`, not a fresh uid that silently inherits the arc.

---

## 11. RELEASE STRATEGY

**Problem:** seven independent ways to declare a release, each tuned to one clip, evaluated as an OR on a single tick.

**Design:**

1. Each criterion produces a **weighted vote**, not a verdict. Distance-based release (the only one requiring actual separation growth) carries the highest weight; `crit_held_static` and `crit_standing_ground` (which fire while the person stands still) carry the lowest.
2. A release candidate requires the weighted sum over a `release_window_frames` window to exceed a bar — replacing "any one criterion on any one tick."
3. **Agency precondition:** no release may fire unless the pair's `CARRYING` entry was itself agency-proven (§10 / RCM-01). A release from a never-truly-carried object is not a release.
4. **Latch freshness:** the AIDM separation latch keeps its existing staleness rule (`_separation_stale`), which is correct as written.
5. `POSSIBLE_RELEASE → RELEASE_CONFIRMED` requires the candidate to survive the window **and** the object to still be the same uid.

**Protects:** every current true positive, because the high-weight criteria are exactly the ones firing on real drops. **Risks:** clips that only ever satisfied one low-weight criterion. Those must be identified *before* the change by instrumenting which criterion fires per frozen clip (§18.3, M-2).

---

## 12. GROUND / RECOVERY STRATEGY

**GROUND CONTACT ≠ ABANDONMENT.**

- `POSSIBLE_GROUND_CONTACT` requires the object, under its stable uid, observed resting on the actor's ground plane on **live** ticks. Synthetic credit from `_release_pose_on_ground()` may support *continuity*, but must **not** by itself satisfy the ground requirement for a violation. (Today it can, and does — RC-4.)
- `ABANDONED` requires ground contact **plus** no recovery for the non-recovery window **plus** the actor still observable — so "no recovery" is a real observation, not an absence of data.
- `RECOVERED` requires the **same** object uid returning to a hand/carry association. Today's recovery bar (strong lift ≥ 0.25 · ph, sustained ≥ 2 ticks, AIDM grip) is *stricter* than the abandonment bar (a 5-tick timer). **These must be symmetric:** it must not be easier to accuse than to exonerate.

**Window length — measure, do not guess.** Required measurement (§18.3, M-3): across all frozen clips plus the two phone videos, extract the distribution of (a) time from ground contact to recovery in true recovery cases, and (b) time from ground contact to the actor leaving frame in true litter cases. The non-recovery window is set at a documented percentile of (a) — not at a round number. Until measured, the plan carries the window as `TBD-M3`.

---

## 13. VALID DISPOSAL STRATEGY

**Definition — general, camera-agnostic, no filenames, no fixed coordinates:**

> A `VALID_DISPOSAL` holds when an object whose agency was established with an actor ceases to be observable on the ground plane, and the last reliable observations of that object are spatially and temporally consistent with entry into a container region.

**Evidence, classified:**

| Evidence | Class | Notes |
|---|---|---|
| Container **region** established (detected container, or operator zone, or persistent static container-scale object) | **Necessary** — at least one source | Without any container hypothesis there is no disposal claim. Today only the operator zone exists, and it is empty by default. |
| Object's last observed trajectory directed **toward/into** the region | **Necessary** | The `M.MOV` signature: y 594→522, *rising*, toward the container. Cheap to compute; already in the frame log. |
| Object **not** subsequently observed on the ground plane | **Necessary** | Already computable (`live_ground_ticks == 0`). |
| Object disappears (occlusion into/behind the container) rather than settling | Strong supporting | Distinguishes disposal from a drop beside the bin. |
| Actor's hand rises above the container rim before separation | Strong supporting | Needs pose + container top edge. |
| Actor departs empty-handed (no carried object re-acquired) | Supporting | |
| Post-disposal absence sustained for the horizon | Supporting | |
| Object/actor identity continuity across the whole arc | **Necessary** | Without it the claim is `UNCERTAIN`, not `VALID_DISPOSAL`. |

**Container region sources, in confidence order:**

1. Operator-configured `bin_zones` — highest confidence, static cameras only, already implemented.
2. A detected container class — **does not exist today**; requires either adding container classes to the detector or a COCO-based proxy.
3. A *derived* static-container hypothesis: a semantic-waste-class object that is stationary from first sighting, container-scale, and persistent across the clip.

Source (3) is available **immediately with no new model** and would have covered both phone videos.

**Incomplete evidence:** container hypothesis present but trajectory ambiguous → `UNCERTAIN`, review lane, **no violation**. This is the key asymmetry: a plausible disposal blocks accusation even when it cannot be proven. Accusing requires proof; declining to accuse does not.

**Critical regression guard:** `IMG_5290` is a real ground drop *in a dumpster scene*. Any container hypothesis must be **spatially specific** — the object's terminal position must be consistent with the container region, not merely "a container is in frame" — or `IMG_5290` and `IMG_5295`, both explicitly annotated "dumpster in scene / past dumpster", will regress into false negatives.

---

## 14. FINAL DECISION / TEMPORAL HORIZON

Two-phase commit:

```
candidate → PROVISIONAL verdict → observation window → FINAL verdict → persistence
```

- A provisional verdict is computed exactly as today, is **visible in telemetry**, and is **never persisted as a violation**.
- The observation window stays open for `decision_horizon_ticks` (value = `TBD-M3`) or until the actor is no longer observable — whichever is **later** for recovery evidence and **earlier** for departure evidence.
- During the window, arriving evidence may move the verdict: recovery → `RECOVERED`; container-consistent disappearance → `VALID_DISPOSAL`; identity break → `UNCERTAIN`.
- Only when the window closes is a **final** verdict emitted, exactly once.

**End of stream.** A clip ending inside the window is the hard case, and the current `_apply_end_of_stream_walkaway_credit()` resolves it by *manufacturing* departure, stationarity and abandonment evidence (`max(..., cfg.min_*)`) — i.e. by assuming guilt when the data runs out. That is backwards. Required policy: **a horizon that cannot close resolves to `UNCERTAIN`, not to `VIOLATION`.** This will cost recall on short clips (`IMG_5117` at 3.47 s is the one to watch) and that cost must be measured before acceptance, not assumed away.

**Only `FINAL_VIOLATION` reaches the primary violations dashboard.** `VALID_DISPOSAL`, `RECOVERED`, `REJECTED` and `UNCERTAIN` are persisted to a review lane with full evidence, because they are the training signal for §17.

---

## 15. EVENT LIFECYCLE

**Incident key:** `(actor_uid, object_uid, episode_id)`, where `episode_id` increments when the same pair begins a *new* carry arc after a terminal outcome. Spatial continuity is already implicit in uid continuity.

**One incident = one verdict = one row.** Enforcement points, in order:

1. **Detector:** one `_PairMemory` per key already emits at most one event (`mem.emitted`). Correct today.
2. **Tier arbitration:** must key on `event_actor_person_uid` + `event_object_uid`, **not** `ev.person_track_id` (RC-7), and must arbitrate to a **single** verdict rather than emitting each tier's confirmation (RC-6).
3. **Backend:** the existing cluster dedup stays as defence in depth, and the same clustering must be added to the **live-camera POST path**, which has none.
4. **Database:** a uniqueness constraint on `(analysis_job_id, event_actor_person_uid, event_object_uid, episode)` makes duplication impossible rather than merely unlikely.

---

## 16. TIMESTAMP INVARIANTS

Enforced **before persistence**, on every event, confirmed or rejected:

```
carry_start ≤ release ≤ ground ≤ confirmation
ground      ≤ recovery
release     ≤ disposal
```

plus: every present timestamp lies within the clip duration; a state's timestamp is present iff the arc reached that state.

**Violation policy:** an event failing an invariant is **not persisted as a violation**. It is persisted as `UNCERTAIN` with the failing invariant named in `details`, and logged at ERROR. Silently repairing timestamps would hide the identity bug that produced them.

Known live source of violations: `_invalidate_stale_release()` preserves `departure_frame/ts` across invalidation (F7 / F9). Fixing RCM-05 removes the cause; the invariant check is the net beneath it.

---

## 17. LEARNING ARCHITECTURE

### 17.1 What exists today

**Online (`adaptive_tuner.LearningStore`) — the problem.**

- *Learns:* one-directional relaxation steps for 6 rejection reasons, applied to tier 0 of every subsequent video.
- *Does not learn:* anything from a confirmation being wrong. There is no feedback path from an operator verdict, a frozen-eval FPR, or a false positive into this store.
- *How it changes inference:* `resolve_inference_overrides()` → `build_tier_configs()` → tier-0 config. Under `deterministic=True` (uploads) it reads `learning/inference_pin.json`; under live camera it reads the **live, mutating** `learning/learning.json`.
- *Persistent state:* `learning/learning.json` (95 videos, mutable) and `learning/inference_pin.json` (the pin).
- *Pinned:* the pin is written only by `promote_learning_pin()` — but that function performs **no evaluation and no gate**. It is a straight copy of live → pin. The pin is a *snapshot mechanism*, not a quality gate.
- *Poisoning:* a single video that produces N rejection reasons advances N reasons by one step each, permanently, with no review. 95 such videos have saturated 5 of 6 reasons. `[RUNTIME]`

**Offline (`learning_offline/`) — sound; keep it.** Operator verdict → dataset (50/50 replay) → fine-tune (`freeze=10`, `lr0=0.001`) → frozen eval → gate (`F1↑ AND FPR ≤ 1.1×`) → versioned atomic weight promotion. This is the correct shape. It governs **weights only**, never thresholds.

### 17.2 Target architecture

```
REAL VIDEOS
   └─ operator verdict  (confirm / reject / valid-disposal / recovered / uncertain)
       └─ EXPERIENCE DATASET   (immutable, versioned, hash-addressed per clip)
           └─ OFFLINE LEARNING
               ├─ detector weights      (existing learning_offline path — keep)
               └─ threshold candidate   (NEW — symmetric search, both directions)
                   └─ FROZEN EVALUATION (same harness, both objectives)
                       └─ PROMOTION GATE   F1↑ AND temporal-FPR ↓-or-flat
                           └─ IMMUTABLE PRODUCTION PIN  (hash-stamped in the report)
```

**Rules:**

- No production run may mutate any file that a later production run reads. The live-camera path's read from the mutating `learning.json` must be closed.
- Threshold candidates are searched **symmetrically**: a false positive tightens, a miss loosens, and the gate arbitrates on *both* objectives together.
- `promote_learning_pin()` must require a passing gate decision, not merely be invoked.
- Every emitted event already carries `details.active_thresholds`; extend it with the pin hash so any decision is reproducible from the artifact alone.

---

## 18. TEST MATRIX

### 18.1 Scenario coverage vs today's frozen set

| Scenario | Frozen coverage today | Action |
|---|---|---|
| TRUE LITTER | `IMG_5117`, `IMG_5119`, `IMG_5290`, `IMG_5295`, `IMG_5306`, `A` | keep |
| VALID BIN DISPOSAL | `IMG_5305` (negative) | **add `M.MOV` as `VALID_DISPOSAL`** after P1-1 lands (label correction, version bump to v1.1) |
| TEMPORARY GROUND + RECOVERY | **none** | **record** — the largest coverage hole |
| CARRY ONLY | `IMG_5115` | keep |
| PRE-EXISTING LITTER | `IMG_5303` (weak) | **add `IMG_5613`** |
| STATIC CONTAINER | **none as a labelled scenario** | **add `M.MOV`, `IMG_5613`** |
| OBJECT TRACK DROPOUT | unit only (`test_job57_stale_tracking`) | keep unit; add a real clip |
| OBJECT REBIND | unit only | keep |
| PERSON RE-ID | `IMG_5118` | keep |
| MULTI-PERSON | `IMG_5118` | keep |
| MULTI-OBJECT | **none** | record |
| DUPLICATE EVENT | measured, not labelled | becomes a **metric**, not a clip |
| SHORT VIDEO | `IMG_5117` (3.47 s) | keep — the horizon regression canary |
| OCCLUSION | **none** | record |
| FRAME EDGE | `A` (late entry) | partial |
| LOW CONFIDENCE / LOW LIGHT / SMALL OBJECT | **none** | record |

### 18.2 Metrics

| Metric | Today | Target | Source |
|---|---|---|---|
| Clip precision | 0.889 | ≥ 0.95 | `run_frozen_eval` |
| Clip recall | 1.000 | ≥ 0.90 (no worse than −1 clip) | `run_frozen_eval` |
| **Temporal precision** | **0.286** | **≥ 0.85** | `run_frozen_eval` |
| Temporal recall | 1.000 | ≥ 0.90 | `run_frozen_eval` |
| FPR on hard negatives | 1.0 (1 of 1 bin negative fails) | 0.0 | `run_frozen_eval` |
| Confirmations per real incident | up to 9 | exactly 1 | **new metric** |
| Event timing error | `mean_latency` reported | within `Δt = 3 s` | existing |
| **Actor accuracy** | not measured | 100 % on `IMG_5118` | **new** — requires per-clip actor GT |
| **Object accuracy** | not measured | 100 % | **new** — requires per-clip object GT; this is the metric that would have caught `M.MOV` |
| Evidence validity | file existence only | crop contains the accused object | **new** |
| Reproducibility | `check_determinism_3runs.py` exists | 3× identical verdicts + identical pin hash | existing harness |

**Object accuracy is the most important new metric in this plan.** Every current metric scored `M.MOV` as a *correct* prediction while the system was accusing a dumpster.

### 18.3 Required measurements before any threshold is chosen

- **M-1** Track-gap and inter-tick-jump distributions per clip → `max_track_gap_frames`, `max_object_jump_ratio`.
- **M-2** Which release criterion fires, per clip, per confirmation → release vote weights.
- **M-3** Ground-contact→recovery and ground-contact→actor-exit distributions → non-recovery window, decision horizon.
- **M-4** Per-confirmation `adaptive_tier` → how much recall the tier ladder actually buys.
- **M-5** Frozen eval at pinned thresholds vs YAML defaults, side by side → the true cost of freezing the ratchet.
- **M-6** (H1) Detector-only container classification survey.

None of these require a code change to the decision path; M-2 and M-4 need telemetry fields only.

---

## 19. REPAIR PRIORITY

### P0 — Stop false semantic accusations at the identity / release / ground level

**P0-0 · Measurement pass (M-1…M-6).**
*Why:* every P0 threshold below is otherwise a guess. *Where:* `evaluation/`, `scripts/`. *Test:* none (read-only). *Risk:* none. *Rollback:* n/a.

**P0-1 · Key all temporal object cues by stable uid.**
*Why:* RC-1, the single mechanism behind RCM-01. *What:* `_bag_history` keyed by `object_uid`; `_is_stationary()` and the motion-sync cue read it. *Where:* `littering_event_detector.py:1666-1722`, `:1770-1776`. *Test:* new unit — a static object under 3 churning track ids must read `stationary=True` throughout. *Metric:* `M.MOV` / `IMG_5613` no longer reach `CARRYING`. *Risk:* ◆ — may change stationarity on genuine bags mid-churn. *Rollback:* single-commit revert; the raw-id path is retained behind a flag for one release.

**P0-2 · Agency precondition for CARRYING + first-seen-static veto.**
*Why:* RC-3, RCM-01, RCM-04. *What:* (a) an object stationary from first sighting under its stable uid and never wrist-associated can never become an event object, regardless of confidence; (b) the no-pose carry-origin fallback requires **self-motion of the object's own uid**, not motion correlation with a walker. *Where:* `update()` pair-creation gate `:1385-1398`; `_advance_pair` carry gate `:2380-2400`. *Test:* `IMG_5613`, `IMG_5303`, `M.MOV` → no confirmations; all 8 positives keep confirming. *Risk:* ◆. *Rollback:* config flag `require_object_agency`.

**P0-3 · Complete release invalidation + causal timestamps.**
*Why:* RCM-05, RCM-10. *What:* `_invalidate_stale_release()` additionally resets `ground_evidence_frames`, `bin_zone_frames`, `separated_frames`, `max_post_release_norm_distance`, `abandonment_frames`, `departure_frame/ts`, `feet_release_streak`, `desync_release_streak`; and `release_invalid` clears on a fresh, continuity-valid release (which also fixes F8's recall loss). *Where:* `:2054-2080`. *Test:* `tests/test_job57_stale_tracking.py` A–G unchanged + a new test asserting no stale ground survives. *Risk:* ▼ precision / ◆ recall on P0-3b. *Rollback:* separable into 3a (clear more) and 3b (clear the flag).

**P0-4 · Object-identity continuity gates.**
*Why:* RCM-06. *What:* size-ratio and velocity-plausibility gates inside `ObjectIdentityManager.update()`; below the bar, mint a fresh uid and mark the arc `UNCERTAIN`. *Where:* `object_identity.py`. *Test:* uid-per-clip telemetry. *Risk:* ◆. *Rollback:* gates behind config, default-off until M-1 lands.

**P0-5 · Freeze the learning ratchet.**
*Why:* RC-5 / RCM-11. *What:* disable `LearningStore.record()`'s step advancement in all production paths (keep the history log for analysis); close the live-camera read of the mutating store; **choose the baseline pin from M-5, not by assumption.** *Where:* `adaptive_tuner.py`. *Test:* frozen eval at both baselines; determinism 3×. *Risk:* ▲ recall. *Rollback:* restore the current pin file — it is version-controlled state.

**P0-6 · Tier arbitration, not disjunction.**
*Why:* RC-6 / RCM-12. *What:* extend the tier-0 veto to **every** rejection reason (a relaxed tier may never overturn a strict rejection of the same incident); key `_is_duplicate`/`_accept` on `event_actor_person_uid`. Decide from M-4 whether the ladder earns its keep at all. *Where:* `adaptive_tuner.py:562-660, 706-760`. *Test:* frozen eval; per-confirmation tier telemetry. *Risk:* ◆ recall. *Rollback:* `max_tiers=0`.

### P1 — Recovery and valid-disposal semantics

**P1-1 · `VALID_DISPOSAL` as a first-class outcome** (§13). Includes the derived static-container hypothesis (source 3), which needs no new model. *Risk:* ▲ — `IMG_5290` / `IMG_5295` regression watch. *Rollback:* feature flag.
**P1-2 · Two-phase commit + decision horizon** (§14), including replacing the end-of-stream guilt credit with `UNCERTAIN`. *Risk:* ◆ recall on short clips.
**P1-3 · `RECOVERED` as a live state with a bar symmetric to abandonment** (§12). *Risk:* ◆.
**P1-4 · Debounced release voting** (§11). *Risk:* ◆.

### P2 — Lifecycle, dedup, timestamps, dashboard

**P2-1** Person-identity confidence → `UNCERTAIN` below bar (RCM-07).
**P2-2** One incident = one row: uid-keyed arbitration, live-path clustering, DB uniqueness constraint (RCM-09).
**P2-3** Timestamp invariant enforcement before persistence (§16).
**P2-4** Review lane in the dashboard; only `FINAL_VIOLATION` on the violations page (RCM-13).

### P3 — Learning hardening

**P3-1** Symmetric, offline, verdict-driven threshold learning with a real gate (§17.2).
**P3-2** Pin hash recorded in every event's `details`.
**P3-3** Frozen set v1.1: `M.MOV` → `VALID_DISPOSAL`, add `IMG_5613`, add recorded recovery / multi-object / occlusion clips. **Only after P1-1 is validated** — changing labels before the architecture is repaired would hide the defect.

### P4 — Detector training

**P4-1** Only if M-6 / H1 proves container misclassification is a detector-layer problem that container-region reasoning cannot contain. The HSV promotion-band gap on `M.MOV` (real bag detected, not promoted, because it was *raised* out of the carry band) is a cheaper, more targeted fix and should be evaluated first.

---

## 20. FILE / MODULE IMPACT MAP

| File | Repairs | Nature |
|---|---|---|
| `littering_event_detector.py` | P0-1, P0-2, P0-3, P1-1..4, P2-3 | Core. Additive states + gate changes. **No rewrite.** |
| `inference/tracking/object_identity.py` | P0-4 | Add continuity gates |
| `inference/tracking/person_identity.py` | P2-1 | Expose association confidence |
| `adaptive_tuner.py` | P0-5, P0-6, P3-1 | Arbitration + learning |
| `learning/inference_pin.json` | P0-5 | Data (version-controlled) |
| `config/events.yaml` | P0-*, P1-* | New keys, documented defaults |
| `backend/routers/analysis.py` | P2-2, P2-3, P2-4 | Persistence + outcome lanes |
| `backend/models.py` | P2-2, P2-4 | `outcome` column, uniqueness constraint |
| `backend/routers/events.py`, `backend/schemas.py` | P2-4 | Expose outcome |
| `dashboard/src/pages/Violations.tsx` + new review lane | P2-4 | UI |
| `evaluation/run_frozen_eval.py`, `evaluation/metrics.py` | §18.2 | New metrics (object/actor accuracy, confirmations-per-incident) |
| `evaluation/frozen_test_set.v1.json` | P3-3 | → v1.1, after P1-1 |
| `learning_offline/*` | P3-1 | Extend to thresholds |
| `tests/` | all | New per-repair tests; the existing suite is the regression net |

**Untouched:** `inference/capture/*`, `inference/evidence/*`, `inference/visualization/*`, `inference/pose/*`, `inference/detection/*` (unless P4 triggers), `scripts/*`, determinism plumbing.

---

## 21. ROLLBACK STRATEGY

Per-group discipline, non-negotiable:

```
CHECKPOINT (tag + frozen-eval baseline JSON + determinism 3×)
  └─ ONE REPAIR GROUP
      └─ unit tests            → must be green
          └─ frozen eval        → compare against the checkpoint baseline
              └─ real phone videos (M.MOV, IMG_5613)
                  └─ ACCEPT (tag) │ ROLLBACK (revert the group, whole)
```

- Every group is one commit or one revertible commit range. **No stacking a patch on a regressed group.**
- Every behavioural group ships behind a config flag defaulting to the *current* behaviour for one cycle, so rollback is a config change before it is a revert.
- Acceptance requires: temporal precision **not worse**; clip recall **not worse by more than one clip** (and if it is, the lost clip is named and the trade explicitly accepted); determinism intact.
- `learning/inference_pin.json` is version-controlled state — rolling back thresholds is a file restore.

---

## 22. DEFINITION OF DONE

Validated on real data, not asserted:

1. `PERSON → GROUND → ABANDON → DEPART` → `FINAL_VIOLATION` — all 8 frozen LITTER clips.
2. `PERSON → CONTAINER` → **no violation** — `IMG_5305`, `M.MOV` (the latter as `VALID_DISPOSAL`).
3. `PERSON → GROUND → SAME OBJECT RECOVERED` → **no violation** — on a recorded recovery clip (must be recorded; none exists).
4. `PRE-EXISTING LITTER + PASSER-BY` → **no violation** — `IMG_5613`, `IMG_5303`.
5. Insufficient evidence → `UNCERTAIN`, review lane, **no accusation**.
6. Object accuracy = 100 %: every confirmed violation's object is the object the person physically handled.
7. Actor accuracy = 100 % on `IMG_5118`.
8. One physical incident = exactly one persisted event. Temporal precision ≥ 0.85.
9. Timestamp invariants hold on every persisted event; violations route to `UNCERTAIN`.
10. 3× replay of every frozen clip yields identical verdicts and an identical pin hash.
11. No production path mutates state that a later production run reads.
12. No video-specific patch exists: no filename, frame index, or fixed pixel coordinate appears in any decision path. *(The current code is already clean of filename checks — clip names appear only in comments and config commentary. That property must be preserved.)*
13. Every accepted repair group has a recorded before/after frozen-eval comparison.

---

## 23. REMAINING UNKNOWNS

1. **The true cost of freezing the learning ratchet.** 87 recorded `NO_RELEASE_TRANSITION` occurrences drove release sensitivity to its floor. How much of current recall depends on that? **M-5 answers this, and must run before P0-5 is accepted.**
2. **Whether the tier ladder earns its keep.** If tier 0 confirms nearly everything, the ladder is pure precision cost. **M-4.**
3. **Why `IMG_5118` flipped between runs.** Either config drift between runs or genuine non-determinism. **H2.**
4. **Whether `garbage_bag_v2.pt` reliably mislabels containers.** **H1 / M-6** — determines whether P4 is needed at all.
5. **No recovery clip exists.** The `GROUND → RECOVERED` path — one of the three Definition-of-Done distinctions — **cannot currently be validated on real data.** This must be recorded before P1-3 can be accepted. It is the largest single gap in the validation set.
6. **Real-world event rate.** "False positives per hour" cannot be computed without continuous footage; the frozen set is clip-based. A continuous recording is needed for that metric.
7. **Operator-verdict volume.** P3-1's symmetric learning needs labelled verdicts. How many exist in the DB today is unmeasured.
8. **Ground-plane geometry on non-static cameras.** All ground reasoning assumes the actor's feet line is a usable ground proxy. Phone footage with camera motion may break this assumption; not yet measured.

---

## APPENDIX A — FORENSIC ANSWERS FOR THE LATEST FALSE-POSITIVE CASES

| # | Question | `M.MOV` (job 57) | `IMG_5613` (job 50) |
|---|---|---|---|
| 1 | Object the person actually handled | Small yellow bag | A clear water bottle (never discarded) |
| 2 | Object the system thought was handled | The **green dumpster** (`Garbage Bag`, tracks 10006/10011/10014) | A static container / pre-existing sack |
| 3 | Same physical object? | **No** | **No** |
| 4 | When did object identity become uncertain? | It was never correct. The real bag existed only as `color_candidate_yellow` (f181–f213) and was never promoted to semantic; a container region was tracked as the object from f185. | Never correct — a static object was the event object from the start |
| 5 | When did person identity become uncertain? | It did not. Person identity behaved correctly. | It did not. |
| 6 | When did stale state begin affecting decisions? | Not the primary cause here; the arc was built on live (but wrong) observations. | Same |
| 7 | What caused RELEASE? (f206) | The person walked on past the static container; the association's geometry changed while the container stayed put — read as a put-down | Same pattern |
| 8 | What caused GROUND? (f217) | The container's bbox bottom sits at/below the actor's ground line, so `near_ground_plane` was true on every tick | Same |
| 9 | What caused DEPARTURE? | The person kept walking; `abandonment_frames` reached the learned `min_abandonment_frames = 5` | Same |
| 10 | Why was VALID_DISPOSAL not recognised? | It cannot be. No `VALID_DISPOSAL` state, no container detector, `bin_zones` empty — and the real disposal object never entered the FSM | Nothing to dispose; no "no action" outcome exists either |
| 11 | When did it become irreversible? | f241, `mem.emitted = True` at `VIOLATION_CONFIRMED` | at confirmation |
| 12 | What later evidence contradicted it? | The actor walking away empty-handed with nothing new on the ground (t = 6.83 s onward); the container never moving | The bottle never leaving the hand; nothing changing on the ground |
| 13 | Why could later evidence not correct it? | Confirmation is terminal; `PERSON_DEPARTED` explicitly refuses reclaim; and "the object never moved" is not evaluated at any point | Same |
| 14 | Local bug or architectural gap? | **Architectural.** Three independent gaps: no object-agency requirement, no container semantics, no reversible decision horizon. | **Architectural**, the same three |

**The decisive fact:** on `M.MOV` the system produced a semantically perfect littering narrative — carry (f94) → release (f206) → ground (f217) → departure (f242) → violation (f241), confidence 0.924 — about an object that never moved, while the actual disposal happened in full view and was invisible to the decision engine. No threshold change can fix a story told about the wrong object.

---

*End of plan. No code, config, threshold, FSM or frozen-set change was made in producing this document.*
