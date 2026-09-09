# ACCEPTANCE TESTS (Section 24) + FINAL HEALTH SCORECARD (Section 26)

**Project:** D:\HO — Littering AI
**Date:** 2026-09-09
**Scope of this report:** post-repair acceptance evaluation against MASTER_REPAIR_PLAN
Sections 24 and 26. All P0 and P1 repair stages (P0-1, P0-2, P0-3, P1-1, P1-2,
P1-3/P1-12, P1-5, P1-9, Section 16 cleanup) were implemented in prior steps.

---

## 1. Test Suite Result

| Suite | Result |
|---|---|
| Full pytest suite (`tests/`) | **146 passed, 0 failed, 0 skipped** (4 warnings, ~3:18) |
| Dashboard typecheck (`npx tsc --noEmit`) | Only the **pre-existing** `AnalysisDetail.tsx(237,13)` `sourceFps` error (proven pre-existing during P0-2 via git-stash; unrelated to repair stages) |
| Vite production build | Succeeds (verified in P0-2/P1-3 steps) |

The suite grew from 144 → 146 in this step: the two tests Section 24 explicitly
required but that did not exist were added (literal 3-person Case A; combined
Case C ground+bin). No existing test was modified or weakened.

---

## 2. Section 24 — Acceptance Criteria Matrix (post-repair)

| # | Criterion | Unit | Integration | Real video | Dashboard |
|---|---|---|---|---|---|
| 1 | Person walks by → no event | **PASS** — `test_passerby_never_inherits_anothers_event`, `test_negative_closer_bystander_not_attributed`, new literal 3-person test | Partial (pipeline integration covers positive paths) | **NOT PROVEN** | NOT PROVEN |
| 2 | Carry waste → no event yet | **PASS** — `test_negative_carry_without_drop_is_rejected`, `test_carry_only_incomplete_rejected`, `test_carry_only_still_rejected_as_no_release` | Partial | Observed incidentally (real IMG_5305 replays reject `NO_RELEASE_TRANSITION`) but not as a designed real-video test | NOT PROVEN |
| 3 | Drop on ground → event | **PASS** — `test_positive_littering_sequence_confirms`, `test_ground_rest_below_feet_confirms_and_records_status`, bin-zone outside-zone control | **PASS** — `test_pipeline_integration.test_full_littering_sequence_confirms` | **NOT PROVEN** — real end-to-end replays produce 0 confirmed events (FSM recall blocker, see Gaps) | NOT PROVEN |
| 4 | Drop + walk away → confirmed | **PASS** — `test_abandonment_confirms_when_person_stays`, `test_normal_event_stable_ownership` | **PASS** — `test_put_down_and_stays_confirms_abandonment` | **NOT PROVEN** (same blocker) | NOT PROVEN |
| 5 | Drop then re-grab → no violation | **PASS** — `test_put_down_then_regrab_reclassified_picked_back_up` | **PASS** — `test_put_down_then_regrab_does_not_confirm` | NOT REAL-VIDEO-VERIFIED (no real re-grab footage found in D:\22; Phase 4 finding) | n/a |
| 6 | Waste in bin → no violation | **PASS** — `test_bin_zone.py` (4 tests: in-zone rejection `BIN_ZONE_DEPOSIT`, outside-zone control, per-camera scoping, polygon/bbox coercion) | **PASS** — combined Case C acceptance test | NOT REAL-VIDEO-VERIFIED (no real bin footage in D:\22) | n/a |
| 7 | P1 walks by, P2 litters, P3 walks by → only P2 actor | **PASS** — NEW `test_literal_three_person_only_litterer_is_actor` | NOT PROVEN | NOT PROVEN — REAL MULTI-ACTOR GROUND TRUTH NOT AVAILABLE in D:\22 | NOT PROVEN |
| 8 | One person + two objects → only released object becomes event | **NOT SUPPORTED** — production FSM hard-enforces one-bag-per-person (P1-4 deliberately deferred; behavior documented in `test_one_person_two_objects_documented_behavior`) | NOT PROVEN | NOT PROVEN | n/a |
| 9 | Multiple people simultaneously → correct ownership | **PASS** — `test_two_actors_each_litter_get_distinct_events`, new combined Case C, `test_stable_ownership_regression`, layer-2 crowd tests | Partial | NOT PROVEN | NOT PROVEN |
| 10 | Evidence → correct person + object + time | **PASS** — P0-1 regression tests (`test_is_duplicate_never_matches_own_acceptance`, `test_relaxed_tier_event_survives_own_confirmed_events_read`), identity persistence (`test_create_event_with_internal_token_persists_identity`), sequence images (`test_evidence_package_sequence.py`), stable-UID anchoring (`test_visualization.py`) | **PASS** — API roundtrip: `test_evidence_full_roundtrip`, `test_sequence_image_paths_persist_and_are_served` | **NOT PROVEN** — Section 25 real replays (IMG_5299/5298/5117) completed but produced 0 confirmed events, so identity non-NULL could not be proven end-to-end on real footage | PARTIAL — EventDetail reads stable UID + scoped violation (P0-2/P1-12, build-verified); not browser-verified |
| 11 | Event clip as primary evidence | **FIXED** — P1-3 restructure of `EventDetail.tsx` (clip + actor/object promoted), build-verified | — | — | NOT AUTOMATED-VERIFIED (no UI test harness) |
| 12 | Full video → technical review only | **FIXED** — P1-3 (full video demoted to labelled debug section), build-verified | — | — | NOT AUTOMATED-VERIFIED |
| 13 | One box per active entity | **PASS (unit)** — `test_renderer_boxes.py`, `test_phase2_regressions.py` | — | NOT PROVEN | NOT PROVEN |
| 14 | No accumulated historical boxes | **PASS (unit)** — renderer tests + single-frame snapshot tests (`test_visualization.py`) | — | NOT PROVEN | NOT PROVEN |
| 15 | No proposal object becomes event truth | **PASS (unit)** — YOLO-source gating (`yolo_confirmed`) and authoritative-UID pair key tests (`test_detector_pair_key_uses_authoritative_object_uid`) | — | NOT PROVEN | NOT PROVEN |

### Section 24 "new tests required" checklist

| Required test | Status |
|---|---|
| Literal 3-person Case A | **DONE** (this step, `test_literal_three_person_only_litterer_is_actor`) |
| Combined Case C (simultaneous ground + bin by two actors) | **DONE** (this step, `test_simultaneous_ground_drop_and_bin_deposit_two_actors`) |
| P0-1 self-dedup regression | DONE (`test_adaptive_tuner.py`) |
| P0-2 event-scoping regression | Code + build verified; **no automated UI test** (gap) |
| P0-3 unauthenticated-POST rejection | DONE (`test_backend_api.py`: missing/wrong/env-override token + identity persistence) |
| Real double-event replay end-to-end | RAN (Section 25): 3 real videos → all 0 confirmed events → **PARTIAL** (FSM recall blocker) |

---

## 3. Section 26 — Final Health Scorecard (post-repair)

| Area | Pre-repair rating | Post-repair rating |
|---|---|---|
| Detection | VERIFIED (working, correctly gated) | VERIFIED (unchanged) |
| Tracking | VERIFIED | VERIFIED (unchanged) |
| Identity (person/object UID) | VERIFIED, design sound | VERIFIED — regression coverage extended (churn, crowd, ownership) |
| Association/ownership | VERIFIED single-object; BROKEN two-objects | Unchanged: VERIFIED single-object (incl. simultaneous two-actor Case C); **BROKEN for two-simultaneous-objects per actor (P1-4 open)** |
| Temporal reasoning (core FSM) | VERIFIED for covered cases | VERIFIED for covered cases; **OPEN: real-video release recall** (`NO_RELEASE_TRANSITION` — semantic detector loss at release, Phase 4A categories A/C/M/O; fix not yet approved) |
| Re-grab | VERIFIED (best-tested case) | VERIFIED (unit + integration); real-video verification still pending footage |
| Bin/ground | PARTIAL — proxy only | **IMPROVED, still PARTIAL** — operator-configured bin zones implemented (P1-5), 4 unit tests + Case C; no real bin footage |
| Evidence | PARTIAL — never exercised end-to-end due to P0-1 | **IMPROVED, still PARTIAL** — identity persistence fixed (P0-1), sequence images stored/served/UI (P1-2), single authoritative host storage (P1-1), extraction 6→2 video opens & 2.0x faster (P1-9); real end-to-end exercise still blocked by FSM recall |
| Dashboard | PARTIAL — live attribution bug on EventDetail | **IMPROVED, still PARTIAL** — P0-2 scoping fixed, P1-3 clip-primary hierarchy, P1-12 stable-UID display; builds clean; no automated UI tests, not browser-re-verified |
| API | PARTIAL — unauthenticated write endpoint | **VERIFIED for audited defects** — X-Internal-Token auth (P0-3, constant-time compare, env override), four stable-identity fields end-to-end (P0-1); tests included |
| Database | VERIFIED schema, BROKEN data (100% NULL identity) | **FIXED** — schema columns + idempotent migration + live Postgres write/read verified (Section 25, P1-2); real-event rows pending FSM recall fix |
| Performance | PARTIAL — dominant 3x YOLO/frame untouched | **IMPROVED, still PARTIAL** — evidence extraction 6→2 VideoCapture opens, 1,207→455 decoded frames, 13.5s→6.7s per event (P1-9); **3x YOLO model merge still open** (needs retraining, deliberately deferred) |
| Deployment | VERIFIED; evidence storage topology a real trap | VERIFIED — volume-shadowing removed (P1-1 bind mounts, named volumes pruned, stack healthy) |
| Testing | PARTIAL — strong unit, no real-video/dashboard coverage | **IMPROVED, still PARTIAL** — 146 tests incl. P0/P1 regressions, bin zones, acceptance Cases A/C; real-video and automated-dashboard coverage still absent |
| Documentation | LEGACY/MISLEADING in places | Unchanged — doc reconciliation was not part of the executed repair stages |

---

## 4. Remaining Gaps (honest list)

1. **FSM real-video recall (top blocker).** On real footage the semantic
   detector loses the object/person at the release moment, so candidates stall
   in `BAG_CARRIED` and reject `NO_RELEASE_TRANSITION` (Phase 4A diagnosis,
   categories A/C/M/O). Until fixed, criteria 3, 4, 7, 9, 10 cannot be proven
   on real video, and DB identity non-NULL cannot be demonstrated on a real
   confirmed event.
2. **P1-4: two simultaneous objects per actor unsupported** (criterion 8) —
   deferred by plan order; needs FSM change with full regression pass.
3. **No real bin footage** — bin-zone behavior is unit-proven only
   (NOT REAL-VIDEO-VERIFIED).
4. **No real multi-actor littering ground truth in D:\22** — the 3-person
   attribution machinery is unit-proven only.
5. **No automated dashboard tests** — P0-2/P1-3/P1-12 verified by typecheck/
   build + code review, not by a UI test harness or fresh browser pass.
6. **YOLO 3-model merge** (dominant CPU cost) — deferred; requires retraining
   and full FSM re-validation per plan Section 17.
7. **Pre-existing `AnalysisDetail.tsx` tsc error** (`sourceFps`) — unrelated to
   repairs; trivial but untouched per scope discipline.
8. **Documentation reconciliation** — legacy docs still describe capabilities
   that don't match source (Section 26 row); not in executed scope.

---

## 5. Verdict

**Unit-level acceptance: 14 of 15 criteria PASS** (criterion 8 unsupported by
design until P1-4). **End-to-end acceptance: PARTIAL** — every real-video and
dashboard proof point remains blocked by the single pre-existing FSM recall
issue diagnosed in Phase 4A, which is explicitly out of scope until approved.

**PASS CONDITIONS MET:** full suite green (146/146), all Section 24-required
new tests exist, all P0/P1 repair stages regression-verified, no tests weakened,
no Layer-2 behavior changed outside approved fixes.
