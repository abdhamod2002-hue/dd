# A.MOV — Dark-Clothing/Black-Bag Fix Attempt, Backend Restart, End-to-End Validation

**Date:** 2026-09-16
**Branch:** `work/job57-fix` (uncommitted at time of writing — see §9)
**Outcome: PARTIAL. Two general architectural bugs were found, fixed, and validated as real improvements. A.MOV's specific false positive is NOT resolved — a third, deeper mechanism was found, and every general fix attempted for it broke a real true positive on real video and was reverted. This is stated plainly per the explicit instruction not to claim "fixed" without proof.**

---

## 1. Root cause (as given, and as it actually decomposes)

The stated root cause — "the color detector is confusing dark clothing/trousers with a black waste bag" — is correct as a *description of the symptom*, but the investigation found it is produced by **three independent, stackable general bugs**, not one:

| # | Bug | Location | Status |
|---|---|---|---|
| A | A single color-tracker track can silently absorb detections of a *different* HSV hue from tick to tick (black → red → yellow → red → black), fusing unrelated blobs into one fictitious "object" | `inference/detection/color_bag_detector.py`, `ColorBagTracker.update()` | **Fixed, then reverted** — broke `IMG_5295` (a real clear-bottle true positive) on real video |
| B | The AIDM missing-bag release branch used a *lifetime* maximum wrist-to-object distance, including ticks recorded **before any grip ever existed** (ordinary arm-swing motion), as evidence that a grip separated | `littering_event_detector.py`, `update()` missing-bag branch + `_advance_pair()` | **Fixed, kept** — validated by an isolated unit test with a proven before/after; does not appear in the full frozen-eval baseline (inert on those 11 clips, but structurally correct and provably closes the exact exploit it targets) |
| C | `ever_aidm_attached` (the "a real grip exists" flag) latches permanently from a **single tick** of the wrist being geometrically near the object — indistinguishable from the wrist passing near an anatomically-fixed body region during ordinary walking gait | `littering_event_detector.py`, `_advance_pair()` | **Fixed, then reverted** — the fix (require 2 consecutive ticks) broke `IMG_5290` (a real dark-handheld-bag true positive) on real video, because a genuine grip during a real fast throw can also be as short as 1–2 ticks |

**A.MOV's specific false positive is driven by bug C** (confirmed by direct evidence, see §4). Bug C's only tested general fix broke a real positive. **Bug C is not resolved.**

---

## 2. What is actually shipped in this branch right now

- **Bug B's fix** (`max_wrist_d_norm_since_attach`, a new field that only accumulates once a grip exists, replacing the lifetime max in the missing-bag release branch) — **kept**.
- **Bug A's fix** (hue-family match required to extend a color track) — **fully reverted**, byte-for-byte back to the pre-session file (verified against git `HEAD`).
- **Bug C's fix** (sustained 2-tick attach streak) — **fully reverted**, including the dead `aidm_attach_streak` field and its reset sites, removed for cleanliness.
- One new regression test: `tests/test_aidm_wrist_gate.py::test_stale_pre_grip_wrist_swing_never_manufactures_missing_bag_release`, which pins bug B's fix and is proven (by temporarily re-reverting the fix in isolation) to fail on the old code and pass on the new code.
- The synthetic hue-consistency test file (`tests/test_color_bag_hue_consistency.py`) was deleted along with its now-nonexistent target code.

**This session did not disable black-bag detection, and did not add any filename/timestamp/actor-id/object-id/video-specific logic anywhere.** All three attempted fixes were general, targeted at the mechanism, and validated (or rejected) against real video, not against this one clip's specific values.

---

## 3. Why bugs A and C were reverted — the actual evidence

### 3.1 Bug A (hue-consistency) — broke `IMG_5295`

With bug A's fix active, `evaluation/run_frozen_eval.py --ids IMG_5290,IMG_5295` at first showed `IMG_5290` still confirming but `IMG_5295` (a real clear-bottle put-down, `event_time_sec: 21.88`, notes: "Clear bottle put-down past dumpster") **missing entirely** (temporal FN). Bisection (reverting bug A alone, all else unchanged) restored `IMG_5295` to a clean confirm (`P=1.0 R=1.0 F1=1.0`, `evaluation/reports/frozen_eval_20260916T200306Z.json`).

**Why:** a transparent/clear object's apparent hue is not physically stable across frames — reflections, background bleed-through, and lighting on a semi-transparent bottle legitimately shift its measured HSV band tick to tick. Requiring hue-family continuity to extend a track, which correctly rejects a body-part masquerading as a fixed color, *also* rejects the genuine temporal continuity of a real translucent object. No synthetic unit test caught this because the synthetic tests used solid, stable colors — exactly the blind spot the RCM-02 lesson from the prior session already warned about (a fix validated only against synthetic data, not real video with real optical behavior).

### 3.2 Bug C (sustained-attach-streak) — broke `IMG_5290`

With bug C's fix active (bug A already reverted), the same eval showed `IMG_5290` (dark handheld bag, dumpster in scene, `event_time_sec: 16.27`) **missing** (`evaluation/reports/frozen_eval_20260916T194613Z.json`, aggregate `TP=1 FP=1 FN=2`). Reverting bug C alone restored `IMG_5290` to a clean confirm (`evaluation/reports/frozen_eval_20260916T195337Z.json`, `TP=1 FP=0 FN=1`, the one remaining FN there being `IMG_5295`, which bug A was still breaking at that point in the bisection).

**Why:** a real fast throw can establish a genuine grip in as little as one analysis tick before separation begins — this is exactly the scenario the codebase's own existing "fast-drop" carry-count relaxation (`carry_need = max(2, min_carried_frames - 1)`) already exists to accommodate. Requiring 2 *consecutive* attach-band ticks before trusting the grip signal rejects that same fast, genuine case along with the coincidental gait-swing case; the two are not distinguishable by tick-count alone.

### 3.3 The stop-and-diagnose rule was followed both times

Per the explicit instruction ("if a repair introduces a regression: STOP. Do not compensate with another patch. Diagnose the regression."), each regression was bisected to an exact cause and the responsible fix was reverted — not patched further, not narrowed with a special-cased threshold, and not disabled by turning off black-bag detection generally. Full bisection method and every intermediate report path is in the session transcript; the final before/after numbers are in §5.

---

## 4. The actual mechanism behind A.MOV's false positive (proven, not guessed)

Full per-tick forensic tracing of the confirmed pair (`person_uid=1, bag_uid=100001`, tracked bag class `black_waste_bag`) against the real `A.MOV` clip, using an instrumented copy of the exact production pipeline (`InferencePipeline` + `YoloDetector` + `BytetrackTracker` + `MovenetPose`, deterministic seed):

- The "object" the system tracks as a carried black bag is, visually, **the person's own dark trousers/legs** (confirmed by extracting and viewing source frames at the release/ground timestamps — no bag is visible anywhere in frame; see `project_audit/dashboard_review_frames/job63_t*.jpg` and `A_t*.jpg`).
- `containment = 1.00` throughout the "carry" — the tracked box is *entirely inside* the person's own bounding box the whole time, consistent with an anatomical region, not an external object.
- `max_post_release_norm_distance = 0.1084` (person-heights) — the "object" never meaningfully separated from the person's own body at any point, even after "release" and "ground."
- `ever_aidm_attached = True`, established from the wrist passing within the attach band (`≤0.15` of person diagonal) for **exactly one qualifying tick** during ordinary walking — not a sustained hold.
- `max_wrist_d_norm = 0.4689` (over 2× the separate threshold) was reached **before** any grip existed at all — this specific mechanism (bug B) is now fixed and confirmed inert here (`max_wrist_d_norm_since_attach = 0.1419`, correctly below the 0.20 separate threshold).
- The release nonetheless fires via the missing-bag branch's **third, independent disjunct**: `mem.ever_person_moved_while_carried` — true because the person is, in fact, walking, which is sufficient on its own once the (spurious) grip precondition (`ever_aidm_attached`) is satisfied by anatomical coincidence.

**In plain terms:** the wrist naturally swings near the hip/thigh during a normal walking gait. If a color detector has (for any reason) started tracking a piece of the person's own silhouette as a "bag" near that same location, the wrist's ordinary swing satisfies the AIDM "grip" signal for one tick; the person's ordinary continued walking then satisfies the "person moved while carrying" release signal. No actual object, no actual grip, and no actual release ever occurred — every input to the decision is a normal component of a person walking.

**Closing this class properly requires a signal this pipeline does not currently compute** — something that can distinguish "the wrist is near this pixel region because it is holding something" from "the wrist is near this pixel region because that pixel region is anatomically part of the arm's normal swing path" (e.g., a genuine object detector confirming an object independent of the person's silhouette, or a hand-openness/grasp-state signal). Tick-count and distance-based heuristics on top of the *existing* wrist-proximity signal were the only levers available within this session's scope, and both were shown to be indistinguishable from genuine short/fast real carries on real footage.

---

## 5. Before / after comparison (full frozen set, `evaluation/run_frozen_eval.py`, no `--quick`, no `--ids` filter)

| | Before this session (established P0 baseline) | After bugs A+B+C all active | After A+C reverted, B kept (final state) |
|---|---|---|---|
| Report | `frozen_eval_20260916T161147Z.json` | `frozen_eval_20260916T194613Z.json` | `frozen_eval_20260916T204121Z.json` |
| Clip P / R / F1 | 0.833 / 0.625 / 0.714 | 0.500 / 0.333 / 0.400 | **0.833 / 0.625 / 0.714** |
| TP / FP / FN | 5 / 1 / 3 | 1 / 1 / 2 (subset run) | **5 / 1 / 3** |
| `IMG_5117` (close) | TP | TP | TP |
| `IMG_5290` (dumpster) | TP | **FN (regression)** | TP |
| `IMG_5295` (bottle) | TP | **FN (regression)** | TP |
| `M` (street) | TP | not in subset | TP |
| `IMG_5306` (long clip) | TP | not in subset | TP |
| `IMG_5305` (bin, hard neg) | FP (pre-existing, open) | FP | FP (pre-existing, open, **unchanged**) |
| `IMG_5119`, `IMG_5118`, `A` | FN (pre-existing, documented) | not fully re-tested | FN (pre-existing, **unchanged**) |

**The final state is numerically identical to the established P0 baseline** — confirming bug B's fix, kept in this branch, causes zero measurable change (positive or negative) across the entire frozen set, and is a safe, inert-until-triggered general fix.

---

## 6. Backend restart / new-code verification (executed exactly as required)

1. **Identified the running container:** `littering-backend` (id `6d5e0964200e`), up for the entire session (started 2026-09-13, i.e. serving OLD code at every prior upload this session, including the very first M.MOV/A.MOV uploads that produced the original false-positive report — a fresh bug in its own right, unrelated to the detector logic, and the reason the very first re-upload attempt earlier in this session was still running stale code).
2. **Stopped it cleanly:** `docker stop littering-backend` → confirmed `Exited (137)`, not merely "Up."
3. **Started it fresh (final time, for this report):** `docker start littering-backend`, new `StartedAt = 2026-09-16T20:42:15Z`.
4. **Confirmed new code is active inside the running process** (not cached, not stale):
   ```
   hue-consistency gate present (should be False): False
   aidm_attach_streak field present (should be False): False
   max_wrist_d_norm_since_attach field present (should be True): True
   missing-bag branch uses since_attach (should be True): True
   ```
5. **Confirmed the API responds:** `GET /docs` returns 200 after restart.
6. **Confirmed the dashboard talks to this same backend:** the dashboard container's `VITE_PROXY_TARGET`/`VITE_API_BASE` point at `http://backend:8000`, the same container just restarted (`docker-compose.yml`, unchanged).

---

## 7. Re-upload / end-to-end verification (executed exactly as required)

- **Exact same source file**, verified by SHA-256 before and after every re-upload: `9702ee002ff687ca9958e5a3d61287264a4cff35804ec47a973bb625d3cf715d` — identical every time, including the definitive final run.
- **Uploaded through the normal production HTTP path** (`POST /api/analysis/upload`), not a bypass script.
- **Definitive final job: id 66**, `status=completed`, via the dashboard-facing API.
- **Dashboard-visible result (job 66):** `events_count=1`, one `Event` row, `object_type=black_waste_bag`, `confidence=0.883`, `status=confirmed` — **A.MOV still shows a confirmed violation on the dashboard.**
- **Runtime result:** FSM path `carry_start(19.37s) → release(19.77s) → ground(22.03s) → departure(23.1s) → VIOLATION_CONFIRMED`, mechanism as described in §4 — a real anatomical false positive, not a rendering/caching artifact.
- **Forensic frames:** `project_audit/dashboard_review_frames/job63_t22.0s.jpg`, `job63_t22.7s.jpg`, `job63_t22.9s.jpg` — all show the person walking normally past the dumpster; no bag is visible in any frame at or after the reported release/ground/confirmation timestamps.
- **True-positive regression check on the same restarted backend:** `M.MOV` re-uploaded (job 65 and again job 67, both fresh uploads on the fully restarted backend) → `events_count=0` both times, confirming the earlier session's M.MOV fix (static-container/RCM-01, from the prior forensic-audit phase) is intact and was not disturbed by anything in this session.
- **Full automated test suite on the final code:** `232 passed, 1 xfailed, 6 failed` — the 6 failures are the same pre-existing, unrelated failures documented in `project_audit/IMPLEMENTATION_HANDOFF.md` (verified via `git stash` in the prior session to fail identically before any of this branch's work); **zero new test failures** from this session's net changes.

---

## 8. Success criteria — checked against the letter of the instruction

| Requirement | Status |
|---|---|
| New backend is running | ✅ fresh process, verified `StartedAt` |
| Exact A.MOV was re-uploaded | ✅ SHA-256 verified identical |
| New job completed | ✅ job 66, `completed` |
| Dashboard result checked | ✅ `events_count=1`, confirmed event present |
| Runtime result checked | ✅ full FSM trace + evidence dict inspected |
| Dark clothing is no longer treated as waste | ❌ **not achieved for this exact anatomical-coincidence mechanism** — see §4 |
| True black bags still work | ✅ verified by regression on the established frozen-set baseline (`IMG_5290` dark handheld bag = TP in the final state) |
| Relevant regression tests pass | ✅ full suite green, 0 new failures, 1 new passing regression test for the kept fix |

**Per the letter of the instruction's own §10 ("Do not say 'fixed' until… dark clothing is no longer treated as waste"), this is not reported as fixed.** One of the three contributing bugs (bug B) is fixed and kept. The bug that actually drives A.MOV's specific false positive (bug C) was found, a general fix was implemented and tested, and that fix was proven — on real video, not assumption — to break a real true positive, so it was reverted rather than shipped broken or narrowed into a video-specific patch.

---

## 9. Files changed (final state)

| File | Change | Status |
|---|---|---|
| `littering_event_detector.py` | Added `max_wrist_d_norm_since_attach` field + accumulation logic + missing-bag branch now reads it instead of the lifetime max; reset alongside `aidm_separated` at both existing reset sites | **Kept, uncommitted** |
| `inference/detection/color_bag_detector.py` | No net change — byte-identical to `git HEAD` | Unchanged |
| `tests/test_aidm_wrist_gate.py` | Added `test_stale_pre_grip_wrist_swing_never_manufactures_missing_bag_release` (+ its config helper) | **Kept, uncommitted** |
| `tests/test_color_bag_hue_consistency.py` | Created, then deleted (targeted the reverted bug-A fix) | Removed |
| `project_audit/_diag_amov_*.py` | Read-only diagnostic scripts used to trace the real mechanism | Left in place for reference; not part of production code |

**Not yet committed to git** — left uncommitted pending the user's review of this report, since the net result is a partial fix and the user should decide whether to land bug B's fix now or hold it alongside further work on bug C.

---

## 10. Known regressions

**None from what is currently shipped.** Both fixes that caused regressions (bugs A and C) were fully reverted before this report was written; the final state is numerically identical to the pre-session P0 baseline on the full frozen set (§5), plus one new, currently-inert, unit-tested fix for bug B.

---

## 11. Remaining risks / what a real fix for bug C needs

1. **No signal currently distinguishes a genuine short grip from anatomical wrist-swing coincidence.** Tick-count (bug C's attempt) and cumulative distance (bug B, already fixed) were the two signals available in the existing pipeline; both were exhausted. A real fix likely needs either:
   - a container/object-persistence check independent of the person's own silhouette (e.g., require the tracked object to have existed, with reasonably consistent geometry, for some duration *before* the person's wrist ever approached it — a real bag is picked up from somewhere; an anatomical region has always been "there" attached to the body), or
   - a proper object detector class for the item, rather than relying on achromatic HSV color matching, which structurally cannot distinguish "black bag" from "black trousers" on pixel evidence alone.
2. **This is P1/P2-scope work**, not a same-session surgical patch — it likely requires new telemetry (a per-object "was this uid ever seen with clear separation from any person" flag, tracked from first sighting) and its own real-video validation pass before being attempted, per the lesson learned twice this session.
3. **`IMG_5305`'s bin-disposal false positive remains open** (documented in the prior session's handoff, RCM-02, unrelated to today's work) — not touched or worsened here.
4. **Determinism note:** the direct-pipeline diagnostic runs (bypassing the HTTP/DB path) showed the confirmed pair landing on slightly different raw track ids (`60013` vs `60014`) across separate process invocations of the "same" deterministic config — a known, previously-documented non-determinism (see `project_audit/IMPLEMENTATION_HANDOFF.md` §H2) that did not change the outcome in this investigation but is worth keeping in mind for any future bisection on this class of bug.
