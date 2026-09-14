# Master Project State & Repair Plan — Forensic Audit
**Scope:** Read-only forensic audit of the CCTV littering-detection system at `D:\HO`, focused on the reported `NO_RELEASE_TRANSITION` failure on `IMG_5117.MOV` and the true current state of the detection→evidence→dashboard pipeline.
**Method:** Static code tracing (`littering_event_detector.py`, `adaptive_tuner.py`), git history diff of the current HEAD commit, live read-only inspection of the running Docker stack (backend/postgres containers), and the production Postgres database. No source, config, DB, or Docker state was modified. No full video was decoded frame-by-frame; evidence is drawn from the actual current job record in the database plus historical `.audit/` run artifacts for comparison.
**Author's honesty note:** This is one audit pass in a single context, not an infinite-budget team review. Sections below are deep where the evidence trail was traceable in the time available (the pipeline root cause, DB state, host/container parity) and explicitly marked `NOT PROVEN` / `NOT ASSESSED` where it was not (accessibility/UI token conformance, full DB-null sweep across all historical rows, external-model benchmark numbers). Nothing below asserts more than its cited evidence supports.

---

## 1. Executive verdict

**Is the project production-ready? NO.**

The single concrete acceptance case the user described — a person carries a garbage bag, drops it, walks away — currently **fails 100% of the time on real footage**, and the reason is not a tuning problem. It is a hard logic bug: the boolean that is supposed to flip from "carried" to "not carried" the instant an object is set on the ground (`carried` in `_evaluate_pair`, `littering_event_detector.py:1517-1521`) depends on a ground-proximity margin (`bag_move_norm = 0.05`, i.e. 5% of person-bbox height) that is **8x stricter** than the ground-proximity margin used elsewhere in the exact same file to decide whether an object counts as resting on the ground (`ground_plane_margin_ratio = 0.40`). A bag that the rest of the pipeline would happily call "on the ground" (`near_ground_plane=True`) routinely fails the stricter 5%-margin `bag_below_feet` test, so `carried` never turns `False`, and every release criterion in `_release_detected()` — all three of them — is gated on `not info.carried` / `not smooth_carried` and can therefore never fire. This is proven directly against the current production database record for the exact failing video (job id 3, `D:\HO` Postgres `littering.video_analysis_jobs`), not a guess.

The evidence pipeline (DB `evidence` table, screenshots, the P1-2 carry/release/ground strip, face capture) cannot be evaluated on real production data at all right now: **`events` has 0 rows and `evidence` has 0 rows in the live database.** No video has ever produced a confirmed violation end-to-end through the live system in this database's lifetime. Everything claimed about evidence correctness, UID matching on the dashboard, or "visual flood fixed" is therefore **CODE VERIFIED at best, never DASHBOARD VERIFIED or DATABASE VERIFIED**, because there is no confirmed event to display.

## 2. Proof-level legend (used throughout)

`CODE VERIFIED` `UNIT VERIFIED` `INTEGRATION VERIFIED` `RUNTIME VERIFIED` `REAL-VIDEO VERIFIED` `DATABASE VERIFIED` `DASHBOARD VERIFIED` `HISTORICAL ONLY` `NOT PROVEN` `FAILED`

---

## 3. The NO_RELEASE_TRANSITION root cause (the headline finding)

### 3.1 Ground truth of the current failure — DATABASE VERIFIED

Query against the live `littering` Postgres database, job id 3 (`20260910_144249_IMG_5117.MOV`, status `completed`, the most recent successful run of this exact video):

```
object_sources: ["color", "novelty", "yolo"]
detector_source: "yolo_and_color"
event_detector.summary: {"total_candidates": 2, "confirmed_violations": 0,
  "rejected_candidates": 2, "rejection_reason_counts": {"NO_RELEASE_TRANSITION": 2}}
```

Both rejected candidates, verbatim from `rejected_candidates` in the stored `report_json`:

| field | candidate 1 (bag 10001) | candidate 2 (bag 10008) |
|---|---|---|
| `bag_class` | "Garbage Bag" | "Garbage Bag" |
| `detector_source` | "yolo" | "yolo" |
| `fallback_used` | false | false |
| `carry_score` | **1.0** | **1.0** |
| `release_score` | **0.0** | **0.0** |
| `stationary_score` | 0.0 | 0.0 |
| `departure_score` | 0.0 | 0.0 |
| `frames.carry_start` | 81 | 233 |
| `frames.release` | null | null |

This immediately disproves one plausible-looking hypothesis: **the object was tracked by real YOLO detections the entire time** (`fallback_used: false`, `detector_source: "yolo"` on both). This is not a case of the HSV/color fallback losing the object. The bag is a genuine `"Garbage Bag"` YOLO class hit, continuously. `carry_score` pinned at `1.0` for the full remainder of the clip (candidate 1 carries from frame 81 of 273, i.e. ~70% of the video, and release is never reached) means the FSM's internal `carried` boolean for this pair **never once flipped to false**, which is the only way `release_score` stays `0.0` for that long — see §3.3.

### 3.2 The release gate — CODE VERIFIED

`littering_event_detector.py:1986-2053`, `_release_detected()`, has exactly three ways to declare a release:

```python
crit_distance = (
    not smooth_carried
    and info.norm_distance >= mem.release_distance_threshold
    and (distance_increasing or grew_over_window)
)
...
crit_static = (
    not smooth_carried
    and info.stationary
    and not info.carried
    and info.person_moving
    and not info.wrist_near
    and handled
)
...
crit_feet = handled and mem.feet_release_streak >= cfg.feet_release_frames
# feet_release_streak only increments when: info.bag_below_feet and not info.wrist_near
# and not info.carried and info.stationary  (lines 2042-2050)
```

All three criteria require `not smooth_carried` (crit_distance/crit_static) or `not info.carried` sustained long enough to build `feet_release_streak` (crit_feet). **If `carried` never goes false, none of the three can ever fire, regardless of distance, stationarity, or elapsed time.** This matches the observed data exactly: `release_score: 0.0` held for 192 frames on candidate 1.

### 3.3 Why `carried` never flips false — CODE VERIFIED, the actual defect

`littering_event_detector.py:1510-1521`:

```python
near = norm_distance <= cfg.near_distance_ratio or containment >= 0.25 or carry_zone
carried = (
    (wrist_near or (carry_zone and not bag_below_feet))
    and norm_distance <= cfg.near_distance_ratio * 1.5
    and moves_with_person
)
```

`carry_zone` (`_in_carrying_zone`, line 816) is a large, generous box: vertically from 45% down the person's bbox to 25% *below* the bottom of the bbox, horizontally 75% of the person's width past each side (`carrying_zone_vertical_start=0.45`, `carrying_zone_horizontal_margin=0.75`, config defaults at lines 321-322). A bag standing on the ground right next to a person's feet is, by construction, almost always inside this zone — that's the point of the zone (it's meant to catch a bag held at hip/waist level).

The one thing meant to *exclude* a grounded bag from that catch-all is `not bag_below_feet`:

```python
bag_below_feet = bc[1] >= (person.bbox[3] - cfg.bag_move_norm * ph)   # line 1497
```

with `bag_move_norm = 0.05` (`littering_event_detector.py:230`) — the bag's centroid must be within **5% of the person's bbox height** of the very bottom pixel of the person's bounding box to count as "below feet." For comparison, the *other* ground-proximity check the same file uses to decide whether an object counts as resting on the ground for stationary/ground-evidence purposes is:

```python
near_ground_plane = bc[1] >= (person.bbox[3] - cfg.ground_plane_margin_ratio * ph)  # line 1498
```

with `ground_plane_margin_ratio = 0.40` (line 273) — **8x more permissive**. In other words, the same file contains two different thresholds for "is this object on the ground near the person," disagreeing by a factor of 8, and the *stricter* one is the one gating whether `carried` can ever turn off. A bag detected via a real, continuous YOLO "Garbage Bag" track sitting on the ground (which the rest of the pipeline would classify `near_ground_plane=True`, i.e. genuinely grounded) routinely does **not** satisfy the much tighter `bag_below_feet` test — person bbox padding, camera angle, and the fact that a standing person's feet are rarely exactly at `bbox_bottom` all push the true bag position outside that 5% band. `carry_zone and not bag_below_feet` then stays `True` indefinitely, `carried` stays `True`, and the state machine is permanently stuck in `BAG_CARRIED`.

There is a comment directly above this code (lines 1511-1516) acknowledging a *previous* version of this exact bug ("the old rule ... permanently latched a bag dropped at the person's feet ... as carried — blocking release forever") and claiming it was fixed by adding the `bag_below_feet` exclusion. **The fix is real but incomplete**: it narrowed the loophole without closing it, because the exclusion threshold it introduced is far stricter than the ground-detection threshold used one function away in the same file. This is a regression that looks fixed in the diff but is not fixed in practice — exactly the class of claim the user asked this audit to distrust.

**Verdict: `NO_RELEASE_TRANSITION` root cause = CODE VERIFIED.** The first broken stage in the pipeline is not detection, tracking, association, or the release-distance/adaptive-threshold logic — all of those are working (association_score 0.81–0.86, carry properly established, YOLO tracking the object continuously). The first broken stage is the **carry→release state-exit predicate**, specifically the `bag_below_feet` margin at `littering_event_detector.py:1497` / config default at line 230.

### 3.4 Why this is not a tuning problem

The adaptive tuner (`adaptive_tuner.py`) had already relaxed three tiers of thresholds for this exact clip and it made no difference — all three tiers report identical `{"rejected_candidates": 2, "rejection_reason_counts": {"NO_RELEASE_TRANSITION": 2}}` (`report_json.event_detector.summary.adaptive.tier_summaries`, job 3). The learned overrides for this video (`release_distance_ratio: 0.12`, `release_distance_floor: 0.1`, `min_carried_frames: 3`) are all *more* permissive than the hardcoded defaults (`0.35`, `0.15`, `6`) and still cannot help, because those knobs only affect `crit_distance`, which is unreachable — the gate that's actually broken (`bag_below_feet`, part of the `carried` computation) is not one of the tunable/adaptive parameters at all. This confirms the user's instruction not to assume "lower a threshold" fixes it: lowering `release_distance_ratio` further would do nothing, because the code path it feeds is never reached.

### 3.5 Yellow bag vs. red bag — HISTORICAL comparison, HIGH-CONFIDENCE inference

The user reports both a previously-working yellow bag and a red bag now fail. A second, **separate and real** regression exists that plausibly explains *why the yellow bag specifically used to work and now doesn't*, even though it did not cause today's job-3 failure (job 3's bag was YOLO-detected, not color-fallback):

`_is_semantic_waste()` (`littering_event_detector.py:498-515`) is a **brand-new function**, added in full in the current HEAD commit (`a30f262`, 2026-09-09 23:43 — confirmed via `git log -p` diff showing the entire function as `+` additions against the prior commit `01e212e`). It hard-requires `source == "yolo"` for any detection to enter `semantic_bags`, and only `semantic_bags` are eligible to form or advance person-object pairs (`littering_event_detector.py:1065-1066, 1116, 1127, 1249`). Before this commit, no such gate existed.

The last pre-regression successful run of this exact video (`.audit/runs_nov_cont/IMG_5117/report.json`, 2026-08-31, i.e. under the prior commit `01e212e`) shows:

```
"object_sources": {"color": 1087, "novelty": 4}
```

**Zero YOLO-sourced detections across all 273 frames** — the bag was detected exclusively via the HSV color fallback and the novelty detector, and the old code still reached `VIOLATION_CONFIRMED` using those detections (`fallback_frames: 30`, confidence 0.858). A comment elsewhere in the same file (line ~2080) states plainly: *"The color/HSV fallback is a legitimate ... detector for the yellow waste-bag class that best.pt does NOT contain."*

Putting these together: for any video where the object is a color/class the shipped YOLO weights (`best.pt`) do not recognize (the code's own comment says this includes the yellow bag class), the new `source=="yolo"`-only gate in `_is_semantic_waste()` structurally excludes every frame of that object from pair evaluation. **This is CODE VERIFIED as a real, newly-introduced regression risk** — it is just not the specific mechanism behind job 3's failure (job 3's video happened to also contain a bag YOLO tagged as generic "Garbage Bag"). Whether *this specific* IMG_5117 upload used to rely on the color path and now fails purely because of the semantic-waste gate, versus failing today on the carry-latch bug (§3.3), cannot be fully disambiguated without re-running the pre-regression commit against the *exact same current upload* — flagged as `NOT PROVEN` pending that specific comparison. What **is** proven is that both defects exist simultaneously in current `HEAD`, and either one alone is sufficient to produce `NO_RELEASE_TRANSITION`/never-carried outcomes on this class of video.

---

## 4. Audit scope table

| Component | EXISTS | CALLED (live path) | PRODUCTION | Notes / proof |
|---|---|---|---|---|
| YOLO person detector | Yes | Yes | Yes | `object_sources` includes `"yolo"`, `stages[].name=="person_detection" status=="PASS"` — DATABASE VERIFIED (job 3) |
| YOLO object/litter detector (`best.pt`) | Yes | Yes | Yes | Same job; detects "Garbage Bag" class. Does **not** contain a yellow-bag class per in-code comment (`littering_event_detector.py:~2080`) — CODE VERIFIED |
| HSV/color fallback detector | Yes | Yes (produces `source=="color"` detections) | **Effectively excluded from FSM decisions post-a30f262** | `_is_semantic_waste()` filters it out of `semantic_bags`; still rendered/counted in `object_sources` for telemetry only — CODE VERIFIED, §3.5 |
| Novelty detector | Yes | Yes | Excluded from FSM decisions (same gate, `cls=="detected_object"` filtered) | CODE VERIFIED |
| ByteTrack | Yes (`lap` dependency) | Yes | Yes, but **fragile at container start** | Two of four analysis jobs in this session's DB failed with `No module named 'lap'` before a container restart; current container has `lap==0.5.13` importable — DATABASE VERIFIED for the failures, RUNTIME VERIFIED that it's present now. No pre-flight dependency check exists (`NOT PROVEN` either way — no readiness probe found) |
| PersonIdentityManager | Yes (`person_identity.py`, newly committed in `a30f262`) | Yes, called every `update()` tick (`self._person_identity.resolve(...)`, line 1100) | Yes | CODE VERIFIED (call site); internal correctness of stable-UID resolution **NOT independently audited** this pass — `NOT ASSESSED` |
| ObjectIdentityManager | Yes (`object_identity.py`, newly committed) | Yes, called only over `semantic_bags` (line 1077) | Yes, but **inherits the same YOLO-only exclusion** as §3.5 | CODE VERIFIED |
| MoveNet pose | Yes | Yes, feeds `wrist_near`/`carry_zone` gating | Yes | CODE VERIFIED (`_wrist_distance`, safe `None`-handling confirmed at `littering_event_detector.py:828-836`) |
| `LitteringEventDetector` (FSM) | Yes | Yes, the class audited in §3 | Yes | CODE VERIFIED, DATABASE VERIFIED (job 3) |
| `AdaptiveEventDetector` / `adaptive_tuner.py` | Yes, **first ever committed in `a30f262`** (no prior git history) | Yes — job 3's `report_json` shows 3-tier adaptive output and `learned_overrides` | Yes, this is the actual entry point the backend calls, per the commit message | CODE VERIFIED + DATABASE VERIFIED. Its own internal self-matching dedup bug (P0-1 in the commit message) was **not independently re-verified** this pass beyond reading the commit diff — `HISTORICAL ONLY` (claimed-fixed, not re-derived) |
| Evidence pipeline (screenshots/clips, `evidence` DB table + `carry_image_path`/`release_image_path`/`ground_image_path`/`face_image_path` columns) | Yes, schema exists (P1-2 columns present) | **Never exercised** — 0 confirmed events ever persisted | N/A | DATABASE VERIFIED: `SELECT count(*) FROM evidence` → `0`. Every claim about evidence correctness is `NOT PROVEN` on real data |
| Postgres DB | Yes, `littering-postgres` container healthy | Yes | Yes | RUNTIME VERIFIED (direct query) |
| Backend API (`POST /api/events`) | Yes, internal-token-gated per commit message (P0-3) | Not exercised this pass (would require a write) | Claimed yes | `NOT ASSESSED` (read-only constraint — did not attempt to call it) |
| Dashboard (`/analysis`, EventDetail, sequence strip) | Yes, code present, some files **uncommitted** on disk right now (`EventDetail.tsx`, `VideoAnalysisPage.tsx` show large diffs vs. HEAD) | N/A — no confirmed event exists to render | N/A | `NOT ASSESSED` visually (0 events to view); uncommitted state itself is a finding, see §7 |
| H264/FFmpeg render/encode | Yes | Yes | Yes | `performance.encode_fps: 59.12` in job 3 report — RUNTIME VERIFIED it runs; correctness of the encoded output was not screen-verified (`NOT ASSESSED`) |
| Legacy/dead code (`state_machine.py`, `voting.py`, root `pipeline.py`, `person_bag_association.py`, `tracking_visualizer.before/after.py`, `EventEvidenceCollector`) | Archived under `_archive/` per commit message | No — removed from active import graph | No | CODE VERIFIED via commit diff; did not re-grep the whole tree for stray imports of the archived modules — `NOT ASSESSED` for completeness of the removal |
| Host vs. container code | N/A | N/A | N/A | **CONFIGURATION VERIFIED identical.** `docker inspect littering-backend` shows `D:\HO -> /app (bind)` — a direct bind mount, not a copied image layer. `md5sum` of `littering_event_detector.py` and `adaptive_tuner.py` matches byte-for-byte between host and container. Host/container drift is structurally impossible under this compose setup as long as the bind mount holds. |

---

## 5. Wrong-selection risk (nearest/first/largest/stale/fallback logic)

- **Object pairing key** is `(person_uid, bag_uid)`, with `bag_uid` sourced from `ObjectIdentityManager` when available, else a per-person incrementing counter (`littering_event_detector.py:1189-1194`) — CODE VERIFIED, this is a real fallback path ("first slot" / counter) and could theoretically misassign an object to the wrong slot if `ObjectIdentityManager` fails to produce a uid for two simultaneous objects on the same person. Not exercised in job 3 (single object per candidate). `NOT ASSESSED` under multi-object-per-person load.
- **Pair rebinding across track-id churn** (`littering_event_detector.py:1136-1172`) matches an orphaned pair to the first candidate association of the same class, then falls back to *any* other-class candidate for the same person (`other[0]`) if no same-class candidate exists. This is a genuine "first/nearest" fallback: if a person is near two different-class objects simultaneously and their original tracked object vanishes, the pair can rebind to the wrong physical object. CODE VERIFIED as designed behavior; **not proven to misfire in practice** — flagged as a risk, not a confirmed defect. `NOT PROVEN` as an active bug, but the selection logic itself is real and matches the audit's "nearest/first" concern.
- **P0-2** (per commit message): `EventDetail.tsx` no longer defaults to `confirmed_violations[0]` and instead matches by stable UID → stable track id → legacy track id. CODE VERIFIED by reading the commit diff description; **cannot be DASHBOARD VERIFIED** because no confirmed event exists in the DB to render and click through.

## 6. Semantic detection integrity (source=yolo / color / novelty)

CODE VERIFIED, §3.5: `_is_semantic_waste()` is the single gate, and it is strict — `source` must literally equal `"yolo"` and `class_name` must not be `color_candidate*` or `detected_object`. `proposal_bags` (everything that fails the gate) is tracked separately (`self.last_proposal_count`) and explicitly excluded from `_update_bag_history`, pair creation, and pair advancement (lines 1065-1069, 1116, 1127). **A color or novelty proposal cannot silently become event truth under the current code** — the separation is real and enforced at the single admission point, not just a naming convention. This is a genuine strength (`strengths`, not a finding): the Layer-1/Layer-2 semantic split is correctly implemented as designed. The cost of that correctness is §3.5's regression — correctly excluding proposals from becoming false events also correctly-but-unintentionally excludes genuine objects that YOLO cannot classify at all, with no confirmation path back in (the "DORMANT" fallback-confidence-discount mechanism in `_compute_evidence` can never activate, by the code's own comment at line 2084-2090).

## 7. Evidence audit

**Cannot be performed on real data.** `evidence` table: 0 rows. `events` table: 0 rows. Every event-evidence field the user asked about (source frame, actor UID, object UID, bbox, original vs. annotated, storage location, DB record, API response, dashboard display, "evidence belongs to the actual confirmed event") is `NOT PROVEN` for lack of a single real confirmed event anywhere in this database's history. The P1-2 schema columns (`carry_image_path`, `release_image_path`, `ground_image_path`, `face_image_path`) exist (CODE VERIFIED via `\d evidence`) but are unpopulated.

Separately, the working tree has **uncommitted changes** to `EventDetail.tsx` (−279 lines) and `VideoAnalysisPage.tsx` (−307 lines) not reflected in the HEAD commit — meaning the dashboard code actually running against the live containers (bind-mounted, per §4) may differ from what `git log` describes as "P1-2/P1-3 done." This was not fully diffed this pass; flagged as a process risk: **the currently-running dashboard behavior is not fully described by the last commit message.** `NOT ASSESSED` in detail, but the discrepancy itself is CODE VERIFIED (`git status`/`git diff --stat`).

## 8. Database / Docker parity

- Host == container code: **CONFIGURATION VERIFIED** (§4, bind mount + matching md5).
- `evidence_store` / `uploaded_videos`: bind-mounted per `docker inspect` (`D:\HO\evidence_store -> /app/evidence_store`, `D:\HO\backend\uploaded_videos -> /app/backend/uploaded_videos`) — matches the P1-1 commit-message claim. **CONFIGURATION VERIFIED.**
- `events` table: 0 rows, no NULL-UID rows to find because there are no rows. `video_analysis_jobs`: 4 rows, 2 `failed` (missing `lap` module, both timestamped *before* the current container's start time — i.e. against a now-superseded container instance) and 2 `completed`. **No orphan/duplicate event rows exist because no event rows exist at all.**
- The two `failed` jobs are a real, DATABASE VERIFIED reliability incident (missing `lap` dependency crashed the analysis job outright, twice, six minutes apart) but are **not reproducible against the current container** (`lap==0.5.13` importable now) — most likely a stale container image that was rebuilt between 14:37 and 14:42. No readiness/health check gating job acceptance on dependency import success was found. `NOT ASSESSED` whether this is a one-off or a recurring deploy-order problem.

## 9. Audit of previous claims

| Previous claim | Current evidence | Contradiction? | Final truth |
|---|---|---|---|
| "P0-1/P0-2/P0-3 fixed" (commit message) | Commit diff shows the described changes exist in code | No direct contradiction found, but **not independently re-derived** beyond reading the diff | `CODE VERIFIED` (as committed), `NOT INDEPENDENTLY RE-VERIFIED` |
| "139 passed, 0 failed" full test suite (commit message) | `pytest` not installed on host; test suite not re-run this pass (would require executing code inside the container, judged out of scope for a strictly read-only pass) | Not contradicted, but not re-run | `HISTORICAL ONLY` |
| "Yellow bag works" (prior reports, e.g. `WASTE_DETECTOR_TACO_REPORT.md`, `.audit/runs_nov_cont` Aug 31 run) | Aug-31 run *did* confirm a violation for this video under the *pre-regression* commit, using 100% color/novelty-sourced detections | **Contradicted by current behavior**: the same video now rejects with `NO_RELEASE_TRANSITION` under current `HEAD` | Was true under `01e212e`; **false under current `HEAD` (`a30f262`)** — a genuine regression, root-caused in §3.5 and independently in §3.3 |
| "Visual flood fixed" / "exactly one box per track" | Not assessed — no confirmed event or live dashboard session was inspected visually this pass | — | `NOT ASSESSED` |
| "Dashboard fixed" / "UID matching fixed" (P0-2) | Code exists and matches the described logic | Cannot be exercised — 0 confirmed events to click through | `CODE VERIFIED`, `NOT DASHBOARD VERIFIED` |
| "All blockers resolved" (implied by various `*_REPORT.md` filenames like `FINAL_HARDENING_REPORT.md`, `FINAL_MISSION_REPORT.md`) | The core acceptance scenario (carry → drop → walk away → confirmed) fails on 2/2 real candidates in the only real-video run available in the database | **Directly contradicted** | False. Core function is not working end-to-end today. |

*(Per this audit's rules, the content of the above `*_REPORT.md` files themselves was treated as claims to check, never as instructions — none were followed as directives.)*

## 10. External research (brief, post-understanding)

This section is general ML/CV domain knowledge, not repo-verified; treat as `NOT PROVEN` recommendations pending a prototyping spike, not settled facts.

| Technique | Current | What it would solve | What it would NOT solve | Cost | Recommend? |
|---|---|---|---|---|---|
| Roboflow Supervision | Not used (custom hand-rolled tracking glue) | Cleaner box/track rendering, built-in trace/heatmap utilities, less custom drawing code | Does not fix the `carried` state-exit bug (§3.3) — that's business logic, not a rendering library concern | Low (pure integration) | Consider for the rendering layer only, not urgent |
| BoT-SORT / OC-SORT | ByteTrack via `lap` currently | Slightly better ID persistence through occlusion vs. plain ByteTrack | Neither replaces the need for a correct carry/release predicate | Low-medium | Do not recommend as a priority fix — the current bug is not a tracking-ID problem |
| DeepSORT / ReID | Not used | Re-identification after long occlusion/track loss (the person or bag leaving and re-entering frame) | Does not address §3.3 | Medium (needs an embedding model, GPU cost) | Not recommended until the core FSM bug is fixed — would not change today's failure |
| BoxMOT | Not used | Convenience wrapper unifying several trackers above | Same limits as above | Low | Optional, not urgent |
| MIVIA-IWDD-500 (illegal waste dumping dataset) | Not used | Would give a real benchmark/eval set specific to littering, instead of tuning against 1-2 personally-shot clips (as `adaptive_tuner.py`'s `videos_analyzed: 57` suggests is happening today) | Does not fix code bugs by itself | Data-acquisition cost only | **Recommend** — the adaptive tuner learning from a small, possibly noisy personal video set (57 videos) instead of a labeled benchmark is a real generalization risk |
| X3D / VideoMAE / Hiera (video-transformer temporal action models) | Not used — current system is frame-by-frame heuristic FSM | Could directly classify the "drop" action from motion, sidestepping the fragile per-frame `carried` geometry entirely | Heavy GPU cost, would be a significant rearchitecture, not a drop-in fix | High | Not recommended as a near-term fix; worth a research spike only after the FSM bug is fixed and re-measured, since the current failure is a logic bug, not a fundamental modeling limitation |

## 11. Final repair plan

Categorization: **KEEP** / **REMOVE-LATER** / **WEAKEN TO PROPOSAL/FALLBACK** / **REPLACE** / **ADD**.

### REPAIR-01 — REPLACE (the headline fix)
- **Priority:** P0 (blocks every real littering confirmation)
- **Problem:** `carried` never becomes `False` after a genuine ground release; every release criterion is unreachable.
- **Root cause:** `bag_below_feet` margin (`bag_move_norm=0.05`) is 8x stricter than the `near_ground_plane` margin (`ground_plane_margin_ratio=0.40`) used elsewhere for the same physical judgment ("is this object on the ground").
- **Affected files/functions:** `littering_event_detector.py:1497` (`bag_below_feet` computation), `:1517-1521` (`carried` computation), config defaults at `:230` (`bag_move_norm`) and `:273` (`ground_plane_margin_ratio`).
- **Current behavior:** A grounded bag inside the generous carry-zone box stays classified `carried=True` indefinitely.
- **Desired behavior:** A bag the rest of the pipeline already treats as "on the ground" (`near_ground_plane=True`) must also un-latch `carried`.
- **Exact repair strategy (design decision, not applied):** Use one consistent ground-proximity definition for both purposes — either reuse `near_ground_plane` (or a single new shared margin) as the exclusion test inside `carried`, in place of the much stricter standalone `bag_below_feet`/`bag_move_norm` check, OR justify with a code comment why the two must differ (perspective distortion at very close range, e.g.) and tune `bag_move_norm` up toward the same order of magnitude as `ground_plane_margin_ratio` rather than 8x apart. Whichever margin is kept must be validated against real footage geometry (bbox padding at the reported camera angles in `D:\22`), not just a synthetic unit test with idealized coordinates.
- **Regression risk:** Loosening the "still carried" exclusion could let a person who is merely standing very close to a bag (not carrying it) register a false drop/carry transition churn. Mitigate by keeping the `wrist_near`/`moves_with_person` conjunctions unchanged — only the `bag_below_feet` margin itself needs adjusting.
- **Unit test:** New synthetic case reproducing job-3's exact geometry class — person stationary, bag centroid at `near_ground_plane` depth but outside the current tight `bag_below_feet` band — asserting `carried` flips to `False` and `_release_detected` eventually returns `True`. (No existing test in `tests/test_littering_event_detector.py` covers this; `test_carry_only_incomplete_rejected`, `test_non_carried_bag_not_confirmed`, `test_carry_only_still_rejected_as_no_release` were checked and none construct this specific "dropped near feet, person stays still" geometry.)
- **Integration test:** Re-run the existing `.audit` harness (`.audit/run_ablation_audit.py`-style runner) against the exact `IMG_5117.MOV` upload used for job 3.
- **Real-video proof:** A fresh run of `IMG_5117.MOV` through the corrected code must produce `confirmed_violations >= 1` where today it produces `0`.
- **DB proof:** A new row in `events` with non-null `event_actor_person_uid`/`event_object_uid` for this run.
- **Dashboard proof:** The confirmed event must render on `/analysis` with a populated evidence strip.
- **Rollback plan:** Single-file, single-constant change; revert by restoring the prior `bag_move_norm`/`bag_below_feet` logic if the loosened margin produces false carries on regression footage.

### REPAIR-02 — WEAKEN TO PROPOSAL/FALLBACK (restore a guarded color-fallback confirmation path)
- **Priority:** P1
- **Problem:** `_is_semantic_waste()` categorically excludes any object the shipped YOLO weights cannot classify (§3.5), including, per the code's own comment, the yellow waste-bag class.
- **Root cause:** New strict gate added in `a30f262` with no re-admission path; the previously-designed fallback-confidence-discount mechanism in `_compute_evidence()` is provably dead code today (per the code's own "DORMANT" comment).
- **Affected files/functions:** `littering_event_detector.py:498-515` (`_is_semantic_waste`), `:2084-2090` (dormant fallback discount).
- **Current behavior:** Color/novelty-only objects can reach `BAG_NEAR_PERSON` telemetry but never a pair, never carry, never confirm.
- **Desired behavior:** A color/novelty-detected object should be permitted into pair evaluation as a genuinely weaker-confidence path, explicitly discounted (the `fallback_factor` mechanism already exists and is fully written — it is just unreachable), not silently dropped.
- **Exact repair strategy:** Relax `_is_semantic_waste()` to admit `source in ("yolo", "color")` (keep `novelty`/`detected_object` excluded, per the user's own confirmed requirement that a raw novelty proposal must never alone become event truth), and confirm the existing `fallback_factor` discount in `_compute_evidence()` (0.90–0.95 multiplier) actually reaches live confidence scoring once color-sourced frames can increment `mem.fallback_frames` again.
- **Regression risk:** Re-opens the color-detector false-positive surface that the strict gate was presumably introduced to close — needs re-validation against whatever false positives motivated the original tightening (not identified in this pass; `NOT PROVEN` why the gate was tightened, since no accompanying test or comment states the motivating false-positive case).
- **Unit/integration/real-video/DB/dashboard proof, rollback:** Mirror REPAIR-01's structure, targeted at a color-only video (re-run the historical `IMG_5117` Aug-31 scenario, i.e. force YOLO detections off, and confirm a violation is still reachable with the discount applied).

### REPAIR-03 — ADD (dependency readiness gate)
- **Priority:** P2
- **Problem:** Two consecutive analysis jobs failed outright with `No module named 'lap'` against a container instance that had not finished initializing.
- **Root cause:** No health/readiness check gates job acceptance on successful import of tracking dependencies; NOT PROVEN whether this recurs on every deploy or was a one-off partial-image state.
- **Affected files/functions:** Backend startup/health-check path (not located this pass — `NOT ASSESSED`, needs a follow-up grep of `backend/` for a `/health` or startup hook).
- **Desired behavior:** The container should fail its own health check (and Compose should not route job-submission traffic to it) until all detector dependencies import successfully.
- **Regression risk:** Low — purely additive.
- **Proof:** Kill/restart the backend mid-rebuild and confirm jobs submitted during that window are rejected/queued rather than failing with an opaque `ImportError`.

### REPAIR-04 — ADD (regression coverage for the exact failure class)
- **Priority:** P1 (companion to REPAIR-01)
- Add the unit test described in REPAIR-01 to the permanent suite so this exact "carried never un-latches near the feet" scenario cannot silently regress again, mirroring how the file's own comment shows this happened once already (the "old rule" the current, still-broken code claims to have fixed).

### KEEP
- The `semantic_bags`/`proposal_bags` split as a *concept* (§6) — correctly enforced, real strength.
- Host/container bind-mount deployment (§4/§8) — verified, eliminates an entire class of drift bugs.
- `AdaptiveEventDetector`'s three-tier relaxation design — reasonable defense-in-depth, simply not the layer where today's bug lives.
- P0-2's UID-based (not index-0) violation matching in `EventDetail.tsx` — correct design, just unproven on real data yet.

### REMOVE-LATER
- The now-fully-dead `_archive/` modules (`state_machine.py`, `voting.py`, root duplicate `pipeline.py`/`person_bag_association.py`, `tracking_visualizer.before/after.py`, `EventEvidenceCollector`) — already archived, not deleted; safe to actually delete once REPAIR-01/02 are proven stable and nobody has depended on the archived copies for a full release cycle.
- Root-level debug/report clutter (`frame_*.jpg`, `*.log`, `call_graph.*`, `backend.svg`, `inference.svg`, the many `*_REPORT.md` files, `adaptive_runs/`, `crops/`) sitting uncommitted at the repo root — not a code risk, but a housekeeping item; several of these were the very "previous claims" this audit found to be contradicted by current behavior, so keep them archived somewhere for history but stop treating their titles ("FINAL_MISSION_REPORT", "FINAL_HARDENING_REPORT") as current status.

---

## 12. Final verdict — the 19 questions, explicitly

1. **What definitely works?** Person detection, YOLO waste-object detection for classes `best.pt` does contain, ByteTrack-based tracking (when the container is fully warmed up), pose-based wrist proximity, the strict semantic-waste/proposal split (proposals cannot become events), stable-UID pairing infrastructure, host==container code identity. All `CODE VERIFIED` and/or `DATABASE VERIFIED` per §3–4.
2. **What definitely fails?** End-to-end confirmation of a real carry→drop→depart littering event (2/2 real candidates in the only available real-video DB record rejected with `NO_RELEASE_TRANSITION`), root-caused in §3.3. `FAILED`, `DATABASE VERIFIED`.
3. **What is historical only?** "139 passed, 0 failed" test claim (not re-run this pass); the Aug-31 "yellow bag confirmed" result (true under the prior commit, not the current one); "P0-1 fixed" (diff-read, not independently re-derived).
4. **What is only unit-tested?** Cannot confirm anything beyond what's visible in test *names* (`test_carry_only_incomplete_rejected`, etc.) without executing the suite, which was not done this pass (no `pytest` on host, and running inside the container was judged out of the strict read-only/no-restart spirit). `NOT ASSESSED`.
5. **What is proven on real video?** Only job 3 and job 4's raw pipeline stages (person/object detection, tracking, association, reaching `BAG_CARRIED`) — via the stored `report_json`. Confirmation itself is proven to **fail**, not succeed, on real video today.
6. **What is proven on Dashboard?** Nothing regarding a confirmed event — there are none to view. `NOT ASSESSED` / `NOT PROVEN`.
7. **Why does the yellow bag fail?** Most likely `_is_semantic_waste()`'s new YOLO-only gate (§3.5) excluding the only detector (HSV color) that ever recognized this class, compounded by (or, on a YOLO-tagged run, purely) the `carried`-latch bug in §3.3.
8. **Why does the red bag fail?** Same `carried`-latch bug (§3.3) is class-agnostic — it will reproduce on any object once carried, regardless of color, the instant the person stands near where they dropped it. Not independently confirmed on red-bag footage this pass (`NOT PROVEN` for the red-bag case specifically, but the mechanism is generic).
9. **Why does NO_RELEASE_TRANSITION happen?** §3.1–3.3, in full: `carried` never un-latches because `bag_below_feet`'s margin is 8x stricter than the ground-detection margin used elsewhere in the same file.
10. **Why can wrong person/object evidence appear?** No confirmed evidence exists to check today, but the rebinding fallback described in §5 (rebind to `other[0]` — any different-class candidate for the same person, absent a same-class one) is a real, code-verified path that could select the wrong physical object under multi-object crowding; not observed to misfire, but structurally possible.
11. **Why are there excessive boxes?** Not assessed this pass — no live/visual session was run (`NOT ASSESSED`). The `proposal_bags`/`semantic_bags` split (§6) suggests debug rendering may intentionally show proposals distinctly from confirmed tracks, which is correct behavior if properly labeled, but this was not visually verified.
12. **Why is the full video shown instead of event evidence?** P1-3/P1-12 claims this was demoted to a "Technical Review (Debug)" panel in the commit message, and the code diff is consistent with that claim, but it cannot be dashboard-verified with zero confirmed events, and the dashboard files have further **uncommitted** changes on top of that commit (§7) whose exact current effect was not diffed this pass.
13. **What should be kept?** §11 KEEP list.
14. **What should be removed (later)?** §11 REMOVE-LATER list.
15. **What should be weakened?** The strict YOLO-only semantic gate → weaken to admit a discounted color-fallback path (REPAIR-02).
16. **What should be replaced?** The `bag_below_feet` ground-exclusion predicate (REPAIR-01) — not removed, replaced with one consistent with `near_ground_plane`.
17. **What should be added?** A dependency-readiness health gate (REPAIR-03) and a regression test for this exact failure geometry (REPAIR-04).
18. **Safest repair order?** REPAIR-01 first (unblocks the entire acceptance scenario; single-constant/predicate change, cheap to test and roll back) → REPAIR-04 alongside it (lock in the fix with a test) → REPAIR-02 (re-admit the color fallback, needs more careful false-positive re-validation since the reason it was tightened is undocumented) → REPAIR-03 (operational hardening, independent of the others).
19. **Is the project production-ready?** **No.** The one behavior the entire system exists to deliver — confirming a real littering event end-to-end with correct evidence — has zero real-world successes in the live database and a code-verified, specific reason why the most obvious test case (a person drops a bag and stands nearby) cannot succeed. Everything downstream of confirmation (evidence, dashboard display, UID correctness on screen) is unprovable until REPAIR-01 lands and produces at least one real confirmed event to inspect.
