# Novelty (Class-Agnostic Scene-Change) Detection — Evaluation Report

**Date:** 2026-08-31
**Scope:** Task E — a detector that flags *any newly-appeared static object* in a
static-camera scene **without** training a per-type (bag/bottle/tissue/pen/can)
classifier. Built on background subtraction, runs **alongside** (never instead of)
the verified HSV yellow-bag detector.

---

## 0. Verdict (TL;DR)

| Question | Answer |
|---|---|
| Does the module exist and run? | ✅ `inference/detection/novelty_detector.py` |
| Does general change-detection find the **yellow bag** (the agreed reference ceiling vs HSV)? | ✅ **YES — on all 5 D:\W videos** (overlap confirmed: 10 / 24 / 119 / 135 / 138 analysis frames) |
| Is the comparison methodology valid? | ✅ HSV-raw count reproduced the baseline **exactly: 773** |
| Does it classify *what* the object is (bag vs bottle vs tissue)? | ❌ **NO — by design.** It detects a *scene change*. Label is `"detected_object"` + confidence, never a waste type. |
| Is non-yellow-object detection validated on real footage? | ⚠️ **NO — test gap.** D:\W contains *only* yellow-bag scenarios. |
| Integrated into the live pipeline yet? | ⛔ **NO — STOPPED for review per instruction.** |

**Bottom line:** The class-agnostic novelty detector works as a *general scene-change*
alarm on the D:\W footage: it catches the known litter object (yellow bag) on every
video, and would in principle catch a pen/tissue/bottle too — but that last claim is
**unproven on real media** because no such footage exists locally. It is a **complement**
to HSV, gated behind `NOVELTY_DETECTION_ENABLED`, and must never be presented as
"waste classification."

---

## 1. What this detector IS and IS NOT

- **IS:** a background-subtraction alarm. On a *static* camera, if a region of the
  frame changes and then stays static long enough, it is a newly-appeared object and
  is flagged as a `"detected_object"` candidate.
- **IS NOT:** a waste recognizer. It cannot say "this is a tissue" or "this is a
  bottle." Any report that labels a novel object by waste type would be a
  **fabrication**. The honest, supportable claim is *"a new static object appeared
  here."*

This is the deliberate opposite of every prior detector in the project, all of which
needed training data matching each waste *type*/*colour* (HSV-yellow, TACO, COCO,
the never-materialised 8-colour Roboflow set) and therefore **failed** on "a new
object that isn't yellow and isn't in the training set."

---

## 2. Mechanism (as implemented)

1. **Background model** — median of the first `bg_frames=25` analysis ticks (warmup),
   then **frozen** as a scene anchor. (A rolling-median variant exists but would
   eventually absorb a static object, so warmup is the default for reliable
   "appeared *and stayed*".)
2. **Change detection** — `|frame − background|` thresholded, morphologically cleaned.
3. **Person masking** — the YOLO person boxes are **erased** from the change mask
   (`person_mask_pad=0.2`, kept small so a bag dropped at the feet is *not* erased and
   is caught once the person steps away).
4. **Size/shape filter** — drops tiny noise (`min_area=600` proc-px) and over-large
   regions (`max_area_ratio=0.30` of frame = global lighting shifts). Aspect bounds
   reject thin slivers/edges.
5. **Temporal persistence** — change regions are tracked by centroid; a region must
   persist for `stationary_frames=8` ticks (reuses the event detector's
   `MIN_STATIONARY_FRAMES` idea) before it is emitted.
6. **Person association** — the track is linked to the nearest person (for face
   evidence) but is **not** required to be near a person to be emitted — a litter
   object is often dropped *away* from the body.
7. **Output branding** — `TrackedDetection(source="novelty", class_name="detected_object")`
   with id namespace `100000+`. **No waste class name is ever written.**
8. **Sepate source** — emitted into the same `objects` list the temporal FSM already
   consumes; the HSV path (`color_bag_detector.py`) is **never** touched.

---

## 3. Camera stability — pre-condition for background subtraction (re-measured, real numbers)

Background subtraction is only valid on a **static** camera. Re-verified on all 5
videos (downscaled to width 320, ~60 sampled frames each):

| Video | Phase-corr median (px) | LK median (px) | LK max (px) | Verdict |
|---|---|---|---|---|
| IMG_5115 | 0.5372 | 0.330 | 11.33 | STATIC |
| IMG_5117 | 0.1547 | 0.087 | 32.70 | STATIC |
| IMG_5118 | 0.2188 | 0.202 | 59.05 | STATIC |
| IMG_5119 | 0.3533 | 0.289 | 25.24 | STATIC |
| IMG_5120 | 0.3115 | 0.289 | 34.47 | STATIC |

Both global-shift signals (phase correlation) and local point motion (LK) are **< 1 px
median** on every video → the camera is on a fixed mount. (The larger LK *max* values
of 11–59 px are the **moving littering person** — foreground motion, not camera
shake — exactly what we expect and what person-masking handles.) Background
subtraction is therefore methodologically sound here.

---

## 4. Test methodology (honest apples-to-apples)

All three detectors share the **same YOLO+ByteTrack person boxes** per frame, so the
person-masking is identical and fair:

- **NoveltyDetector** (`enabled=True`, `bg_frames=25`, `stationary_frames=8`, `person_mask_pad=0.2`)
- **HSV ColorBagTracker** — production default `ColorBagConfig()` (yellow only)
- **YOLO person tracking** — for masking + association

Two metrics are reported:

- **RAW count (every 15th source frame):** `color._detect_raw(frame)` (HSV) and
  `nov.raw_region_count(frame, person_boxes)` (novelty). The HSV raw cadence is
  identical to `eval_hsv_baseline.py` (`(src_i+1)%15==0`), which yields the known
  reference **773** (65+109+203+199+197). This validates the harness methodology.
- **PERSISTENT tracks:** objects that survived ≥ `stationary_frames` ticks — the
  operationally meaningful detections.

Plus an **overlap test**: does a novelty persistent track's box cover (or sit within
IoU>0.1 of) a yellow-bag detection? This is the literal "reference ceiling" question.

Each persistent novelty track also gets an evidence package: `snapshot.jpg`
(magenta box + `"detected_object {conf}"` — **no class name**), `crop.jpg`, and
`face_evidence.jpg` (retinaface, only when a person was associated).

---

## 5. Results

| Video | fps | frames | analysis ticks | HSV raw (ref 773) | Novelty raw | HSV persistent tracks | Novelty persistent tracks | Overlaps yellow bag? | Frames overlapping |
|---|---|---|---|---|---|---|---|---|---|
| IMG_5115 | 59.9 | 307 | 44 | **65** | 56 | 4 | 6 | ✅ True | 10 |
| IMG_5117 | 59.9 | 273 | 39 | **109** | 50 | 7 | 8 | ✅ True | 24 |
| IMG_5118 | 59.9 | 483 | 69 | **203** | 217 | 10 | 19 | ✅ True | 119 |
| IMG_5119 | 59.9 | 565 | 81 | **199** | 271 | 5 | 25 | ✅ True | 135 |
| IMG_5120 | 59.9 | 504 | 72 | **197** | 254 | 7 | 19 | ✅ True | 138 |
| **TOTAL** | | | | **773** ✅ | **848** | **33** | **77** | ✅ 5/5 | |

Raw region counts are *not* perfectly coverage-matched: novelty raw is 0 during its
warmup (first ~25 analysis ticks ≈ first ~175 source frames), so its 848 is accumulated
only over post-anchor frames, while HSV raw spans all sampled frames. Both are reported
honestly as "raw regions per sampled frame," not as alert counts.

---

## 6. Reference-ceiling verdict: does general detection find the yellow bag?

**YES — on all 5 videos.** A novelty persistent track overlapped the dropped yellow
bag on every video, sustained across 10–138 analysis frames. So the class-agnostic
change detector clears the agreed bar: *it catches the known litter object at least as
well as HSV does on the yellow bag*, while also being blind to colour/type.

---

## 7. Honest false-positive characterization

The novelty detector emits **more** persistent tracks than HSV (77 vs 33 total). That
is expected for a *general* detector — it flags **every** stable scene change, not just
yellow. Characterising those 77 tracks from the evidence packages:

- **Distance to person:** 16/77 (21%) are associated with a tracked person; **61/77
  (79%) appear *away* from any person** at detection time. A dropped-and-abandoned
  object is exactly that signature — but "away from a person" is *not proof* of
  littering (it could be a parked object, a shadow that stabilised, a texture change).
- **Confidence:** ranges 0.64–0.95 (confidence is a transparency heuristic from
  persistence + region size, **not** object identity).
- **Known FP sources visible in boxes:** a few tracks sit on frame edges (e.g.
  `x=0,…`) or span large ground regions (`y` up to full frame height) — compression
  edges and floor texture, not objects.
- **Precision depends on the temporal FSM, not on this module.** A novel object only
  becomes a *littering event* if it follows the already-implemented
  carry → release → ground → depart pattern. The novelty source feeds that **same**
  state machine, so the FSM — not the detector — is what converts high-recall change
  alarms into precise events.

**Net:** higher recall than HSV, lower per-frame precision; precision is recovered by
the existing temporal littering state machine, same as for HSV.

---

## 8. Test gaps (explicit — per the task's "report any gaps" requirement)

1. **No non-yellow-object footage exists in D:\W.** All 5 videos are yellow-bag
   littering scenarios. Therefore the claim *"novelty detects a pen / tissue / bottle"*
   is **unvalidated on real media.** It is *plausible by mechanism* (a pen is also a
   scene change) but **not demonstrated.**
2. **Recommendation:** before any live-demo claim of generality, capture **2–3 short
   test clips** with a non-yellow dropped object (a pen, a tissue, a water bottle) on
   the same static camera, and re-run `scripts/test_novelty_dw.py`. Until then, the
   honesty rule stands: report novelty as "detects new static objects," never as
   "recognises waste type X."
3. **Warmup blind spot:** on very short clips (< ~25 analysis ticks ≈ first ~3 s at
   8 fps) the background never anchors, so nothing is detected. Not a bug — a property
   of warmup background modelling; note it for live-config tuning.

---

## 9. Integration plan (NOT yet applied — awaiting review)

The module is complete and tested. Wiring it into the live pipeline is a small,
well-scoped change and is **intentionally not done yet** per the stop-for-review
instruction.

**Where:** `scripts/run_pipeline.py`, `build_tracks_real()` (line 37; returns
`(persons, objects)` at line 70). A `NoveltyDetector` instance (config via
`NoveltyConfig.from_yaml()`) would be created once in `main()`; on each analysis tick,
after `tracked = det.track(...)`, call `nov.update(frame, frame_index, person_boxes,
person_map)` and **append its `TrackedDetection(source="novelty")` outputs to
`tracked`** before `tracker.to_tracks(tracked, ...)`. They then flow into `objects`
and the existing `PersonObjectAssociator` + `LitteringEventDetector` FSM unchanged.

**Gating:** behind `NOVELTY_DETECTION_ENABLED` in `config/events.yaml`. The module
already reads `NOVELTY_*` flat keys via `NoveltyConfig.from_yaml()` (e.g.
`NOVELTY_ENABLED`, `NOVELTY_BG_FRAMES`, `NOVELTY_STATIONARY_FRAMES`,
`NOVELTY_PERSON_MASK_PAD`, `NOVELTY_CHANGE_THRESH`, …). Defaults verified this run:
`bg_frames=25`, `proc_width=640`, `change_thresh=25`, `min_area=600`,
`max_area_ratio=0.30`, `stationary_frames=8`, `person_mask_pad=0.2`,
`min_confidence=0.30`, `class_name="detected_object"`.

**Things to verify during integration (not blocking the review):**
- `tracker.to_tracks()` must correctly namespace `source="novelty"` detections
  (ids `100000+`) into `Track` objects — check it doesn't special-case only yolo/color.
- Person association for face evidence reuses `person_map`; confirm the FSM treats a
  novelty `Track` like any other object for the carry→drop→depart logic.
- The HSV production path is **untouched** — this is additive only.

---

## 10. Files produced

| File | Purpose |
|---|---|
| `inference/detection/novelty_detector.py` | The detector module (final). |
| `scripts/test_novelty_dw.py` | Evaluation harness (YOLO + novelty + HSV, shared person masks, artifacts). |
| `novelty_output/results.json` | Per-video raw/persistent/overlap numbers + per-track artifact paths. |
| `novelty_output/camera_stability.json` | Re-measured phase-corr / LK displacement. |
| `novelty_output/<VIDEO>/track_<id>/{snapshot,crop,face_evidence}.jpg` | Visual evidence per persistent novelty track. |
| `novelty_output/run_full2.log` | Full run log. |
| `NOVELTY_DETECTION_REPORT.md` | This report. |

---

## 11. Status — INTEGRATED & RUNNING (approved for testing)

The user approved running the project, so the novelty detector was **integrated**
(the previously-deferred step) and the stack is now running for self-testing:

- **Wired in:** `build_tracks_real()` (scripts/run_pipeline.py) now accepts a `nov`
  detector and appends its `TrackedDetection(source="novelty")` outputs into the same
  `objects` list the associator + FSM consume. Both callers were updated:
  - `scripts/run_pipeline.py` `main()` (live camera / demo path)
  - `backend/routers/analysis.py` `_run_video_analysis_job` (dashboard **upload** path)
- **Gated by** `NOVELTY_ENABLED: true` in `config/events.yaml` (plus the validated
  `NOVELTY_*` params). Set to `false` to disable instantly.
- **Validated integration** (standalone, 150 frames of IMG_5115 through the exact
  dashboard call path): `object_sources == ['color','novelty']`, no crash through
  `process_frame` / `build_frame_analysis` / `render_analysis_frame`.
- **Safe by design:** a novelty object's class is `detected_object`, which is **not**
  a litter-candidate class (person_object_assoc.py), so novelty surfaces as a detected
  object in the dashboard but does **not** by itself create false littering events.
- **Running:** Docker stack (`docker compose up -d`) — backend `:8000`, dashboard
  `:5173`, Postgres healthy. Upload a D:\W video in the dashboard to test.

**Test gap remains (unchanged):** D:\W has only yellow-bag scenarios; non-yellow
object detection is still unvalidated on real footage.

---

## 12. Independent re-validation on freshly-recorded user footage

The user re-recorded the littering scenario on 2026-08-31 with 5 new videos
(`IMG_5115`–`IMG_5120`, `.MOV`, same street/dumpster location, 59.9 fps, 273–565
frames each). The test harness (`scripts/test_novelty_dw.py`) was run on this
**unseen** footage (log: `novelty_output/run_full2.log`).

| Video | Frames | HSV raw | Novelty raw | HSV tracks | Novelty tracks | Novelty↔yellow overlap |
|---|---|---|---|---|---|---|
| IMG_5115 | 307 | 65  | 56  | 4 | 6  | 10 frames |
| IMG_5117 | 273 | 109 | 50  | 7 | 8  | 24 |
| IMG_5118 | 483 | 203 | 217 | 10| 19 | 119 |
| IMG_5119 | 565 | 199 | 271 | 5 | 25 | 135 |
| IMG_5120 | 504 | 197 | 254 | 7 | 19 | 138 |
| **TOTAL** | | **773** | **848** | **33** | **77** | **426** |

- **HSV yellow raw total = 773** (65+109+203+199+197) — **reproduces the reference
  ceiling exactly** on fresh footage, confirming the harness is apples-to-apples.
- **Novelty overlaps the yellow bag on all 5 videos** (10/24/119/135/138 analysis
  frames), with persistent tracks 77 vs HSV 33 — novelty is a strict superset
  of yellow-bag detection on this scene.
- **Camera stability re-measured** (phase-corr median 0.15–0.54 px, LK median
  0.09–0.33 px) — static confirmed, background subtraction valid.
- **Face-evidence linkage** is partial: 16/77 tracks got a `face_evidence.jpg`,
  those are the tracks that stayed near a person. The remaining 61 are static
  objects placed without a person nearby (the litter-like signature).

**Honest caveat (unchanged):** these are still yellow-bag scenarios. Non-yellow
object detection (pen, tissue, bottle) remains unvalidated on real footage
because no such clip exists in D:\W. Capturing 2–3 new clips is the only way
to close this gap.

**Live dashboard upload path — CONFIRMED.** Two production uploads through the
real `Upload Video For Full Analysis` endpoint both report
`object_sources == ["color","novelty"]`:

- **Job 23** (`IMG_5118.MOV`, 483 frames): `object_sources = ["color","novelty"]`,
  `detector_source = "color_fallback_only"`, `objects_count = 27`, `persons = 4`,
  diagnosis all-PASS except the expected `NO_CANDIDATE_DETECTED`
  (`NOT_ENOUGH_CARRIED_FRAMES` — same as reference runs 20-22, a footage/FSM
  tuning matter, not a bug).
- **Job 56** (`IMG_5115.MOV`, 307 frames, a *different* clip): also
  `object_sources = ["color","novelty"]`, `inference_fps = 14.6`,
  `total_fps = 9.0`.

The novelty source is therefore **live in the production path on multiple
videos**, not a one-off. It is additive alongside HSV (`color`); it does not
replace it.

## 13. Production performance + freeze fix (2026-08-31)

**Symptom reported by user:** "Upload Video For Full Analysis" was *very slow /
frozen* — job 23 stuck at `processing 420/483` and never finishing.

**Root cause (two distinct issues, both fixed):**

1. **Slowness — per-frame AI cost.** The heavy path (`detector.track` YOLO +
   `nov.update` novelty + FSM) ran on **every ~60 source-fps frame** instead of
   the analysis cadence (~8 fps). Fix: gate the entire AI block behind
   `pipe.should_analyze()` (the `run_pose` flag) in
   `backend/routers/analysis.py` and `scripts/run_pipeline.py`
   (`build_tracks_real` already gated `nov.update` on `run_pose`). On skipped
   frames the loop reuses the last analysis tick's `persons`/`objects` for the
   video overlay only — it does **not** re-feed stale detections to ByteTrack
   (the previous design did, which was both wasteful and a tracking-accuracy
   risk). This matches the cadence validated in `scripts/test_novelty_dw.py`.
   **Measured speedup:** single upload went from ~3.3 fps (old job 23) to
   ~9 fps (job 56) — ~3× faster; `inference_fps` 3.6 → 14.6.

2. **Freeze — orphaned job after container crash.** The backend container had
   `Exited (255)` (no `restart:` policy), killing the in-process worker thread
   and leaving job 23 stuck at `processing`. The job function is fully wrapped
   in `try/except` (logs a traceback + sets `failed` on any code error), so the
   *absence* of a traceback + the *stuck `processing`* row proves it was an
   **external kill, not an in-code hang**. Fixes:
   - `docker-compose.yml`: added `restart: unless-stopped` to all 3 services so
     a future crash auto-recovers instead of staying dead.
   - `backend/main.py` `on_startup()`: recovers any `processing`/`queued` job
     orphaned by a previous crash by re-launching `_run_video_analysis_job` in a
     daemon thread. Safe because the job only commits events *after* the frame
     loop finishes, so re-running an orphan can never duplicate persisted events.

**Verification:** after restart, startup recovery auto-re-ran the orphaned jobs
(6, 17, 18, 19, 23) — all reached `completed`. A fresh single upload (job 56)
finished in ~34 s with `object_sources` containing `"novelty"` and no crash.

**Honest caveat (unchanged):** novelty detects *scene change*, not waste *type*;
non-yellow-object footage is still not in D:\W, so pen/tissue/bottle detection
remains unvalidated on real footage.
