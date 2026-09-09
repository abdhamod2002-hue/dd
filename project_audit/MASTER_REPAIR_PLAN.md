# MASTER REPAIR PLAN — Littering AI Forensic Architecture Audit

**Audit date:** 2026-09-09 · **Scope:** Read-only. No files edited, no training, no DB/Docker mutations, no new video analysis run.
**Method:** Six parallel read-only forensic agents each independently verified a slice of the system against **current source** (`D:\HO`, uncommitted working tree — confirmed to be exactly what the running `littering-backend` container executes, `docker inspect StartedAt` today, bind-mounted, no `--reload` so code is frozen at container-start) and, where safe, against the **live running stack** (`littering-backend`, `littering-postgres`, `littering-dashboard`, all up ~6h at audit time) via read-only `curl`/`docker exec`/`psql` queries. This document merges their findings. Every claim below is tagged:

- **VERIFIED FROM CURRENT SOURCE** (file:line)
- **VERIFIED FROM RUNTIME** (command + output)
- **VERIFIED FROM DATABASE** (query + result)
- **HISTORICAL** (named report/markdown — treated as a lead, not truth)
- **INFERRED** (reasoning without direct proof)
- **NOT VERIFIED** (checked, could not confirm)

No category is silently upgraded into another anywhere in this document.

---

## 1. Executive Summary

The project is a real, working, non-trivial CV pipeline — YOLO detection, ByteTrack tracking, a genuinely sophisticated hand-built temporal FSM (`littering_event_detector.py`, 2184 lines) with stable-identity resolution, ownership locking, regrab handling, and an evidence-packaging layer. Large parts of the "hard" design work are already correct **in source**: the semantic-waste gate (`_is_semantic_waste`) really does keep HSV/Novelty proposals out of event decisions; the person/object identity managers really do defend against ID churn; the regrab→`PICKED_BACK_UP` path is unit- and integration-tested and works.

But the system is **not production-ready**, for reasons that are almost all about the *seams* between components, not the components themselves:

1. **Every single event in the live production database (30/30) has NULL stable actor/object identity columns.** Two distinct, fully root-caused bugs cause this — one in the live-camera ingestion schema (100% reproducible, structural), one in the adaptive multi-tier event detector's own deduplication logic (a self-matching bug that also silently discards the entire evidence package for any event confirmed by a non-primary tier).
2. **The dashboard page a human reviewer actually uses to triage a flagged event (`EventDetail.tsx`) has a live, exploitable cross-event attribution bug** — it shows event index `[0]` of the whole video's report regardless of which event the reviewer opened. Confirmed exploitable today against 6 real job/event pairs in production.
3. **The event-creation API has no provenance check** — any HTTP client can `POST /api/events` and create a "confirmed" violation indistinguishable from a real AI detection. This was silently exercised by a leftover throwaway script and is sitting in the production DB as row 29.
4. Bin-vs-ground detection **does not exist** — it's a single height-above-feet-line heuristic tuned to one validation video. A low ground-level bin will confirm as a false-positive litter event.
5. Evidence storage is genuinely split between a Docker named volume (what the dashboard actually serves) and a host-visible folder (stale, divergent, actively misleading to anyone auditing by hand).
6. The single biggest CPU cost — three full YOLO forward passes on every source frame, unconditionally — has not been addressed by any of the (real, measured) performance work done so far; that work correctly fixed a different bottleneck (Novelty running every frame) but left the dominant one untouched.

None of this requires new AI research. It requires fixing specific, located, already-diagnosed defects in a specific order (Section 20). The detection/tracking/FSM core is the strongest layer of this system; the persistence↔evidence↔dashboard seam is the weakest.

---

## 2. What This Project Actually Is

**In one sentence:** a system that watches a video, tracks every person and every waste-looking object in it, and tries to answer one narrow legal-style question per (person, object) pair — *did this specific person carry this specific object and then abandon it on the ground* — and, when the answer is yes, packages proof (crops, a clip, a face image) that a human reviewer can trust points at the right person and the right object.

1. **What it does:** ingests a video (uploaded file, or a live camera feed), detects people and candidate waste objects frame by frame, tracks each one across frames, resolves tracking churn into a small number of *stable* identities, runs a temporal state machine per (person, object) pair to decide whether a littering event happened, and — only for confirmed events — extracts image/video evidence and writes it to a database a React dashboard reads.
2. **The problem it solves:** raw object detection tells you "there is a bag in this frame." It does not tell you *who* put it there, *whether* they carried it first, or *whether* someone innocent will get blamed because they happened to be standing nearby. This project's actual engineering content is the middle 90% — turning noisy per-frame detections into one attributed, temporally-justified accusation.
3. **What the AI is supposed to understand:** not "is this a bag" (that's the easy part, solved by a fine-tuned YOLO model), but "did carrying → releasing → abandoning happen, by this exact tracked person, to this exact tracked object, as opposed to a bystander, a bin-deposit, or a drop-then-pickup."
4. **Backend responsibility:** owns the two production entry points (`backend/routers/analysis.py` for uploaded video, a standalone `scripts/run_pipeline.py` for a live camera), runs the shared inference core, persists confirmed events + evidence to Postgres, and exposes it all over a FastAPI REST surface.
5. **Dashboard responsibility:** upload a video and watch its progress; browse job history; open a specific confirmed event and see who/what/when plus supporting evidence; a live-monitoring view; a (currently non-functional) settings view.
6. **Database responsibility:** the single source of truth for `cameras`, `video_analysis_jobs`, `events`, `evidence`, `users` — Postgres 16 in production, matching the ORM schema in `backend/models.py`.
7. **Layer 1 (Detection/Tracking/Identity):** turns pixels into `(person_uid, object_uid)` pairs that survive ID churn, occlusion, and re-detection. Does **not** decide anything about littering.
8. **Layer 2 (Temporal Event Recognition):** `LitteringEventDetector` (wrapped, in production, by a 3-tier `AdaptiveEventDetector`) walks each pair through `NO_BAG → BAG_NEAR_PERSON → BAG_CARRIED → BAG_RELEASED → BAG_ON_GROUND → PERSON_DEPARTED → VIOLATION_CONFIRMED` (or `PICKED_BACK_UP`), freezing the actor/object identity at the moment carrying is first established so a later ID switch can't swap in a different "actor."
9. **Layer 3 (Evidence):** for confirmed events only, re-opens the *original, unannotated* video and cuts a person crop, an object crop, a face crop, a time-bounded clip, and (unreachably, currently) carry/release/ground sequence frames — then writes them to disk and a DB row.
10. **Video upload → dashboard, end to end:** `VideoAnalysisPage.tsx` uploads → `POST /api/analysis/upload` saves the file and schedules a background job → `_run_video_analysis_job()` runs the full per-frame detect→track→identity→FSM→render loop, encodes the annotated video, and for each confirmed event calls the evidence-package writer → writes `Event`/`Evidence`/`VideoAnalysisJob` rows → the dashboard polls job status, then reads events/evidence/manifest back over the REST API.

---

## 3. Product Goal (restated against verified reality)

The stated goal — identify the true actor, the true object, the true timestamp, and expose only event-focused evidence — is **the correct design intent and is largely implemented in the temporal engine** (`_select_primary_associations`, the carry-time identity freeze, the `other_person_closer` ambiguity guard). It is **not yet honored end-to-end**: the frozen identity that Layer 2 computes correctly is dropped or ignored at three different downstream points (live-camera schema, adaptive-tier dedup bug, dashboard's non-scoped event lookup) before it reaches a human reviewer. See Section 12/21.

---

## 4. Current Architecture

Two production entry points converge on one shared inference core.

### 4A. Video-upload path (the one `/analysis` and the sample videos in `D:\22` exercise)

| # | Hop | File:Line | Status |
|---|---|---|---|
| 1 | Upload UI | `dashboard/src/pages/VideoAnalysisPage.tsx` | Active |
| 2 | `POST /api/analysis/upload` | `backend/routers/analysis.py:928-972` | Active |
| 3 | Background job orchestrator (the real entry point, ~570 lines) | `backend/routers/analysis.py:357-926` (`_run_video_analysis_job`) | Active |
| 4 | Video source | `inference/capture/camera_source.py` | Active |
| 5 | Detection (3 YOLO models + HSV fallback) | `inference/detection/yolo_detector.py:316-421` | Active, every frame |
| 6 | Cross-model dedup | `yolo_detector.py:100-150` | Active |
| 7 | Track adapter | `scripts/run_pipeline.py:38-114` | Active |
| 8 | ByteTrack namespacer | `inference/tracking/bytetrack_tracker.py:49-158` | Active |
| 9 | Pose (throttled) | `inference/pose/movenet_pose.py:173-193` | Active, analysis-tick only |
| 10 | Novelty proposal | `inference/detection/novelty_detector.py` | Active, additive/non-authoritative |
| 11 | Pipeline orchestration | `inference/pipeline.py:173-242` | Active |
| 12 | **Event engine (3-tier adaptive wrapper around the FSM)** | `adaptive_tuner.py:307-563` wrapping `littering_event_detector.py:800-2006` | **Active, authoritative** |
| 13 | Stable identity resolution | `littering_event_detector.py:837,844,892-924` → `person_identity.py`, `object_identity.py` | Active |
| 14 | Evidence (live-buffer snapshot) | `inference/evidence/evidence_manager.py:100-179` | Active |
| 15 | Frame renderer → analyzed video | `inference/visualization/frame_analysis.py`, `tracking_visualizer.py`, `h264.py` | Active |
| 16 | End-of-video flush | `inference/pipeline.py:373-389` | Active |
| 17 | DB write — Event | `backend/routers/analysis.py:665-702` | Active in code; **populated fields are NULL in production, see §11/§21 P0-1** |
| 18 | Evidence package (Layer 3) | `inference/visualization/evidence_package.py:235-438` | Active |
| 19 | Face capture (geometric heuristic, not ML) | `inference/evidence/face_evidence.py:50-264` | Active but not real face detection |
| 20 | DB write — Evidence | `backend/routers/analysis.py:752-763` | Active |
| 21 | Manifest/report | `analysis.py:801-910` | Active |
| 22 | Job/event API | `analysis.py:1093-1235` | Active |
| 23 | Dashboard render | `VideoAnalysisPage.tsx:470-514` | Active — reads the in-memory `report_json`, not the persisted DB columns |

### 4B. Live-camera path (`scripts/run_pipeline.py`, drives `LiveMonitoring.tsx`/`LiveCamera.tsx`)

Shares steps 5-13 above (same detector/tracker/pipeline/FSM — confirmed genuinely one engine, not two implementations). Diverges at persistence: `inference/pipeline.py:391-463` (`_maybe_post_backend`) POSTs to `backend/routers/events.py:103-133` (`create_event`) using `schemas.EventCreate`, which **structurally has no stable-identity fields at all** (Section 21, P0-1a). Not running at audit time (code-verified only, not runtime-verified).

### 4C. Shared DB → API → Dashboard leg

`models.Event`/`models.Evidence` → `backend/routers/events.py`/`evidence.py` → `dashboard/src/lib/api.ts` → `Dashboard.tsx`, `Violations.tsx`, `EventDetail.tsx`. `EventDetail.tsx` reads only the legacy `event.person_track_id`, never the stable UID fields, even where they exist (Section 21, P1-12).

---

## 5. Current AI Stack (algorithm inventory)

| Algorithm | Role | Status | Drives events? | Drives evidence? |
|---|---|---|---|---|
| YOLO person detector (`yolov8n.pt`) | COCO person class | **Active/production** | Yes (no persons ⇒ no pairs) | Yes |
| `best.pt` | Custom litter classes (bottle/cup/etc.) | **Active** | Yes | Yes |
| **`garbage_bag_v2.pt`** | Production waste-bag model | **Active — THE production model** | Yes (primary) | Yes |
| `waste_bag_real_v1.pt` | Rejected model (0-0.5% agreement with independent ground truth, trained with zero negatives) | **Confirmed dead** — no code path ever loads it | No | No |
| `taco_transfer_v1.pt` | Independent eval-only ground-truth model | Offline tooling only | No | No |
| HSV / `ColorBagTracker` | Pixel-color fallback proposal | **Active fallback, proposal-only by design and by gate** | No (excluded by `_is_semantic_waste`) | No |
| `NoveltyDetector` | Class-agnostic background-subtraction proposal | **Active, additive, non-authoritative by design and by gate** | No (excluded) | No |
| ByteTrack (algorithm) + `BytetrackTracker` (wrapper) | Raw tracking + ID namespacing | Active | Indirectly (supplies all downstream IDs) | Indirectly |
| `PersonIdentityManager` | Stable person UID across churn | **Active, authoritative** | Yes | Yes |
| `ObjectIdentityManager` | Stable object UID across churn | **Active, authoritative** | Yes | Yes |
| MoveNet | Pose (wrist/shoulder/torso) | Active, throttled | Yes (carry/release wrist signal) | No |
| **`PersonObjectAssociator`** | Described in its own docstring as "🔴 core contribution" | **Dead code — instantiated (`pipeline.py:98`), `.update()` never called** | No | No |
| **`LitteringEventDetector`** | Per-pair temporal FSM | **Active, authoritative** | **Is** the decision | Gates evidence eligibility |
| **`AdaptiveEventDetector`** | 3-tier concurrent wrapper + cross-video learning store | **Active, always-on in production** (both entry points hardcode `auto_tune=True` despite the dataclass default being `False`) | Yes — this, not bare `LitteringEventDetector`, is the real production engine | Yes |
| `state_machine.py` / `voting.py` (legacy FSM/voter) | Pre-`LitteringEventDetector` implementations | **Dead — only their own unit tests import them** | No | No |
| Face detection (`FaceEvidenceCapture`) | "region" backend = fixed geometric head-box heuristic, NOT ML face detection | Active, but not what its name implies (`retinaface`/InsightFace backend exists but the production container doesn't ship the dependency) | No | Yes (only after confirmation) |
| Evidence scoring | Weighted carry/release/stationary/departure/association formula | **Active, authoritative**, gates confirmation at `min_event_confidence=0.80` | Yes | N/A |
| H.264 transcode | `mp4v` → browser-playable via `ffmpeg` subprocess | Active — up to 3 separate transcodes per confirmed event | No | Yes |
| Rendering/overlay | Draws current-frame boxes only (verified: no historical-box accumulation in current source) | Active | No (read-only visualization of FSM state) | No |
| `EventEvidenceCollector` | A second, self-contained evidence writer | **Dead — zero callers found** | No | No |
| Root-level `pipeline.py` + `person_bag_association.py` | A separate, unused duplicate pipeline | **Dead — neither production entry point imports it** | No | No |

**Detector semantics (verified as sound):** `_is_semantic_waste()` (`littering_event_detector.py:347-364`) requires `source=="yolo"` and rejects `color_candidate_*`/`detected_object` class names. This gate is applied before pair evaluation, dedup, and evidence assembly, and is the *only* call path used by every production entry point. No current-source path was found by which an HSV or Novelty proposal becomes "waste," drives an event, or becomes selected evidence. (One caveat: historical DB rows with `object_type="detected_object"`/`color_candidate_black"` exist as confirmed events from Aug–Sep runs; whether these predate this gate being added to the uncommitted working tree, or reveal a residual leak, was **not fully resolved** — flagged as a targeted re-verification item, not a confirmed current bug.)

---

## 6. Current Backend

FastAPI app (`backend/main.py`), routers: `analysis`, `cameras`, `events`, `evidence`, `statistics`, `status`, `stream`. Live DB confirmed to be genuinely served (statistics/status/events all return real computed data). Key structural facts:

- `POST /api/events` (`events.py:103-133`) accepts `schemas.EventCreate`, which has **no stable-identity fields** and **no provenance check of any kind** — see Section 21, P0-3.
- `get_analysis_manifest`/`get_analysis_job_events` don't declare a `response_model` — future field renames won't be caught by any contract.
- Evidence file resolution: `evidence.py` path-traversal-guarded (`_safe_resolve`), but `GET /api/evidence/event/{id}/snapshot|video` picks the **oldest** evidence row (`order_by(id.asc()).first()`) despite the docstring claiming "most recent" — low impact today since almost every event has exactly one evidence row.
- Docker volume topology (`docker-compose.yml:36-44`): `./:/app` (bind mount, source) with `evidence_store:/app/evidence_store` and `uploaded_videos:/app/backend/uploaded_videos` as **named volumes that shadow the bind mount at those subpaths**. Confirmed at runtime: host `D:\HO\evidence_store`/`backend\uploaded_videos` and the container's actual served paths **genuinely diverge** (43 vs 33 uploaded files; host evidence_store full of orphaned hex-hash directories that don't correspond to any current event).

---

## 7. Current Frontend

| Page | Data | Verdict |
|---|---|---|
| Dashboard / Command Center | `/api/cameras`,`/api/events`,`/api/statistics`,`/api/status` | Real API data |
| VideoAnalysisPage (upload+result) | Live polling + report JSON | Real API data; **correctly** structured (Primary Event Evidence first, full video demoted to a labelled "Debug"/Technical Review section) — this is the example the rest of the app should copy |
| AnalysisDetail (job history detail) | `/api/analysis/jobs/{id}/*` | Real API data |
| **EventDetail** (opened from Violations — the real reviewer triage page) | `/api/events/{id}/review` | Real API data, but **`confirmed_violations[0]` bug** (Section 21, P0-2) and **full video shown with equal prominence to the event clip**, unlike VideoAnalysisPage (Section 21, P1-3) |
| EvidencePage (gallery) | `/api/events` + `/api/evidence` per event | Real API data; thumbnail lookups capped to first 30 events |
| LiveMonitoring | `/api/cameras`,`/api/status` | Real API data; `offline` is an honest default, not mock |
| **Settings** | **hardcoded `const SETTINGS = [...]` array** | **100% mock — never fetches config**, can silently drift from the real deployed thresholds |

No global mock-data-fallback pattern was found (`useFetch` has no fallback path) — Settings is an isolated case, not a systemic pattern.

---

## 8. Current Database

**PostgreSQL 16** (confirmed live). Schema (`cameras`, `events`, `evidence`, `video_analysis_jobs`, `users`) matches `backend/models.py` exactly, including the Phase-C/D stable-identity columns. Live query result (VERIFIED FROM DATABASE):

```
SELECT count(*), count(event_actor_person_uid), count(event_object_uid),
       count(event_actor_person_track_id), count(event_object_track_id), count(analysis_job_id)
FROM events;
→ total=30 | actor_uid=0 | object_uid=0 | actor_track=0 | object_track=0 | has_job_id=27
```

**Every event, including the newest, has NULL in all four stable-identity columns.** Root causes are fully diagnosed in Section 21, P0-1. No orphaned evidence rows (`evidence LEFT JOIN events ... WHERE events.id IS NULL` → 0). Six job IDs currently have 2 events sharing one `report_json` (88→{27,28}, 87→{25,26}, 84→{22,23}, 78→{19,20}, 77→{17,18}, 66→{14,15}) — this is the precondition that makes P0-2 (EventDetail's `[0]`-indexing bug) live-exploitable today, not theoretical. A separate local `littering_demo.db` (SQLite, 0 rows) has a schema that **predates** the stable-identity columns entirely and was never migrated — no Alembic migrations exist in this repo; schema changes rely on `create_all`, which does not alter existing tables.

---

## 9. Current Evidence System

| Artifact | Real? | DB column | API route | Dashboard |
|---|---|---|---|---|
| `snapshot.jpg`, `evidence.mp4` | Yes | Yes | Yes | Yes |
| `person.jpg`, `waste.jpg` | Yes | Yes | Yes | Yes |
| `face_evidence.jpg` | Yes (geometric heuristic, not ML) | Yes | Yes | Yes |
| `event_clip.mp4` (correctly time-bounded to carry→departure ±5s, from the clean original video) | Yes | Yes | Yes | Yes |
| **`carry.jpg`/`release.jpg`/`ground.jpg`** | Yes, code writes them | **No column** | **Not served** | **Never rendered** |
| `analyzed.mp4` (full video, all tracks) | Yes | Yes | Yes | Yes — correctly demoted to "Debug" on VideoAnalysisPage, **not** demoted on EventDetail |
| `manifest.json`, `frames.jsonl` | Yes | Yes (manifest) | Yes | manifest partially read; frames.jsonl fetched by no page |

Selection logic **is** genuinely anchored to the frozen carry-time actor/object identity in source (`evidence_package.py:41-67`, exact `track_id`/`object_uid` match, never "nearest"/"largest"/"first"). But because of P0-1, **100% of real historical events fell back to the legacy churn-prone `person_track_id`/`bag_track_id` fields** instead of the frozen ones — the anti-wrong-person design exists and is sound, but has not been exercised successfully by any real production run to date. The historical "thousands of stacked boxes in one snapshot" bug (from `PHASE2_REPORT.md`) is **confirmed fixed** — snapshots are now cut from the clean, unannotated original video with 0-2 boxes drawn, not from the fully-annotated `analyzed.mp4`.

---

## 10. Current Performance

CPU-only deployment (confirmed: `torch==2.4.1+cpu` wheels, no GPU reservation in `docker-compose.yml`, `device="cpu"` hardcoded). Dominant cost, per source frame, unconditionally: **up to 3 full YOLO forward passes** (person model + `best.pt` + `garbage_bag_v2.pt`) plus a conditional 4th HSV pass — measured historically at ≈152ms/frame for just two of the three models (`_l1_bench.txt`), consistent with the historically-measured 4-9 fps end-to-end ceiling (`FINAL_HARDENING_REPORT.md`, `.workbuddy-ai` notes). MoveNet/Novelty/FSM are correctly throttled to `analysis_fps` — this was already fixed once (a documented 3-4x speedup) but the detection stage, which is larger, was never addressed. Separately, **each confirmed event re-opens and re-decodes the original video at least 3 times** (frame-seeks, clip extraction, face capture) and can trigger **up to 3 separate ffmpeg H.264 transcodes** of overlapping footage — a fixed, serial, per-event cost that scales with event count, independent of `analysis_fps`.

---

## 11. Known Problems (as given in the brief) — verification status

| # | Claim | Status |
|---|---|---|
| 1 | Full video shown too prominently | **Partially true, partially fixed** — fixed on VideoAnalysisPage, not fixed on EventDetail (P1-3) |
| 2 | Wrong-person/unrelated-object evidence snapshots | **Confirmed as a real, currently-active risk** — not because the selection *logic* is wrong (it isn't), but because it's never been exercised: 100% of historical events used the non-frozen fallback identity (P0-1 consequence) |
| 3 | Thousands of boxes on one image/video | **Not reproducible in current source's box-drawing loop** (current-frame-only, verified in both `.before`/`.after`/current visualizer); the *historical* root cause (snapshot cut from the fully-annotated video) is confirmed fixed. Real, but architecturally different, per-frame clutter can still occur from noisy color/novelty fallback firing (159 distinct object ids observed in one 30s real clip) |
| 4 | HSV/Novelty risk becoming "waste" visually/logically | **Not found in current source** — `_is_semantic_waste` gate verified solid; one unresolved historical-DB anomaly flagged for targeted re-check |
| 5 | Stable UIDs becoming NULL in old DB records | **Confirmed, current, not just historical** — 30/30 live rows, two fully diagnosed root causes (P0-1) |
| 6 | Production semantic waste model = `garbage_bag_v2.pt` | **Confirmed** |
| 7 | `waste_bag_real_v1.pt` permanently rejected, unused | **Confirmed** — file still on disk but genuinely unreachable from any code path |
| 8 | `PersonIdentityManager`/`ObjectIdentityManager` exist to survive churn | **Confirmed, active, well-designed** |
| 9 | `LitteringEventDetector` is the authoritative engine; `state_machine.py`/`voting.py` superseded | **Half-true** — the legacy modules are indeed dead, but the actual production engine is `AdaptiveEventDetector` (a 3-tier wrapper + cross-video learning store), not bare `LitteringEventDetector` — this distinction matters (P1-8) |
| 10 | `PersonObjectAssociator` may be dead/unused | **Confirmed dead** — instantiated, never invoked |
| 11 | Evidence split between Docker volume and host storage | **Confirmed, current, actively divergent** |
| 12 | CPU performance dominated by repeated detector/rendering/transcode work | **Confirmed** — 3x YOLO/frame + 3x video re-decode/transcode per event |
| 13 | Historical reports may contradict each other | **Confirmed**, specific contradictions documented in Section 13 |
| 14 | Some scenarios only unit-tested, not proven on real video | **Confirmed** — see Section 26 test matrix; only Case F (regrab) has both unit + integration coverage; no case has real-video or dashboard coverage |
| 15 | Previous agents reported "fixed" while production remained wrong | **Confirmed** — e.g. Phase C's frozen-actor fix is real in source but has never once worked correctly in a production run (P0-1) |

---

## 12. Verified Problems — see Section 21 (the full P0/P1/P2 repair catalogue) for the authoritative, file:line-cited list. Section 11 above cross-references each historical claim to its disposition.

---

## 13. Historical Problems — reports vs. current source, specific contradictions found

| Report claim | Current source reality |
|---|---|
| `FACE_CAPTURE_WIRING_REPORT.md`/`FACE_EVIDENCE_REPORT.md` (titles imply a wired real face pipeline) | `evidence_package.py:189-207` hardcodes `backend="region"` — a geometric head-box heuristic, not ML face detection; `retinaface` exists in code but is unused because the container doesn't ship `insightface`/`onnxruntime` |
| `inference/pipeline.py:65` docstring: `auto_tune` is "OFF by default so legacy behaviour/tests are untouched" | Both production entry points (`analysis.py:461`, `run_pipeline.py:153`) hardcode `auto_tune=True` unconditionally — it is **always on** in practice |
| `littering_event_detector.py:165` dataclass default `require_ground_confirmation=False` | `config/events.yaml:47` sets it `True`, and production loads the YAML — the gate **is** enforced in production; most unit tests construct the dataclass directly and silently run with the gate off, diverging from production |
| `FINAL_HARDENING_REPORT.md`: "NOT real-time... 4.0-5.2 FPS" | Still consistent with current source structure — nothing has since changed the unconditional triple-YOLO-per-frame design that produced that number |
| Various `*_REPORT.md` files describe a "Association → FSM → Voting" pipeline (also echoed in some test docstrings) | Stale — the real production chain has no live association step (`PersonObjectAssociator` is dead) and no voting step (`voting.py` is dead); it's Detect → Track → Identity → `AdaptiveEventDetector`(3×FSM) |

---

## 14. Agent vs. Project Scorecard

| Capability | Code | Production path | Unit test | Real video | DB | Dashboard | Confidence |
|---|---|---|---|---|---|---|---|
| Semantic waste vs. proposal gate | ✅ | ✅ | ✅ | Not verified | Mostly ✅ (1 historical anomaly unresolved) | N/A | High |
| Stable person UID resolution | ✅ | ✅ | ✅ | Not verified | N/A | N/A | High |
| Stable object UID resolution | ✅ | ✅ | ✅ | Not verified | N/A | N/A | Medium (same-class close-proximity merge risk, unproven) |
| Frozen actor/object identity at carry-time | ✅ | ✅ (computed) | ✅ | Not verified | **❌ 0/30 rows show it working** | **❌ dashboard never shows it either** | **Low end-to-end** despite high code quality |
| Bystander non-attribution (Cases A/B) | ✅ | ✅ | ✅ (Case B well-covered) | Not verified | Not verified | Not verified | Medium-High |
| Bin-vs-ground distinction (Case G) | Proxy only, no real bin detection | ✅ (proxy) | ✅ (proxy only) | Not verified | Not verified | Not verified | **Low** — proxy tuned to 1 video |
| Regrab / picked-back-up (Case F) | ✅ | ✅ | ✅ | Not proven | Not verified | Not verified | **Highest of all 7 cases** — unit + integration |
| Two-objects-per-person (Cases D/E) | Only in dead `PersonObjectAssociator` | **❌ not supported by production FSM** | Tests exist only for the unused module | Not proven | Not verified | Not verified | Low |
| Evidence anchored to true event (not proximity/first/largest) | ✅ design | ✅ design | Not directly tested | Not verified | ❌ (fallback path used 100% of the time so far) | ❌ | Low end-to-end |
| Rendering shows current-frame-only boxes (no accumulation) | ✅ | ✅ | Not directly tested | Not verified | N/A | Partially (analyzed video still shows all entities, not just actor+object, at the event frame) | Medium |
| Event clip correctly time-bounded | ✅ | ✅ | Not directly tested | Not verified | N/A | ✅ | High |
| Full-video-vs-clip UX hierarchy | ✅ on VideoAnalysisPage | — | N/A | N/A | N/A | ❌ on EventDetail | Mixed |
| Event-creation provenance control | **❌ none exists** | ❌ | ❌ | N/A | ❌ (row 29 proves exploitation) | N/A | **None** |

---

## 15. KEEP

| Component | Why keep | Production dependency | Risk if touched |
|---|---|---|---|
| `littering_event_detector.py` (the FSM core) | Sound, well-tested design for the hard cases (ownership lock, regrab, evidence scoring) | All events | High — this is the most load-bearing file in the repo |
| `adaptive_tuner.py` (`AdaptiveEventDetector`), minus the dedup bug | The real production engine; tier ladder is a legitimate rescue mechanism | Always-on in both entry points | High — fix the bug in place (Section 21 P0-1b), don't replace the module |
| `PersonIdentityManager` / `ObjectIdentityManager` | Correctly designed, isolated, doing their one job well | Every event | Medium |
| `garbage_bag_v2.pt` | Genuinely the working production model | Detection | Medium |
| `evidence_package.py`'s selection logic (`_find_entity`/`_find_object`, exact-key anchoring) | Correctly designed — the bug is upstream (what gets passed in), not here | Every confirmed event | Medium |
| `_is_semantic_waste()` gate | Verified solid; this is the thing preventing proposal→event leakage | Every event | High — do not loosen without re-verifying the historical DB anomaly first |
| `VideoAnalysisPage.tsx`'s Primary/Debug section split | This is the correct UX pattern — copy it to EventDetail, don't redesign it | Dashboard | Low |
| Rendering pipeline's current-frame-only box contract | Verified correct, no accumulation bug found | Dashboard/analyzed video | Low |
| `require_ground_confirmation` gate itself (the mechanism) | Even though it's only a proxy, disabling it would be worse than keeping it while a real bin detector is designed | Event confirmation | Medium |

## 16. REMOVE-LATER

| Component | Why remove | Evidence unused | Dependencies | Risk | Safe removal stage |
|---|---|---|---|---|---|
| `waste_bag_real_v1.pt` + `runs/waste_bag_real_v1/weights/*` (~48.9MB) | Confirmed permanently unreachable | Confirmed — zero load-path references | None | None | Any time |
| `inference/visualization/tracking_visualizer.after.py` | Byte-identical duplicate of the real file | Confirmed unused | None | None | Any time |
| `inference/visualization/tracking_visualizer.before.py` | Superseded pre-Phase-1 snapshot | Confirmed unused | None | None → ARCHIVE instead of delete (historical value) |
| `inference/behavior/state_machine.py`, `inference/behavior/voting.py` | Dead, only their own tests import them | Confirmed | Their own unit tests | Low (breaks 2 test files — delete those too or leave both as an intentionally-frozen regression pair) | After confirming no external tooling imports them |
| `EventEvidenceCollector` (`littering_event_detector.py:2066-2184`) | Zero callers | Confirmed | None | None | Any time |
| Root-level `pipeline.py` + `person_bag_association.py` | Separate, unused duplicate pipeline | Confirmed neither entry point imports it | Unknown — verify no external script depends on it first | Low-Medium | After a repo-wide import grep |
| 59 root-level debug files (`_l1_*`, `_ev*`, `_job*`, `_check_*`, `tmp_*`, `_test*`) | One-off completed-audit artifacts | N/A | None, except `_l1_bench.txt`'s numbers are cited in this report — copy them out first | None | Any time, after extraction |
| `.commandcode/`, `.diag/`, `.nettest/`, `.tooling/` (untracked dot-dirs) | Scratch/tool output, not project logic | N/A | None | None | Any time |
| `.audit/`, `.uitest/`, `.workbuddy-ai/` | Historical investigation records with some citation value | N/A | None | None | ARCHIVE (zip and move out of the working tree, don't delete) |

## 17. REPLACE

| Current | Candidate | Reason | Benefit | Cost | Risk | Priority |
|---|---|---|---|---|---|---|
| 3 separate YOLO models run every frame | One multi-class fine-tuned YOLO model (person+litter+bag classes merged) | This is the actual dominant CPU bottleneck | ~2-3x throughput (152ms→~60-80ms/frame per `_l1_bench.txt` math) | High — dataset consolidation + retrain | Low once trained; FSM must be re-validated against the new detector's behavior | **High** |
| Per-event triple video re-decode (frame-seeks + clip + face capture, each reopening the original file) | Reuse frames already captured in `frame_records`/`CircularFrameBuffer` from the single main pass | Removes redundant disk I/O + redundant ffmpeg transcodes | Meaningful reduction in per-event wall-clock cost, scales with event count | Medium — refactor evidence_package.py to consume in-memory frames | Low | **High** |
| `AdaptiveEventDetector._accept()`/`_is_duplicate()` self-matching | Exclude the just-accepted event from its own duplicate check | Root cause of P0-1b | Fixes NULL UIDs + missing evidence packages for all relaxed-tier events | Low — a scoped code fix | Low | **Highest — do first** |
| `EventDetail.tsx`'s unscoped `confirmed_violations[0]` | Filter by the viewed event's own actor/object/id before picking a `detectorEvent` | Root cause of P0-2 | Fixes cross-event attribution bleed | Low | Low | **Highest — do first** |
| Height-above-feet-line bin proxy | A real bin/receptacle detector or explicit bin-zone annotation, if the deployment has fixed cameras (it does — static-camera confirmed) | No real bin detection exists today | Removes the Case-G false-positive risk | Medium-High (needs labeled bin data or manual zone config) | Medium | Medium |
| `PersonObjectAssociator` (dead, has correct multi-object logic) | Either delete it, or actually wire its multi-object/two-hands logic into the production FSM to support Cases D/E | Currently the "right" code exists but isn't used | Real two-object-per-person support | Medium — needs the FSM's one-bag-per-person assumption relaxed carefully | Medium — touches core FSM invariants | Medium |

## 18. ADD

- **Provenance/auth control on `POST /api/events`** (P0-3) — at minimum an internal shared secret or mTLS between the live-camera pipeline and the backend; ideally remove the public creation endpoint entirely and replace it with an internal-only ingestion path.
- **A DB column + serving route + dashboard tile for `carry.jpg`/`release.jpg`/`ground.jpg`** — these are already computed; they're just invisible.
- **A response_model on `get_analysis_manifest`/`get_analysis_job_events`** so future schema drift is caught by tests, not discovered in production.
- **An Alembic migration setup** — `create_all()` cannot evolve an existing table; the stale `littering_demo.db` schema is proof this has already caused drift once.
- **A held-out accuracy benchmark against MIVIA-IWDD-500** (400 videos, 200 positive + 200 hard negatives, day/night) — the project's only existing evals are self-collected/self-labeled.
- **Evidence integrity check**: a background job that flags/repairs any `Event` row whose stable-identity columns are NULL but should have been populated (surfacing the P0-1 class of bug automatically going forward, even after the code fix).
- **Production event-replay tests**: re-run one of the 6 double-event jobs (88/87/84/78/77/66) through the fixed code and assert both events now display distinct, correct detector state on EventDetail — the cheapest possible regression proof for two of the P0 fixes at once.

## 19. External Research (summary — CPU-only deployment confirmed, governs everything)

| Current | Candidate | Verdict |
|---|---|---|
| 3x per-frame YOLO | One merged multi-class model | **Recommend investigating** — training problem, not a library swap, but highest-leverage perf fix found |
| ByteTrack (hand-rolled namespacing) | Roboflow `supervision` ByteTrack over a merged `Detections` object | **Partially recommend** — code-quality win, not a speed win |
| ByteTrack | BoT-SORT | **Do not recommend** — solves ego-motion problems this static-camera system doesn't have |
| ByteTrack | OC-SORT / DeepSORT+ReID | **Do not recommend on CPU** — adds a full embedding-network pass per box, worsens the actual bottleneck |
| Hand-rolled FSM + Novelty | X3D / VideoMAE / Hiera (video action recognition) | **Do not recommend** — GPU-class or CPU-heavier than the current 2D approach, and loses per-object bbox evidence entirely (a functional regression, not just a perf one) |
| `NoveltyDetector` (background subtraction) | WACV-2026 "Hybrid Temporal-Spatial Novelty Detection for Illegal Waste Dumping" paper | **Recommend reading** as algorithmic-improvement material for the existing module, not as a new dependency |
| Self-collected evals only | MIVIA-IWDD-500 dataset | **Recommend** — cheap, independent accuracy benchmark |

---

## 20. Target Architecture

```
DETECTION  →  TRACKING  →  STABLE IDENTITY  →  OWNERSHIP  →  TEMPORAL EVENT  →  EVIDENCE  →  DATABASE  →  API  →  DASHBOARD
```

Rules, restated against what was actually found broken:

- **Detection never decides littering.** (Verified true today — keep it that way.)
- **Tracking never decides littering.** (Verified true today.)
- **Rendering never decides littering.** (Verified true today.)
- **The temporal engine's output is a single frozen record per confirmed event** — actor identity, object identity, timestamp — and **that exact record, unmodified, must be what reaches the database, the evidence writer, and the dashboard.** This is the one rule currently violated three separate times (live-camera schema drops it, adaptive-tier dedup discards it, EventDetail ignores it in favor of index `[0]`). Fixing all three closes the actual gap between "the design is right" and "the product works."
- **Evidence consumes the frozen record only** — already true in `evidence_package.py`'s design; just needs to actually receive a non-None record every time.
- **The dashboard consumes only what's in the database for a specific event ID** — never "the first thing in the job's report."

---

## 21. Detailed Repairs

### P0 — catastrophic / product-blocking

---

**P0-1 — Stable actor/object identity is NULL for 100% of production events (two independent root causes)**

- **Severity:** P0 · **Area:** Persistence / Event Engine
- **Symptom:** `events.event_actor_person_uid`, `event_object_uid`, `event_actor_person_track_id`, `event_object_track_id` are NULL for all 30 live rows, including the newest.
- **Root cause A (live-camera path, structural, 100% reproducible):** `schemas.EventCreate`/`EventBase` (`backend/schemas.py:34-47`) has no stable-identity fields at all; `inference/pipeline.py:391-463` (`_maybe_post_backend`) never sends them because its own `PipelineEvent` dataclass (`pipeline.py:75-87`) doesn't carry them either.
- **Root cause B (video-upload path, a real bug in `adaptive_tuner.py`):** `AdaptiveEventDetector._accept()` (`adaptive_tuner.py:522-533`) appends the just-accepted event's own `(person_track_id, ts)` to `self._emitted_confirmed` **before** the `confirmed_events` property is next read. `_dedup_confirmed()`/`_is_duplicate()` (`adaptive_tuner.py:373-375, 494-510, 535-544`) then finds distance=0 against that same entry and treats the event as a duplicate **of itself**, filtering it out of `confirmed_events` on the very next access — which happens in the same tick. This was reproduced live, in-memory, non-destructively during this audit. Any event confirmed by a *relaxed* tier (not the strictest tier-0) is affected — which is common, since tier-0 is intentionally the strictest.
- **Affected files/functions:** `adaptive_tuner.py:_accept, _is_duplicate, _dedup_confirmed, confirmed_events`; `backend/routers/analysis.py:541,636,684-697,723-750`; `backend/schemas.py:EventCreate/EventBase`; `inference/pipeline.py:PipelineEvent,_maybe_post_backend`.
- **Affected layers:** Layer 2 (event engine) → persistence → Layer 3 (evidence, silently skipped as a side effect — see P0-1's second symptom below).
- **Second symptom (same root cause B):** because `det_ev`/`detector_event` is `None` for any relaxed-tier event, `write_event_evidence_package()` is **never called** for it (`analysis.py:723-750`) — the full crop/clip/face evidence package silently doesn't exist for these events; only the raw whole-frame snapshot survives.
- **Current behavior:** relaxed-tier-confirmed events get a DB row and a snapshot, but no stable identity and no rich evidence.
- **Desired behavior:** every confirmed event, regardless of which tier confirmed it, gets its full frozen identity persisted and its full evidence package generated.
- **Exact repair strategy:**
  1. Fix `_is_duplicate`/`_dedup_confirmed` to exclude an event's own already-recorded acceptance from its own duplicate check (e.g., skip entries with the identical `event_id`, not just distance-based matching).
  2. Add `event_actor_person_uid`/`event_object_uid`/`event_actor_person_track_id`/`event_object_track_id` fields to `PipelineEvent` and to `schemas.EventCreate`, and populate them in `_maybe_post_backend`.
  3. Re-verify `_compact_detector_event()` (`analysis.py:58-83`) always emits all four keys (one historical anomaly found where `event_actor_person_uid` was entirely absent from a report where its siblings were present — root cause not fully resolved, needs a live-trace on a fresh run after fix #1).
- **Dependencies:** none blocking; this should be fixed before anything evidence- or dashboard-related, since those depend on this record being real.
- **Regression risks:** changing dedup logic could affect which tier "wins" in edge cases — needs the existing `AdaptiveEventDetector` unit tests re-run plus a new test for the self-match case specifically.
- **Test required:** new unit test asserting a tier-1/2-confirmed event survives its own next `confirmed_events` read; new unit test for `EventCreate` schema completeness.
- **Real-video validation required:** yes — re-run one of the 6 double-event jobs and confirm both events now populate all four fields.
- **Dashboard validation required:** yes — confirm EventDetail/VideoAnalysisPage now display the stable UID for a freshly-generated event.
- **Rollback plan:** revert the two-line dedup-exclusion change; no schema/data migration involved for the code fix itself (the schema fields already exist).

---

**P0-2 — `EventDetail.tsx` shows the wrong event's detector state when a job has multiple confirmed events**

- **Severity:** P0 · **Area:** Dashboard
- **Symptom:** Opening `/violations/27` and `/violations/28` (both belong to job 88) shows the identical "Behavior" checklist and evidence-score breakdown, because `EventDetail.tsx:34` does `report?.event_detector?.confirmed_violations?.[0] ?? null` — unconditionally index `[0]`, never filtered by the currently-viewed event.
- **Root cause:** `report` is the *whole job's* report (`backend/routers/events.py:88-94`, `get_event_review`), shared by every `Event` row with the same `analysis_job_id`; the frontend never re-scopes it to the specific event being viewed.
- **Affected files/functions:** `dashboard/src/pages/EventDetail.tsx:34`.
- **Affected layers:** Layer 3 (evidence display) / Dashboard.
- **Why it happens:** the report review endpoint returns one shared blob per job; nothing in the frontend narrows it back down per-event.
- **Current behavior:** confirmed, live-exploitable against 6 real job/event pairs today.
- **Desired behavior:** `EventDetail.tsx` selects the `confirmed_violations` entry whose actor/object/track-id (or, once P0-1 is fixed, whose stable UID) matches the specific `Event` row being viewed — never a fixed index.
- **Exact repair strategy:** match by `event.event_actor_person_track_id`/`event_object_track_id` (or the DB event's own `person_track_id`/`object_track_id` as an interim key pre-P0-1) instead of `[0]`.
- **Dependencies:** best fixed after P0-1, since a correct stable UID makes the match unambiguous; can be interim-fixed with legacy track-id matching first.
- **Regression risks:** low — purely a frontend selection-logic change.
- **Test required:** a frontend unit/integration test simulating a report with 2+ confirmed_violations and asserting the correct one is picked per event id.
- **Real-video validation required:** yes — the same double-event jobs used for P0-1.
- **Dashboard validation required:** yes, by definition.
- **Rollback plan:** trivial — revert the selection function.

---

**P0-3 — `POST /api/events` has no provenance/authentication check; anyone can fabricate a "confirmed" violation**

- **Severity:** P0 · **Area:** Backend API / Data Integrity
- **Symptom:** `backend/routers/events.py:103-133` (`create_event`) inserts whatever `EventCreate` payload it receives with `status="confirmed"` accepted at face value. Proven exploited: `tmp_create_event.py` (a leftover throwaway script) matches DB row `events.id=29` exactly.
- **Root cause:** the endpoint was built to serve the live-camera pipeline's push model but was never restricted to that caller.
- **Affected files/functions:** `backend/routers/events.py:create_event`.
- **Affected layers:** persistence / evidentiary integrity — this is the boundary the whole system's "it must be the true event" promise ultimately rests on.
- **Current behavior:** any HTTP client on the network can create an indistinguishable-from-real confirmed event.
- **Desired behavior:** only the trusted live-camera pipeline process (or the internal video-upload job) can create a `confirmed` event; external callers get rejected or restricted to a draft/unconfirmed status.
- **Exact repair strategy:** add a shared-secret header or mTLS check scoped to the internal pipeline caller; alternatively remove the public route and replace with an internal-only ingestion function called in-process.
- **Dependencies:** none.
- **Regression risks:** must not break the legitimate live-camera push path — coordinate the secret/cert with `inference/pipeline.py:_maybe_post_backend`.
- **Test required:** a new test asserting an unauthenticated `POST /api/events` is rejected.
- **Real-video validation required:** no (this is an integrity fix, not a detection fix).
- **Dashboard validation required:** no.
- **Rollback plan:** trivial — revert the auth check if it blocks a legitimate integration.

---

### P1 — important

---

**P1-1 — Evidence storage genuinely split between a Docker named volume and the host filesystem**

- **Severity:** P1 · **Area:** Infrastructure / Evidence
- **Symptom:** `D:\HO\evidence_store` (host) and `/app/evidence_store` (container, what the dashboard actually serves) have confirmed-divergent contents (43 vs 33 uploaded files; host side full of orphaned hex-hash directories that match no current event).
- **Root cause:** `docker-compose.yml:40-44` mounts named volumes at subpaths of the bind-mounted repo root, shadowing it.
- **Affected files:** `docker-compose.yml`.
- **Desired behavior:** one authoritative, host-inspectable evidence location, or explicit documentation + tooling (`docker exec`/API-only inspection) so nobody manually browses the shadowed host folder expecting it to be current.
- **Repair strategy:** either bind-mount `evidence_store`/`uploaded_videos` directly to host paths (simplest, matches the stated design intent) or provide a `make sync-evidence` helper that copies volume contents to host for audit purposes, and delete/clearly-mark the stale host folders as legacy.
- **Regression risk:** low — a volume/mount change, verify container restarts cleanly and existing paths in DB rows still resolve.
- **Test required:** manual verification that a freshly-created event's files appear at the expected host path.
- **Rollback:** revert compose file.

---

**P1-2 — `carry.jpg`/`release.jpg`/`ground.jpg` are orphaned artifacts (no DB column, no route, never rendered) and frequently fail to write at all**

- **Severity:** P1 · **Area:** Evidence
- **Root cause:** `evidence_package.py:322-324` silently returns `None` if both actor and object bboxes aren't present at the exact recorded frame; no DB column (`Evidence` model) or route exists to serve them even when written.
- **Repair strategy:** add `Evidence.carry_image_path`/`release_image_path`/`ground_image_path` columns (needs a migration — see P1-9/Section 18's Alembic ask), a serving route, and a dashboard tile; also log (not silently swallow) the None-return case so operators can see how often the sequence images fail.
- **Regression risk:** low, additive-only.
- **Test required:** new test asserting these paths persist and are servable when both bboxes are present.

---

**P1-3 — `EventDetail.tsx` doesn't demote the full analyzed video the way `VideoAnalysisPage.tsx` does**

- **Severity:** P1 · **Area:** Dashboard UX
- **Root cause:** the "Primary Event Evidence vs. Technical Review/Debug" pattern was implemented once, on `VideoAnalysisPage.tsx`, and never ported to `EventDetail.tsx`, which shows the original+full-analyzed video side-by-side as the first, most prominent panel.
- **Repair strategy:** restructure `EventDetail.tsx` to lead with actor/object/clip (same visual weight as `VideoAnalysisPage.tsx:463-526`), demote the full-video comparison to a labelled secondary/debug section.
- **Regression risk:** low, UI-only.
- **Dashboard validation required:** yes.

---

**P1-4 — One-bag-per-person is hard-enforced in the production FSM; Cases D/E (two simultaneous objects per actor) are not supported**

- **Severity:** P1 · **Area:** Event Engine
- **Root cause:** `_select_primary_associations()` (`littering_event_detector.py:1440-1471`) enforces single-object-per-person by design; the correct multi-object/two-hands logic exists only in the dead `PersonObjectAssociator`.
- **Repair strategy:** either accept this as a documented product limitation (cheapest — update docs/tests to state it explicitly instead of leaving it ambiguous), or port the wrist-slot bookkeeping from `PersonObjectAssociator` into the primary-association selection, carefully, since it touches a core FSM invariant.
- **Regression risk:** medium-high if implemented; near-zero if only documented.
- **Test required:** either a new two-object test proving correct behavior, or an explicit "known limitation" test asserting current (single-object) behavior so a future change doesn't silently alter it unnoticed.

---

**P1-5 — No real bin detection; ground-level bins will false-positive as litter**

- **Severity:** P1 · **Area:** Event Engine
- **Root cause:** `require_ground_confirmation`'s only signal is `ground_plane_margin_ratio=0.40` (a feet-line-height proxy), tuned to one validation video (`IMG_5305`); there is no bin class, bounding box, or zone anywhere in the codebase.
- **Repair strategy:** since cameras are static (confirmed), the cheapest real fix is operator-configured bin zones per camera (a polygon or box a human draws once during setup) rather than a learned bin detector — far lower cost/risk than training a new model, and directly matches the "static camera" deployment reality.
- **Regression risk:** medium — must not regress the existing proxy's true-ground-litter recall.
- **Test required:** new test with a bag placed inside a configured bin zone at ground height, asserting no violation.

---

**P1-6 — `Settings.tsx` is 100% hardcoded, can silently drift from real deployed config**

- **Severity:** P1 · **Area:** Dashboard
- **Repair strategy:** add a read-only `GET /api/config` endpoint that returns the actually-loaded `EventDetectorConfig`/`config/events.yaml` values, and point Settings at it (read-only display is enough to fix the drift risk; a write-back UI is a separate, larger feature).
- **Regression risk:** low, additive.

---

**P1-7 — Dataclass defaults contradict deployed config (`require_ground_confirmation`, `auto_tune`); most unit tests silently diverge from production**

- **Severity:** P1 · **Area:** Config / Test integrity
- **Repair strategy:** change the dataclass defaults to match what's actually deployed (`require_ground_confirmation=True`, document `auto_tune` as always-on rather than "off by default"), and audit/update the unit test suite that constructs `EventDetectorConfig` directly to match production defaults unless a test is explicitly testing the gate-off case.
- **Regression risk:** low code risk, but will likely surface currently-hidden test failures — treat that as a feature, not a regression, of this fix.

---

**P1-8 — Cross-video persistent learning store makes verdicts non-reproducible**

- **Severity:** P1 · **Area:** Event Engine / Forensic integrity
- **Root cause:** `adaptive_tuner.LearningStore` (`adaptive_tuner.py:164-276`) persists `learning/learning.json` and permanently relaxes thresholds system-wide based on the history of all previously analyzed videos — the same clip can, in principle, get a different verdict depending on unrelated footage processed in between.
- **Repair strategy:** at minimum, surface the currently-active learned thresholds in every event's stored metadata (so a later audit can tell whether a given verdict depended on drift), and consider scoping learning per-camera rather than globally; hard clamps already bound the blast radius, which is good and should be kept.
- **Regression risk:** medium if per-camera scoping is added (changes shared-state assumptions).
- **Test required:** a test asserting the same input video produces the same verdict regardless of learning-store history (or, if that's accepted as a designed tradeoff, a test asserting the *stored metadata* always discloses which learned thresholds were in effect).

---

**P1-9 — Dominant CPU bottleneck (3x YOLO/frame) and redundant per-event video re-decode/transcode**

- **Severity:** P1 · **Area:** Performance
- **Repair strategy:** see Section 17 (REPLACE table) — merge the 3 YOLO models into one multi-class model (highest leverage, requires retraining); reuse already-decoded frames for evidence extraction instead of re-opening the original video 3+ times per event.
- **Regression risk:** high for the model merge (changes detection behavior, needs full FSM re-validation); low-medium for the frame-reuse refactor.

---

**P1-10 — `PersonObjectAssociator` is dead code self-described as "core contribution," misleading to maintainers**

- **Severity:** P1 · **Area:** Code hygiene / correctness risk
- **Repair strategy:** either delete the `PersonObjectAssociator` class (keep its `Track`/`Keypoints`/`AssociationConfig` dataclasses, which are still used elsewhere) or actually wire it in per P1-4's second option. Do not leave it half-alive and misleadingly documented.

---

**P1-11 — Confirmed-dead legacy engines create audit confusion**

- **Severity:** P1 · **Area:** Code hygiene
- **Repair strategy:** archive `state_machine.py`, `voting.py`, root `pipeline.py`/`person_bag_association.py`, `EventEvidenceCollector` per Section 16.

---

**P1-12 — Dashboard renders the non-authoritative `person_track_id` instead of the frozen stable UID, even where present**

- **Severity:** P1 · **Area:** Dashboard
- **Root cause:** `EventDetail.tsx:194` reads `event.person_track_id` even though `types.ts` declares `event_actor_person_track_id`/`event_actor_person_uid` exist.
- **Repair strategy:** switch the display field once P0-1 guarantees the stable field is actually populated; until then, this is latent (only surfaces on the rare mid-event ID-churn case).

---

### P2 — improvement

| ID | Item | Repair |
|---|---|---|
| P2-1 | Orphaned config keys (`release_window_frames`, `max_fallback_tracker_gap_frames`) never consulted | Remove or wire in; currently confusing dead config |
| P2-2 | Entire color/CSRT fallback confidence-discount mechanism structurally unreachable | Remove the dead branch or re-enable a real fallback-confirmation path deliberately |
| P2-3 | Magic constants tuned to 1-2 validation videos (`stationary_max_pixel_step=12.0`, `ground_plane_margin_ratio=0.40`) | Re-derive from a larger validation set; document provenance in-code |
| P2-4 | Auto-calibration is frame-count-dependent (first 24 ticks define rest-of-video thresholds) | Consider calibrating from a rolling window instead of a one-time first-24-tick sample |
| P2-5 | `bag_move_norm=0.05` anti-furniture gate too permissive (tracking jitter can satisfy "handled") | Raise the threshold or require sustained motion, not a single-frame displacement |
| P2-6 | Duplicated abandonment logic (two independent implementations, `:1039-1073` vs `_advance_pair`) | Consolidate into one function |
| P2-7 | Root-level repo clutter (59 debug files, untracked dot-dirs, stale reports) | Archive per Section 16 |
| P2-8 | `garbage_bag_v2.pt` + `best.pt` both loaded with equal dedup priority, no explicit tie-break | Add explicit model-priority ranking if the two ever disagree in practice |
| P2-9 | `waste_bag_real_v1.pt` + its `runs/` artifacts still on disk (~48.9MB) | Delete or archive — confirmed zero functional risk |
| P2-10 | `tracking_visualizer.after.py`/`.before.py` stray files | Remove/archive per Section 16 |
| P2-11 | `VideoAnalysisPage.tsx` only shows the first persisted event via `slice(0,1)` when a video has multiple confirmed events | Show all confirmed events for the just-analyzed video, not just the first |
| P2-12 | `EvidencePage.tsx` caps thumbnail lookups to the first 30 events | Paginate instead of capping |
| P2-13 | `get_analysis_manifest`/`get_analysis_job_events` lack a `response_model` | Add one for schema-drift protection |
| P2-14 | `/api/evidence/event/{id}/snapshot` picks oldest not newest evidence row | Change `order_by(id.asc())` to `.desc()`, or document why oldest is intentional |

---

## 22. Detailed P0 Repairs — see Section 21 (P0-1 through P0-3 above; kept in one combined section for cross-reference clarity)

## 23. Detailed P1/P2 Repairs — see Section 21 above

---

## 24. Acceptance Tests

| # | Test | Unit | Integration | Real video | Dashboard |
|---|---|---|---|---|---|
| 1 | Person walks by → no event | Existing (partial, 2-actor variant) | Not proven | Not proven | Not proven |
| 2 | Person carries waste → no event yet | Existing | Not proven | Not proven | Not proven |
| 3 | Person drops waste on ground → event | Existing | Not proven | Not proven | Not proven |
| 4 | Person drops and walks away → confirmed | Existing | Not proven | Not proven | Not proven |
| 5 | Person drops then re-grabs → no final violation | **Existing (unit + integration)** | **Existing** | Not proven | Not proven |
| 6 | Person puts waste in bin → no ground-litter violation | Existing (proxy only — no real bin exists) | Not proven | Not proven | Not proven |
| 7 | P1 walks by, P2 litters, P3 walks by → only P2 is actor | Partial (2-actor only, not literal 3-person) | Not proven | Not proven | Not proven |
| 8 | One person + two objects → only released object becomes event object | **Not supported by production FSM** — needs P1-4 first | Not proven | Not proven | Not proven |
| 9 | Multiple people simultaneously → correct ownership | Existing (Case B — well covered) | Not proven | Not proven | Not proven |
| 10 | Evidence → correct person + object + time | Design correct, **not exercised successfully in any real run** (P0-1) | Not proven | Not proven | Not proven — needs P0-1/P0-2 fixed first |
| 11 | Dashboard → event clip shown as primary evidence | Yes on VideoAnalysisPage | — | — | **No on EventDetail** — needs P1-3 |
| 12 | Full video → optional technical review, not primary | Yes on VideoAnalysisPage | — | — | **No on EventDetail** — needs P1-3 |
| 13 | Normal tracking → one box per current active entity | **Verified correct in current source** | — | Not proven | Not proven |
| 14 | No accumulated historical boxes | **Verified correct in current source** | — | Not proven | Not proven |
| 15 | No proposal object becomes event truth | **Verified correct in current source** (one historical DB anomaly flagged for re-check) | — | Not proven | Not proven |

**New tests this plan requires that don't exist today:** the literal 3-person Case A; combined Case C (simultaneous ground+bin by two different actors); the P0-1 self-dedup regression test; the P0-2 event-scoping regression test; the P0-3 unauthenticated-POST rejection test; a re-run of a real double-event job end-to-end after P0-1/P0-2 are fixed.

---

## 25. Deployment / Regression Strategy

1. Fix P0-1 and P0-2 together first — they're independent code changes but should be validated together against the same real double-event job, since that one test proves both fixes at once (Section 18's "production event-replay test").
2. Fix P0-3 next — pure addition of an auth check, isolated from the FSM/evidence logic, lowest risk of the three P0s.
3. Do not touch `littering_event_detector.py`'s FSM transition logic (P1-4, P1-5, P2-3 through P2-6) until P0s are proven fixed and the double-event replay test is green — these later changes are riskier and easier to reason about once the persistence layer is trustworthy.
4. Performance work (P1-9, the YOLO-model-merge) is the largest, highest-risk change in this entire plan (requires retraining) — schedule it last, after every correctness fix, so a training-induced regression isn't confused with a persistence-layer regression.
5. Dead-code archival (P1-10, P1-11, P2-7, P2-9, P2-10) can happen at any time in parallel — it has no functional dependency on anything else, but do it in its own commit(s) so it never gets blamed for an unrelated regression.

---

## 26. Final Health Scorecard

| Area | Rating |
|---|---|
| Detection | VERIFIED (working, correctly gated) |
| Tracking | VERIFIED |
| Identity (person/object UID) | VERIFIED, design sound |
| Association/ownership | VERIFIED for single-object cases; **BROKEN** for two-simultaneous-objects (Cases D/E) |
| Temporal reasoning (core FSM) | VERIFIED, well-tested for the cases it covers |
| Re-grab | VERIFIED (best-tested case in the system) |
| Bin/ground | **PARTIAL** — proxy only, no real bin detection |
| Evidence | **PARTIAL** — design correct, **never successfully exercised end-to-end in production** due to P0-1 |
| Dashboard | **PARTIAL** — one page (VideoAnalysisPage) done right, the reviewer-facing page (EventDetail) has a live attribution bug |
| API | **PARTIAL** — functionally complete but has an unauthenticated write endpoint (P0-3) |
| Database | **VERIFIED schema**, but **BROKEN data** (100% NULL stable-identity columns in production) |
| Performance | **PARTIAL** — one real bottleneck fixed historically (Novelty), the dominant one (3x YOLO/frame) untouched |
| Deployment | VERIFIED CPU-only, Docker-based, working; evidence storage topology is a real operational trap |
| Testing | **PARTIAL** — strong unit coverage for single-actor cases, essentially no real-video or dashboard-level coverage for any case |
| Documentation | **LEGACY/MISLEADING in places** — several historical reports describe fixed/wired capabilities (face detection, association→FSM→voting pipeline, auto_tune "opt-in") that don't match current source |

---

## 27. Final Verdict

1. **Is the project currently production-ready? No.** The core CV/FSM engineering is sound, but the persistence/evidence/dashboard seam has three P0-severity defects that mean the product's central promise — "we show you the correct person and the correct object" — has not actually held for a single one of the 30 real events produced so far.
2. **Single biggest blocker:** the `AdaptiveEventDetector` self-dedup bug (P0-1b) — it silently discards both the stable identity and the entire rich evidence package for any event confirmed by other than the strictest tier, which appears to be common.
3. **Biggest AI problem:** no real bin detection exists (P1-5) — the system cannot currently distinguish "littered" from "disposed of properly" in the one case (ground-level bin) that matters most for false-positive rate.
4. **Biggest tracking problem:** none critical — tracking/identity is the strongest layer of the system. The only open question is unproven same-class close-proximity object-merge risk (Section 5, low-severity, unproven either way).
5. **Biggest evidence problem:** the frozen, anti-wrong-person actor/object identity that the FSM correctly computes has never once reached the database intact (P0-1); evidence selection logic is sound but has been fed the wrong input 100% of the time.
6. **Biggest Dashboard problem:** `EventDetail.tsx`'s unscoped `confirmed_violations[0]` (P0-2) — a live, currently-exploitable cross-event attribution bug on the exact page a human reviewer uses to make a decision.
7. **Biggest performance problem:** three full YOLO forward passes on every source frame, unconditionally — never addressed despite being the largest single cost in the pipeline.
8. **What should be removed:** `waste_bag_real_v1.pt` and its training artifacts, the dead legacy FSM/voting modules, the dead root-level duplicate pipeline, `EventEvidenceCollector`, and ~60 root-level debug files (Section 16).
9. **What should be replaced:** the 3-model YOLO detection stack (merge to one model); the per-event triple video re-decode; the bin-vs-ground proxy (replace with configured bin zones, given static cameras).
10. **What should be added:** an auth/provenance check on event creation; DB columns + UI for the carry/release/ground sequence images; an Alembic migration setup; a read-only config endpoint for Settings; an independent accuracy benchmark (MIVIA-IWDD-500).
11. **What should never be touched without regression tests:** `littering_event_detector.py`'s FSM transition logic (`_advance_pair`, `_release_detected`, `_select_primary_associations`) and `PersonIdentityManager`/`ObjectIdentityManager` — these are the highest-quality, most load-bearing parts of the system and the parts most likely to silently break subtle multi-person guarantees if touched carelessly.
12. **First 5 repair stages (in order):**
    1. Fix P0-1 (adaptive-tier self-dedup bug + live-camera schema gap) and P0-2 (EventDetail event-scoping) together, validated by re-running one real double-event job end-to-end.
    2. Fix P0-3 (unauthenticated event creation).
    3. Fix P1-3 (EventDetail full-video-vs-clip hierarchy) and P1-12 (dashboard reads the now-reliable stable UID field) — cheap, high-visibility correctness wins that ride on stage 1's fix.
    4. Fix P1-1 (evidence storage volume/host split) and P1-2 (expose carry/release/ground evidence) — operational-trust fixes.
    5. Only then: P1-5 (bin zones) and P1-9 (the YOLO model-merge performance work) — the two highest-cost, highest-risk changes, deliberately scheduled last so they're never confused with the correctness fixes above.

---

*This document was produced by six parallel read-only forensic agents (architecture/algorithm inventory; detector semantics/tracking-identity; event recognition/multi-person attribution; evidence forensics/rendering; dashboard/backend/database; performance/dead-code/external-research) and merged by a coordinating session. No source files, configuration, models, or database rows were modified in the production of this report.*
