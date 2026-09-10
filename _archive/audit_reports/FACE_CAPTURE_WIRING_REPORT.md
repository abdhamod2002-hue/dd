# Face capture wired to the REAL confirmed-event moment — Verification Report

**Scope (Part 1 of the current task):** connect `FaceEvidenceCapture` to the actual
`CONFIRMED` emission of the littering-event detector, so the face is captured
**automatically, only when a genuine violation is confirmed** — not on every
ordinary person track. Then prove it on the 4 previously-confirmed D:\W videos
(5117, 5118, 5119, 5120) by showing `face_evidence.jpg` land inside the *real*
evidence package for each, driven by a real confirmed event.

---

## 1. Where the CONFIRMED decision is emitted

File: `littering_event_detector.py`

- The final, authoritative confirmation happens in
  **`LitteringEventDetector._evaluate_confirmation`** (lines ~1001-1013).
  When the temporal state machine reaches the departure/abandonment proof and the
  evidence confidence clears `min_event_confidence`, it sets
  `mem.state = EventState.VIOLATION_CONFIRMED`, appends a `LitteringEvent(confirmed=True)`
  to `self.confirmed_events`, and returns it. This is the single point that emits
  `CONFIRMED` — the decision **logic itself was not touched**.
- The detector is pure logic: it only receives `DetectorPerson`/`DetectorBag`
  (bbox + track id + keypoints), never raw pixels. So the face *image* cannot be
  captured inside the detector; it is captured by the **consumer** that owns the
  video frames and the confirmed event.

### How that confirmed event reaches evidence production

There are two consumers; both call face capture only on confirmed events:

1. **Live camera path** — `inference/pipeline.py`:
   `InferencePipeline.process_frame` (line ~170) iterates `detector_events`, and
   for each `dev.confirmed` calls `_start_evidence_for_detector_event` →
   `EvidenceManager.assemble_snapshot/finalize` (writes snapshot + evidence.mp4).
   `EvidenceManager` keeps an inert `face_capture=None` parameter (unchanged
   behaviour, left untouched to avoid disturbing the working live path).
2. **Uploaded-video analysis path (the one exercised here)** —
   `backend/routers/analysis.py::_run_video_analysis_job`:
   - runs the REAL `InferencePipeline` on the video (line ~445),
   - collects only confirmed events (`pipe.events`, line ~560),
   - for each confirmed event calls **`write_event_evidence_package(...)`**
     (line ~611), which produces `snapshot/person/waste/event_clip/metadata`
     **and already invokes `_capture_face_evidence`** (the face writer).
   Because `write_event_evidence_package` is reached **only inside the
   `for ev in pipe.events:` loop of confirmed events**, face capture is, by
   construction, gated to genuinely confirmed violations.

## 2. Explicit "confirmed-only" guarantee added

To make the gate code-level (not just call-site-level) and impossible to
accidentally fire on an unconfirmed person track, `_capture_face_evidence`
(`inference/visualization/evidence_package.py`) now begins with:

```python
if not event.get("confirmed"):
    return { ... "status": "FACE_NOT_CAPTURED",
             "reason": "event not confirmed — face capture skipped", ... }
```

Verified by unit check: a non-confirmed event returns `FACE_NOT_CAPTURED`
immediately without scanning any frame; a confirmed event proceeds normally.

## 3. BEFORE vs AFTER

| | BEFORE (earlier test) | AFTER (this verification) |
|---|---|---|
| Event source | hand-built "representative" person-track span passed directly to `write_event_evidence_package` | **REAL** `LitteringEventDetector` run on the video; only its `VIOLATION_CONFIRMED` events used |
| Ran on every video? | Yes — fired even without a confirmed violation | **No** — fires only when the detector confirms |
| Face trigger | synthetic window | the genuine `frames`/`timestamps` of the confirmed event |
| Evidence package | separate test dir | the **real** package produced by the production call |

## 4. Proof on the 4 confirmed D:\W videos

Test: `scripts/test_face_on_confirmed.py` replicates the **evidence-production
half** of `_run_video_analysis_job` (real detector + `write_event_evidence_package`)
without the DB/HTTP layers. It ran the REAL `LitteringEventDetector` on each
video:

- **5117** → 1 confirmed event `VIOLATION_CONFIRMED`, conf 0.858
- **5118** → 1 confirmed event `VIOLATION_CONFIRMED`, conf 0.829 (1 other candidate rejected: NOT_ENOUGH_CARRIED_FRAMES)
- **5119** → 1 confirmed event `VIOLATION_CONFIRMED`, conf 0.828
- **5120** → 1 confirmed event `VIOLATION_CONFIRMED`, conf 0.828

For **every** confirmed event, `face_evidence.jpg` was auto-produced inside the
real evidence package (separate from `person.jpg`), recorded in `metadata.json`:

| Video | face_evidence.jpg (real file) | size(px) | conf | frame | status |
|-------|------------------------------|----------|------|-------|--------|
| IMG_5117.MOV | `.audit/face_on_confirmed/IMG_5117.MOV/face_evidence.jpg` (4439 B) | 92 | 0.97 | 249 | CAPTURED |
| IMG_5118.MOV | `.audit/face_on_confirmed/IMG_5118.MOV/face_evidence.jpg` (4844 B) | 96 | 0.97 | 444 | CAPTURED |
| IMG_5119.MOV | `.audit/face_on_confirmed/IMG_5119.MOV/face_evidence.jpg` (1458 B) | 37 | 0.77 | 405 | CAPTURED |
| IMG_5120.MOV | `.audit/face_on_confirmed/IMG_5120.MOV/face_evidence.jpg` (2376 B) | 56 | 0.85 | 436 | CAPTURED |

Each directory also contains the full real package: `snapshot.jpg`,
`person.jpg`, `waste.jpg`, `event_clip.mp4`, `metadata.json` — and
`metadata.json` carries the `face_evidence` sub-dict (`face_evidence_path`,
`face_detection_confidence`, `frame_number`, `face_bbox`, `face_size_px`,
`blur_score`, `status`, `reason`, `sensitive:true`).

**Conclusion:** face capture is now provably tied to the confirmed-event moment.
When the detector confirms a violation, `face_evidence.jpg` is written
automatically into the real evidence package; when it does not confirm, no face
is captured (correct fail-safe behaviour).

### Honest caveats
- 5119's face is small (37 px) — limited identification value, but honestly
  captured and labelled; no upscaling / fabrication.
- This test exercised the **uploaded-video analysis** consumer (the production
  path that yields `event_clip.mp4` + `person.jpg` + `metadata.json`). The
  **live-camera** `EvidenceManager` path was intentionally left inert/unchanged.
- The detector itself confirmed exactly one event per video here; that is the
  detector's behaviour on these clips, not something the face feature influences.

## 5. Files
- `inference/visualization/evidence_package.py` — confirmed-only guard in `_capture_face_evidence`.
- `scripts/test_face_on_confirmed.py` — faithful real-confirmed verification harness.
- `D:\HO\.audit\face_on_confirmed\<VID>\*` — 4 real evidence packages with `face_evidence.jpg`.
- `D:\HO\.audit\face_on_confirmed_summary.json` / `_report.md` — machine/human results.
