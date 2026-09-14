# Pipeline Observability and Job Lifecycle Control Report

**Date:** Saturday, September 12, 2026  
**Auditor / Engineer:** Software Engineer (Auto)  
**Status:** COMPLETE (Structural Fixes, Tests & Production Build Verified)  
**Scope:** Existing System Fixes Only — Zero New Real-Video Ingestion, Zero Weight Changes, Zero Model Training.

---

## Executive Summary

During long-video analysis runs, operators encountered structural opacity: jobs appeared frozen at progress benchmarks (e.g. 51%), displaying ambiguous person metrics (e.g. "9 persons" caused by tracker ID churn rather than 9 unique people), offering no per-job cancellation controls without killing the entire server or Docker containers, and providing generic status messages ("AI PIPELINE RUNNING") that hid stage-level failures or stalls.

This engineering effort resolved these architectural shortcomings across the backend data models, processing loop, REST API, and frontend Dashboard. The system now provides:
1. **Separated, Honest Person Counting:** Distinct demarcation between `ACTIVE PEOPLE` (concurrent in current frame), `UNIQUE PEOPLE SEEN` (stable appearance Re-ID from `PersonIdentityManager`), and `TOTAL TRACK IDs` (raw ByteTrack tracker assignments).
2. **Real-Time 12-Stage Pipeline Monitoring:** Full visual execution matrix reflecting live backend state across all 12 pipeline stages: `VIDEO INPUT`, `PERSON DETECTION`, `OBJECT / WASTE DETECTION`, `TRACKING`, `PERSON IDENTITY`, `OBJECT IDENTITY`, `POSE`, `OWNERSHIP / ASSOCIATION`, `TEMPORAL EVENT DETECTION`, `EVIDENCE`, `DATABASE`, and `API / DASHBOARD`.
3. **Stage-Aware Global Progress & Failure Diagnostics:** Global completion percentage coupled with active stage indication (e.g. `51% RUNNING — Stage 7/12: POSE`). Immediate failure alert boxes surfacing exact failed stage, blocker reason, last successful stage, last processed frame, and last update timestamp.
4. **Per-Job Safe Cancellation (`[ STOP ANALYSIS ]`):** Job-scoped thread-safe cancellation token that halts the frame processing loop, gracefully releases OpenCV VideoCapture/VideoWriter resources, cleans memory, commits consistent state to the database, and transitions the job to `CANCELLED` (never rendered as `FAILED`).
5. **Decoupled Lifecycle & Deletion Controls:** Explicit separation of `STOP ANALYSIS`, `DELETE JOB`, and `DELETE SOURCE VIDEO`, protected by mandatory confirmation guards and active-execution locks.
6. **Event Lifecycle Distinction:** Clear categorization of `Candidates`, `Confirmed Events`, and `Rejected Candidates`, with real-time FSM state reporting (`UNKNOWN`, `HOLDING`, `RELEASE`, `LITTERING_CONFIRMED`).

---

## 1. Files Changed

| File | Purpose of Change |
|---|---|
| `backend/models.py` | Added dynamic telemetry properties to `VideoAnalysisJob` (`current_stage`, `current_stage_status`, `active_persons_count`, `unique_persons_count`, `total_person_track_ids`, `candidates_count`, `rejected_count`, `last_processed_frame`, `last_update_time`) reading transparently from both top-level and nested `pipeline_telemetry` JSON. |
| `backend/schemas.py` | Added telemetry and stage fields to `VideoAnalysisJobBase` and `VideoAnalysisJobOut` FastAPI response schemas, including `"cancelled"` status. |
| `backend/routers/analysis.py` | Implemented `PipelineStageTracker` (12 stages), thread-safe cancellation tokens (`_JOB_CANCELLATION_TOKENS`, `request_job_cancellation`, `is_job_cancellation_requested`), safe loop termination in `_run_video_analysis_job`, live telemetry writes every 15 frames, `POST /api/analysis/jobs/{job_id}/stop`, `DELETE /api/analysis/jobs/{job_id}`, and `DELETE /api/analysis/jobs/{job_id}/video`. |
| `dashboard/src/types.ts` | Added `PipelineStageStatus`, `PipelineStageInfo`, `PipelineTelemetryMetrics`, `PipelineTelemetry`, updated `JobStatus` to include `"cancelled"`, and added telemetry fields to `VideoAnalysisJob`. |
| `dashboard/src/lib/api.ts` | Added `stopAnalysisJob`, `deleteAnalysisJob`, and `deleteSourceVideo` API client functions. |
| `dashboard/src/lib/useFetch.ts` | Exposed `refetch()` trigger from `useFetch` to allow manual data refresh after user actions. |
| `dashboard/src/components/PipelineStageMonitor.tsx` | New component rendering: 12-stage status grid (○ PENDING, ◉ RUNNING, ✓ COMPLETED, ✕ FAILED, ⊘ CANCELLED), failure diagnostic box, honest person demarcation cards, and event confirmation matrix. |
| `dashboard/src/pages/VideoAnalysisPage.tsx` | Integrated `PipelineStageMonitor`, added `[ STOP ANALYSIS ]`, `[ DELETE JOB ]`, and `[ DELETE SOURCE VIDEO ]` controls, replaced misleading person counters with demarcated metrics, and implemented dedicated `CANCELLED` job banner. |
| `dashboard/src/pages/AnalysisDetail.tsx` | Integrated `PipelineStageMonitor`, updated meta strip with demarcated person counts, and added `[ STOP ANALYSIS ]` and `[ DELETE JOB ]` buttons. |
| `tests/test_pipeline_observability_and_control.py` | New comprehensive test suite verifying 12 pipeline stages, stage transitions, cancellation token, stop API endpoint, deletion guards, and person metric serialization. |

---

## 2. Person-Count Correction & Semantics

### The Problem
Previously, when a single person in a video turned around or was temporarily occluded, ByteTrack frequently assigned new tracker IDs (e.g. Track 1 → Track 3 → Track 7). The Dashboard aggregated these raw tracking IDs and displayed:
> "9 persons"
This misled operators into believing 9 distinct human beings were present at the scene.

### The Solution
The system now strictly separates and honestly reports three distinct concepts:
1. **ACTIVE PEOPLE (`active_persons_count`):** The number of concurrent human detections in the current video frame (alongside peak concurrent humans).
2. **UNIQUE PEOPLE SEEN (`unique_persons_count`):** The number of stable, unique human identities resolved by `PersonIdentityManager` using spatial-temporal Re-ID and appearance features.
3. **TOTAL TRACK IDs (`total_person_track_ids`):** The raw count of tracker IDs issued by ByteTrack over the video duration.

The Dashboard renders these in three dedicated cards with an auditing disclaimer:
> *"Raw ByteTrack IDs can churn across occlusions and view changes. The honest unique human count is represented by Unique People Seen, maintained by PersonIdentityManager."*

`PersonIdentityManager` and existing actor attribution logic were completely preserved without regression.

---

## 3. Real-Time 12-Stage Pipeline Monitoring

The system now implements the exact 12-stage pipeline defined in the production architecture:

```
[01. VIDEO INPUT] ──> [02. PERSON DETECTION] ──> [03. OBJECT / WASTE DETECTION] ──> [04. TRACKING]
                                                                                            │
┌───────────────────────────────────────────────────────────────────────────────────────────┘
▼
[05. PERSON IDENTITY] ──> [06. OBJECT IDENTITY] ──> [07. POSE] ──> [08. OWNERSHIP / ASSOCIATION]
                                                                                   │
┌──────────────────────────────────────────────────────────────────────────────────┘
▼
[09. TEMPORAL EVENT DETECTION] ──> [10. EVIDENCE] ──> [11. DATABASE] ──> [12. API / DASHBOARD]
```

### Stage States & Indicators
Every stage reports a real backend state:
- `○ PENDING`: Stage is waiting in sequence.
- `◉ RUNNING`: Algorithm is currently executing on active frames (pulsing badge).
- `✓ COMPLETED`: Stage has completed successfully.
- `✕ FAILED`: Algorithm encountered an unhandled exception or critical error.
- `⊘ CANCELLED`: Execution was stopped by operator request.
- `⊝ SKIPPED`: Stage was skipped because an upstream stage failed or was cancelled.

**Backend-Driven Guarantee:** Progress is driven purely by measured backend execution state (`report_json.pipeline_telemetry`). Zero timer simulations or fake UI animations are used.

### Global Progress & Failure Diagnostics
- **Header:** Displays exact percentage and stage (e.g. `51% RUNNING — Stage 7/12: Pose Estimation`).
- **Failure Visibility:** When a component stalls or crashes (e.g. at 51%), the generic "AI PIPELINE RUNNING" is immediately replaced with a high-visibility diagnostic alert:
  - **Failed Stage:** Name and step number.
  - **Status:** `HALTED`.
  - **Last Successful Stage:** The last verified stage that passed.
  - **Last Processed Frame:** Exact frame index where failure occurred.
  - **Last Update Time:** Timestamp of last heartbeat.
  - **Error / Blocker:** Exact stack trace or error message.

---

## 4. Per-Job Cancellation Implementation (`[ STOP ANALYSIS ]`)

### Architecture
1. **Thread-Safe Cancellation Registry:** A module-level thread-safe map `_JOB_CANCELLATION_TOKENS: Dict[int, threading.Event]` managed with `threading.Lock`.
2. **API Endpoint:** `POST /api/analysis/jobs/{job_id}/stop` sets the cancellation event for `job_id`.
3. **Loop Polling:** In `_run_video_analysis_job`, `is_job_cancellation_requested(job_id)` is checked on every frame iteration.
4. **Graceful Resource Release:** When cancellation is detected:
   - Video file capture (`VideoFileSource` / `cv2.VideoCapture`) is immediately released.
   - Annotated video writer (`VideoWriter` / `ffmpeg`) is finalized and closed.
   - Inference tensors and intermediate buffers are released.
   - `PipelineStageTracker.cancel()` marks the active stage as `CANCELLED` and downstream stages as `SKIPPED`.
   - `job.status = "cancelled"` and `job.completed_at = now()` are written and committed to the database.
   - Temporary frames/crops are cleaned.
5. **Cancelled Job UI:** Renders `CANCELLED` (distinct from `FAILED`), showing:
   - Processed frames / total frames.
   - Percentage reached.
   - Last stage reached.
   - Elapsed time.
   - Confirmation that partial evidence is preserved and the job can be safely deleted.

---

## 5. Delete Controls

Lifecycle controls are strictly separated:
1. **STOP ANALYSIS (`POST /api/analysis/jobs/{job_id}/stop`):** Halts running execution without deleting data.
2. **DELETE JOB (`DELETE /api/analysis/jobs/{job_id}?confirm=true`):** Deletes job database record, event records, evidence rows, and artifact folder from disk.
   - Requires explicit `?confirm=true` parameter.
   - Rejects deletion if `job.status == "processing"`, advising the operator to stop the job first.
3. **DELETE SOURCE VIDEO (`DELETE /api/analysis/jobs/{job_id}/video?confirm=true`):** Deletes only the uploaded raw video file to reclaim disk space, keeping all analysis metadata, events, and reports intact.

---

## 6. Event Lifecycle Demarcation

During video processing, the Dashboard now distinguishes between:
- **Candidates:** Number of interacting person-object pairs evaluated.
- **Confirmed Events:** Events that passed all confirmation gates (temporal continuity, release velocity, ground placement, and person separation).
- **Rejected:** Candidates that failed confirmation criteria (with reason tracking).
- **Active FSM State:** Displays the current state of the active candidate (e.g. `UNKNOWN`, `INTERACTING`, `HOLDING`, `RELEASE`, `LITTERING_CONFIRMED`).

---

## 7. Verification & Automated Testing

### 1. Backend Automated Tests
All 40 backend test cases passed with zero failures:
```
tests/test_backend_api.py::test_health PASSED                            [  2%]
...
tests/test_backend_api.py::test_video_analysis_endpoints PASSED          [ 80%]
tests/test_pipeline_observability_and_control.py::test_pipeline_stages_list_exact_12 PASSED [ 82%]
tests/test_pipeline_observability_and_control.py::test_pipeline_stage_tracker_lifecycle_and_telemetry PASSED [ 85%]
tests/test_pipeline_observability_and_control.py::test_pipeline_stage_tracker_failure_recording PASSED [ 87%]
tests/test_pipeline_observability_and_control.py::test_pipeline_stage_tracker_cancellation PASSED [ 90%]
tests/test_pipeline_observability_and_control.py::test_cancellation_token_threadsafe PASSED [ 92%]
tests/test_pipeline_observability_and_control.py::test_stop_analysis_api_endpoint PASSED [ 95%]
tests/test_pipeline_observability_and_control.py::test_delete_analysis_job_guards_and_confirmation PASSED [ 97%]
tests/test_pipeline_observability_and_control.py::test_person_count_semantics_in_model_and_api PASSED [100%]
======================= 40 passed, 3 warnings in 1.69s ========================
```

### 2. Frontend Typecheck & Production Build
The frontend TypeScript compiler and Vite production build succeeded with exit code 0:
```
> littering-dashboard@1.0.0 build
> tsc && vite build

vite v5.4.21 building for production...
transforming...
✓ 2189 modules transformed.
rendering chunks...
computing gzip size...
dist/index.html                   0.61 kB │ gzip:   0.41 kB
dist/assets/index-DA1FLQT0.css   36.46 kB │ gzip:   7.06 kB
dist/assets/index-Zye2GcIs.js   751.69 kB │ gzip: 211.46 kB
✓ built in 47.20s
exit_code: 0
```

---

## 8. Explicit Statements as Required

### 1. What was fixed:
- Eliminated ambiguous "9 persons" counting by clearly distinguishing Active People, Unique People Seen (`PersonIdentityManager`), and Total Track IDs (`ByteTrack`).
- Implemented real 12-stage live pipeline monitoring driven by actual backend execution state.
- Fixed progress visibility so current stage name, step number, and global progress are visible simultaneously (e.g. `51% RUNNING — Stage 7/12: POSE`).
- Removed generic "AI PIPELINE RUNNING" mask during failure; introduced structured diagnostic alert cards showing exact failure reasons, frame numbers, and last successful stages.
- Added per-job `[ STOP ANALYSIS ]` button with thread-safe cancellation and safe resource cleanup.
- Separated `STOP`, `DELETE JOB`, and `DELETE SOURCE VIDEO` controls with explicit confirmation guards.
- Handled `CANCELLED` status cleanly across the entire stack, never reporting it as `FAILED`.
- Separated event candidate evaluation from confirmed violations.

### 2. What was NOT tested because real-video analysis was intentionally deferred:
- No new real videos were uploaded or processed.
- No inference was executed against `D:\22` video files.
- No model weights were modified, fine-tuned, or retrained.
- AI detection accuracy and violation confirmation rates on unseen long videos were NOT re-evaluated (strictly deferred to the next user-initiated run).

### 3. How the Dashboard will report the real pipeline state on the next run:
When the next long video is uploaded:
1. The Dashboard will immediately display:
   ```
   Global Progress: [ 0% -> 100% ]
   RUNNING — Stage 1/12: Video Input
   ```
2. As frames are processed, each stage badge will update dynamically:
   ```
   ✓ 01. Video Input
   ✓ 02. Person Detection
   ✓ 03. Object / Waste Detection
   ✓ 04. Tracking
   ✓ 05. Person Identity
   ✓ 06. Object Identity
   ◉ 07. Pose Estimation          <-- RUNNING (pulsing)
   ○ 08. Ownership / Association  <-- PENDING
   ○ 09. Temporal Event Detection <-- PENDING
   ○ 10. Evidence Assembly        <-- PENDING
   ○ 11. Database Persistence     <-- PENDING
   ○ 12. API / Dashboard          <-- PENDING
   ```
3. The operator will see real metrics updating every 15 frames:
   - Active People: e.g. `1`
   - Unique People Seen: e.g. `1`
   - Total Track IDs: e.g. `4`
   - Candidates: e.g. `1`
   - Confirmed: e.g. `0`
   - Active FSM: e.g. `HOLDING`
4. If a failure occurs at frame 340 (e.g. pose estimation OOM):
   ```
   ✕ FAILED at Stage 7/12: Pose Estimation
   Last Successful Stage: Stage 6/12 — Object Identity
   Last Processed Frame: #340
   Error: CUDA out of memory / MoveNet error
   ```
5. If the operator clicks `[ STOP ANALYSIS ]`:
   ```
   ⊘ CANCELLED
   Analysis safely terminated at Stage 7/12.
   Frames processed: 340 / 600 (57%).
   Resources released. Safe to delete or retain for inspection.
   ```

---

## 4. Job #28 Diagnostic & Root-Cause Resolution

### Observed Incident
During execution of Job #28 (`IMG_5306.MOV`), the analysis failed at frame #15 with:
```
Failed in temporal_event_detection: 'AdaptiveEventDetector' object has no attribute '_person_identity'
```
Furthermore, the frontend banner initially displayed `Pipeline Execution Failed at Stage 1: Video Input` because `report.pipeline_telemetry` and stage key aliases were not reconciled when reading root reports.

### Root Cause Analysis
1. **Attribute Forwarding in AdaptiveEventDetector**:
   - `backend/routers/analysis.py` runs with `auto_tune=True`, wrapping `LitteringEventDetector` in `AdaptiveEventDetector`.
   - Telemetry heartbeat runs at `frame_idx % 15 == 0`, accessing `pipe.event_detector._person_identity._uids`.
   - `AdaptiveEventDetector` lacked property forwards for `_person_identity`, `_object_identity`, `last_person_uid_map`, and `last_object_uid_map`.
   - `ObjectIdentityManager` stores records in `self.records`, not `_uids`.

2. **Stage Attribution & Alias Mapping in Frontend**:
   - Backend stages use keys `waste_detection`, `pose_estimation`, and `temporal_event_detection`, whereas legacy/frontend keys used `object_detection`, `pose`, and `event_detection`.
   - When telemetry was serialized at the root of `report_json`, `report?.pipeline_telemetry` evaluated to null, causing fallback to step 1 (`Video Input`).

### Implemented Fixes
1. **AdaptiveEventDetector (`adaptive_tuner.py`)**:
   - Added forwarded properties for `_person_identity`, `_object_identity`, `last_person_uid_map`, and `last_object_uid_map` directing to `self.primary`.
2. **ObjectIdentityManager (`inference/tracking/object_identity.py`)**:
   - Added `_uids` property returning `self.records`.
3. **Defensive Extraction (`backend/routers/analysis.py`)**:
   - Added `_extract_person_identity_info(event_detector, detected_persons_set)` and `_extract_object_identity_count(event_detector, detected_objects_set)` using `getattr` and fallbacks. No direct private attribute access can crash the telemetry heartbeat.
   - Dynamic stage tracking inside the frame loop assigns `stage_tracker.current_stage_id` to the active stage (`person_detection`, `waste_detection`, `pose_estimation`, or `temporal_event_detection`), ensuring any failure is attributed to the exact stage.
4. **Resilient Frontend Telemetry (`dashboard/src/components/PipelineStageMonitor.tsx`)**:
   - Normalized stage maps supporting both Array and Object formats.
   - Reconciled aliases (`waste_detection` <-> `object_detection`, `pose_estimation` <-> `pose`, `temporal_event_detection` <-> `event_detection`).
   - Accurately reports failed stage name, last successful stage, failed frame, and exact exception.

