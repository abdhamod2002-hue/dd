# System Architecture

## Overview

The system is a pipeline that turns a static iPhone camera feed into
reviewable littering evidence. It combines off-the-shelf CV models
(YOLO, ByteTrack, MoveNet) with a **custom temporal behavior layer**
that is the project's core contribution.

```
📱 iPhone (Camo/Iriun USB)
        │
        ▼
┌─────────────────┐
│ CameraSource     │  🟢 OpenCV wrapper
│ (cv2.VideoCapture)│
└────────┬────────┘
         │  every frame @ full FPS
         ▼
┌─────────────────┐
│ CircularBuffer   │  🔴 time-based ring buffer (always retains last N s)
└────────┬────────┘
         │  throttled to analysis_fps
         ▼
┌─────────────────┐
│ YoloDetector     │  🟢 best.pt (litter) + yolov8n (person)
└────────┬────────┘
         │  detections
         ▼
┌─────────────────┐
│ ByteTrack        │  🟢 namespaced IDs (person 1..N, object 10001..N)
└────────┬────────┘
         │  tracks
         ▼
┌─────────────────┐
│ MoveNet (lazy)   │  🟢 pose only on tracked persons — biggest CPU saving
└────────┬────────┘
         │  keypoints
         ▼
┌──────────────────────┐
│ PersonObjectAssoc    │  🔴 sticky binding, persistence gate,
│                      │     ID-switch rebind, hand-occlusion fallback
└────────┬─────────────┘
         │  PairObservation per (person, object)
         ▼
┌──────────────────────┐
│ LitteringStateMachine│  🔴 FSM with forward path + reversion edges
│ (one per pair)       │     (HOLDING→RELEASE→GROUND→AWAY→SUSPICIOUS)
└────────┬─────────────┘
         │  on SUSPICIOUS
         ▼
┌──────────────────────┐
│ TemporalVoter        │  🔴 weighted multi-observation scoring + decay
└────────┬─────────────┘
         │  CONFIRM / REVERT / HOLD
         ▼
┌──────────────────────┐
│ EvidenceManager      │  🔴 snapshot + pre/post clip from buffer
└────────┬─────────────┘
         │  POST (async, optional)
         ▼
┌──────────────────────┐
│ FastAPI + PostgreSQL  │  🔵 cameras / events / evidence / statistics
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│ React Dashboard       │  🔵 live feed, violations list, evidence viewer
└──────────────────────┘
```

## Core contribution (the 🔴 layers)

The defensible academic work is concentrated in five modules, all
unit-tested without a camera:

1. **CircularFrameBuffer** — time-based ring buffer so evidence never
   starts recording late.
2. **PersonObjectAssociator** — sticky binding with persistence gating,
   ID-switch rebind (handles tracker loss mid-throw), hand-occlusion
   fallback (torso distance when wrist missing).
3. **LitteringStateMachine** — FSM with explicit reversion edges:
   RELEASE→HOLDING (re-grasp), GROUND→NORMAL (put-down not litter),
   GROUND→HOLDING (retrieval), PERSON_AWAY→GROUND (returned),
   SUSPICIOUS→NORMAL (decay), any→NORMAL (track loss).
4. **TemporalVoter** — weighted multi-observation score with decay;
   re-grasp and person-returned are strong negatives (-4.0) that flip
   the decision.
5. **EvidenceManager** — assembles pre+event+post from the buffer with
   explicit post-window finalize (explains the ~3s demo latency).
6. **LitteringEventDetector** — newer explainable state machine
   (`littering_event_detector.py`) for carry → release → stationary ground →
   departure evidence, explicit rejection reasons, configurable thresholds in
   `config/events.yaml`, and CSRT fallback that cannot confirm without YOLO
   re-confirmation.

## Scope (explicit, not a limitation)

- **Static camera only.** Camera motion compensation is Future Work.
- **Daytime / adequate lighting.**
- **Face recognition (DeepFace) is optional / future phase** — run only
  on the best face frame post-event, not every frame.
- **System detects *potential* littering events for review** — it does
  not declare legal violation.

## Failure modes addressed in design

| Failure | Mitigation |
|---|---|
| Tracker loses object mid-throw (ID switch) | Associator rebind by centroid+class+size within `reassoc_window` |
| Hand occluded at release (wrist missing) | Torso-center fallback distance; release detected by relative motion |
| Multiple people, "away" ambiguity | `person_moving_away` = relative centroid motion, not absolute distance |
| Put-down mistaken for littering | `abandon_window` reversion: GROUND→NORMAL if person never leaves |
| Re-grasp mistaken for littering | RELEASE→HOLDING reversion + voter `w_regrasp=-4.0` |
| Single-frame heuristic accusation | Multi-observation voting with decay; FSM dwell times |
