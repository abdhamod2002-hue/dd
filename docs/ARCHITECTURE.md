# System Architecture

## Overview

The system turns a static iPhone / CCTV video into **event-focused littering
evidence**. Off-the-shelf CV (YOLO, ByteTrack, MoveNet) feeds a custom
temporal behavior layer that is the product's core contribution.

```
iPhone / uploaded video
        │
        ▼
┌─────────────────┐
│ CameraSource     │  OpenCV capture / VideoFileSource
│ CircularBuffer   │  time-based ring for evidence clips
└────────┬────────┘
         ▼
┌─────────────────┐
│ YoloDetector     │  person + litter + bag weights; HSV color fallback
│ NoveltyDetector  │  proposal-only (never event truth alone)
└────────┬────────┘
         ▼
┌─────────────────┐
│ ByteTrack        │  namespaced track IDs
│ PersonIdentity   │  stable person UID across churn
│ ObjectIdentity   │  stable object UID across churn
└────────┬────────┘
         ▼
┌─────────────────┐
│ MoveNet (lazy)   │  pose on analysis ticks only
└────────┬────────┘
         ▼
┌──────────────────────┐
│ LitteringEventDetector│  AUTHORITATIVE temporal FSM
│ (+ AdaptiveEventDetector wrapper in production)
│ carry → release → ground → depart / abandon
│ explicit rejection reasons; config/events.yaml
└────────┬─────────────┘
         │ confirmed only
         ▼
┌──────────────────────┐
│ Evidence package     │  from ORIGINAL video (not annotated)
│ person / face / waste / clip / sequence stills
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ FastAPI + PostgreSQL │  jobs / events / evidence
└────────┬─────────────┘
         ▼
┌──────────────────────┐
│ React Dashboard      │  ForensicAssetPanel = primary
│                      │  full analyzed video = Technical Review only
└──────────────────────┘
```

## Core contribution

1. **CircularFrameBuffer** — evidence never starts late.
2. **Stable identity** — person/object UIDs survive tracker ID churn.
3. **Ownership freeze** — actor/object IDs locked at carry start.
4. **LitteringEventDetector** — explainable carry→release→ground→depart FSM
   with pick-up reversion and bin-zone rejection.
5. **EvidenceManager / write_event_evidence_package** — original-video crops
   and H.264 event clips.

## Archived (not production)

Legacy modules under `_archive/` (old `LitteringStateMachine`,
`TemporalVoter`, `PersonObjectAssociator`) are **not** on the live path.
Do not treat historical docs that name them as current architecture.

## Scope

- Static camera only (no ego-motion compensation).
- Daytime / adequate lighting preferred.
- Detection does not decide littering; Dashboard does not invent actors.
