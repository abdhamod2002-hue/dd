# Littering Event Detector

`littering_event_detector.py` implements a temporal, state-machine-based littering
event detector for our existing YOLO + ByteTrack pipeline.

It is inspired by the reference illegal-dumping project, but it is intentionally
stronger and more explainable.

## Why this module exists

A simple rule such as:

```text
trash near person + person moves away → illegal dumping
```

is not enough for an evidence system. It can accuse a person who merely walked
near existing trash, or a person who put an object down and stayed nearby.

This detector requires a full behavioral sequence:

```text
person carries bag → releases bag → bag becomes stationary on ground → person departs
```

and records evidence for every accepted or rejected candidate.

## States

The detector maintains one state machine per `(person_track_id, bag_track_id)`
pair:

| State | Meaning |
|---|---|
| `NO_BAG` | No bag is currently associated with this person. |
| `BAG_NEAR_PERSON` | Bag is spatially near the person. |
| `BAG_CARRIED` | Bag has been associated long enough to count as carried. |
| `BAG_RELEASED` | Bag separated from the person after being carried. |
| `BAG_ON_GROUND` | Released bag remained spatially stable long enough. |
| `PERSON_DEPARTED` | Person moved farther than the departure threshold. |
| `VIOLATION_CONFIRMED` | All evidence requirements passed. |

## Required evidence

A confirmed event requires:

1. Stable person track ID.
2. Bag associated with that person for at least `MIN_CARRIED_FRAMES`.
3. A clear release/separation transition.
4. Bag remains spatially stable for at least `MIN_STATIONARY_FRAMES`.
5. Person moves farther than `DEPARTURE_DISTANCE_RATIO`.
6. No other person has a stronger association during the event window.
7. Event confidence is at least `MIN_EVENT_CONFIDENCE`.
8. If CSRT fallback was used, YOLO must re-confirm the bag before final
   evidence.

## Rejection reasons

Rejected candidates are emitted with explicit reason codes:

| Reason | Meaning |
|---|---|
| `PERSON_NOT_DETECTED` | Person track disappeared or was not stable enough. |
| `BAG_NOT_DETECTED` | No YOLO-confirmed bag track was available. |
| `LOW_BAG_CONFIDENCE` | Bag confidence was too low. |
| `CSRT_NOT_RECONFIRMED` | CSRT bridged a gap, but YOLO never re-confirmed the bag. |
| `ASSOCIATION_AMBIGUOUS` | Multiple persons were too close in association score. |
| `NOT_ENOUGH_CARRIED_FRAMES` | The person did not carry the bag long enough. |
| `NO_RELEASE_TRANSITION` | No clear separation after carrying. |
| `BAG_NOT_STATIONARY` | Bag did not remain stable on the ground. |
| `PERSON_DID_NOT_DEPART` | Person did not move far enough away. |
| `OTHER_PERSON_CLOSER` | Another person had a stronger association with the bag. |
| `EVENT_CONFIDENCE_TOO_LOW` | Evidence existed but confidence was below threshold. |

## Configuration

Default configuration lives in `config/events.yaml`.

Important parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `ANALYSIS_FPS` | 8.0 | Expected analysis frame rate used for temporal normalization. |
| `DETECTION_LOW_CONF` | 0.10 | Minimum detection confidence accepted as a candidate. |
| `DETECTION_HIGH_CONF` | 0.25 | Confidence used for stronger detection scoring. |
| `MIN_CARRIED_FRAMES` | 8 | Minimum smoothed frames for carry evidence. |
| `MIN_STATIONARY_FRAMES` | 16 | Minimum smoothed frames for ground stability. |
| `MIN_DEPARTED_FRAMES` | 4 | Minimum smoothed frames for departure. |
| `NEAR_DISTANCE_RATIO` | 0.25 | Person-height-normalized distance for "near". |
| `DEPARTURE_DISTANCE_RATIO` | 1.25 | Person-height-normalized distance for departure. |
| `MIN_EVENT_CONFIDENCE` | 0.80 | Minimum confidence to confirm. |
| `MAX_FALLBACK_TRACKER_GAP_FRAMES` | 16 | Maximum CSRT-only gap before rejection. |
| `MAX_PAIR_AGE_FRAMES` | 30 | How long an inactive pair is kept before finalization. |

These are not final truths. Tune them per camera/scene.

## Confidence and evidence scores

Each event includes an evidence breakdown:

| Score | Meaning |
|---|---|
| `carry_score` | How long the bag was associated as carried. |
| `release_score` | Whether a clear release transition was observed. |
| `stationary_score` | How stable the bag remained after release. |
| `departure_score` | How far the person moved after release. |
| `association_score` | Quality of person-bag association. |
| `detection_score` | Quality of bag detection confidence. |
| `confidence` | Final weighted confidence in `[0, 1]`. |

The final confidence is intentionally not treated as legal certainty. It is a
review priority score.

## CSRT fallback policy

CSRT may be used only as a short bridge when YOLO misses a bag.

Rules:

- CSRT can be initialized only from a YOLO-confirmed bag detection.
- CSRT can bridge only a short gap controlled by
  `MAX_FALLBACK_TRACKER_GAP_FRAMES`.
- CSRT cannot independently confirm a violation.
- YOLO must re-confirm the bag before final evidence is produced.

If this policy is not satisfied, the detector rejects the candidate with
`CSRT_NOT_RECONFIRMED` or `LOW_BAG_CONFIDENCE`.

## Integration

The detector is integrated into the video analysis endpoint:

- `backend/routers/analysis.py`
- `/api/analysis/upload`
- `/api/analysis/jobs/{job_id}`

The job report now includes an `event_detector` section:

```json
{
  "event_detector": {
    "summary": {
      "total_candidates": 3,
      "confirmed_violations": 1,
      "rejected_candidates": 2,
      "acceptance_rate": 0.3333,
      "rejection_reason_counts": {
        "PERSON_DID_NOT_DEPART": 1,
        "LOW_BAG_CONFIDENCE": 1
      }
    },
    "confirmed_violations": [],
    "rejected_candidates": []
  }
}
```

The dashboard's Video Analysis page displays this diagnostics section.

## Tuning for a new camera or scene

Recommended workflow:

1. Record 5–10 short clips:
   - real littering
   - person walking near existing trash
   - person putting an object down and staying
   - windy bag movement
   - occlusion / low light
2. Run analysis through the dashboard or `scripts/evaluate.py`.
3. Inspect rejected candidates and reason codes.
4. Tune thresholds:
   - If too many false positives:
     - increase `MIN_CARRIED_FRAMES`
     - increase `MIN_STATIONARY_FRAMES`
     - increase `DEPARTURE_DISTANCE_RATIO`
     - increase `MIN_EVENT_CONFIDENCE`
   - If too many missed events:
     - decrease `MIN_CARRIED_FRAMES`
     - decrease `MIN_STATIONARY_FRAMES`
     - decrease `DEPARTURE_DISTANCE_RATIO`
     - lower `DETECTION_HIGH_CONF`
   - If association is unstable:
     - adjust `NEAR_DISTANCE_RATIO`
     - adjust `ASSOCIATION_MARGIN`
     - adjust `MAX_PAIR_AGE_FRAMES`
5. Re-run and record real counts only.

## Important limitation

This system is assistive. It detects potential littering events for human
review. It does not declare a legal violation and it is not 100% accurate.
