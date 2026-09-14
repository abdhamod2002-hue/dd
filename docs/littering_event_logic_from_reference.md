# Littering Event Logic From Reference Project

Reference project: `https://github.com/vraj1231/Illegal-Dumping-Action-Detection`

> Note: in the current sandbox, cloning the reference repository failed because
> outbound network access was unavailable. This note is therefore based on the
> supplied project description and the common notebook-style structure of that
> reference implementation. It is intentionally a design contract, not a code
> port.

## Summary of the reference approach

The reference project detects illegal dumping with separate detectors for:

- person
- trash / garbage bag
- optionally license plate

It uses YOLOv5-style detection plus DeepSORT tracking. The core decision rule is
simple:

1. Detect a person.
2. Detect a trash object.
3. If the trash object is near a person, associate them by distance.
4. If the person later moves away while the trash remains, mark the event as
   illegal dumping.

This is a useful proof-of-concept because it demonstrates that illegal dumping
can be modeled as a person-object interaction plus a departure condition.

## Main rules and thresholds

The reference logic is effectively:

```text
if trash_near_person and person_moved_away:
    illegal_dumping = True
```

Typical thresholds in such notebook demos are:

- person confidence threshold
- trash confidence threshold
- distance threshold for "near"
- movement threshold for "departed"
- tracker max age for DeepSORT tracks

The exact values are not production-tuned and are not sufficient for reliable
evidence generation.

## Strengths

- Simple and easy to understand.
- Works as a proof-of-concept for controlled videos.
- Separates person detection and trash detection.
- Uses tracking so the system can reason about motion over time.
- Demonstrates the key idea: dumping is a temporal interaction, not a static
  object classification problem.

## Weaknesses

The reference logic is too weak for a production evidence system:

1. **Single proximity rule**
   - "Trash near person" can happen when a person merely walks past existing
     trash.

2. **No carry evidence**
   - It does not require the bag to be carried or held before release.

3. **No release transition**
   - It does not distinguish a throw/drop from a person simply walking away
     from an already-grounded bag.

4. **No stationary ground evidence**
   - It does not require the bag to remain stable on the ground after release.

5. **No confidence scoring**
   - Events are binary, so operators cannot see why an event was accepted or
     rejected.

6. **No explainable rejection reasons**
   - Failed candidates are silently ignored.

7. **Weak multi-person handling**
   - If another person is closer to the bag, the reference rule can accuse the
     wrong person.

8. **Tracker dependence**
   - DeepSORT ID switches can break the person-object association.

9. **No production evidence pipeline**
   - No API, evidence store, dashboard, Docker Compose, or review workflow.

## How this project improves the reference logic

Our implementation keeps the high-level idea but makes it temporal,
configurable, and explainable.

### State machine

The new detector maintains a state machine per `(person_track_id, bag_track_id)`
pair:

```text
NO_BAG
  → BAG_NEAR_PERSON
  → BAG_CARRIED
  → BAG_RELEASED
  → BAG_ON_GROUND
  → PERSON_DEPARTED
  → VIOLATION_CONFIRMED
```

### Required evidence

A violation is confirmed only when all of the following are observed:

- stable person track ID
- bag associated with that person for at least `MIN_CARRIED_FRAMES`
- clear release/separation transition
- bag remains spatially stable for at least `MIN_STATIONARY_FRAMES`
- associated person moves farther than `DEPARTURE_DISTANCE_RATIO`
- no other person has a stronger association during the event window
- event confidence is at least `MIN_EVENT_CONFIDENCE`
- if CSRT fallback was used, YOLO must re-confirm the bag before final evidence

### Explainability

Every rejected candidate receives a reason code, for example:

- `PERSON_NOT_DETECTED`
- `BAG_NOT_DETECTED`
- `LOW_BAG_CONFIDENCE`
- `CSRT_NOT_RECONFIRMED`
- `ASSOCIATION_AMBIGUOUS`
- `NOT_ENOUGH_CARRIED_FRAMES`
- `NO_RELEASE_TRANSITION`
- `BAG_NOT_STATIONARY`
- `PERSON_DID_NOT_DEPART`
- `OTHER_PERSON_CLOSER`
- `EVENT_CONFIDENCE_TOO_LOW`

### Configurable thresholds

Initial values live in `config/events.yaml`. They are starting points, not final
truths, and must be tuned per camera/scene using real validation clips.

## Contract before code changes

This document is the contract:

- Do not copy-paste reference notebook code.
- Keep our FastAPI + YOLO + ByteTrack + Docker + dashboard architecture.
- Preserve existing APIs.
- Prefer temporal evidence over single-frame decisions.
- Prefer explainable rejection reasons over silent failures.
- Do not claim 100% accuracy.
- Report only real metrics from manual review or automated evaluation.
