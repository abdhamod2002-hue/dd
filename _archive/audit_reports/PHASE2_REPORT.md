# Phase 2 Report — Pose-Based Association

**Status:** Implemented, disabled by default, **not committed** (awaiting review).
**Scope of this step:** pixel gate lock-in, baseline commit, Phase 2 carrying-zone
redefinition, and documentation of a deviation forced by a genuine technical constraint.

---

## 1. Pre-flight: pixel gate + baseline (Steps 1 & 2)

1. **Pixel gate set to 12px** — `config/events.yaml`:
   `STATIONARY_MAX_PIXEL_STEP: 12.0`. This is the value confirmed safe by the
   Phase-1 measurements on the 5 test videos (`10px` breaks `IMG_5120`,
   `>=12px` is safe on all 5). No `15px` value exists anywhere in the codebase.
2. **Baseline commit** — already present as
   `01e212e baseline before phase 2 - pixel gate at 12px`. Verified to be a
   *clean* rollback point:
   - the committed `config/events.yaml` has `STATIONARY_MAX_PIXEL_STEP: 12.0`;
   - the committed `littering_event_detector.py` contains **zero** Phase 2 code
     (`grep -c pose_association` == 0) and does contain the pixel-gate field/logic.
   The working tree now carries the uncommitted Phase 2 changes on top of this
   baseline, so a `git checkout 01e212e` restores the pre-Phase-2 state exactly.

> Note: a new commit was **not** re-created for Step 2. The requested commit
> already existed as `HEAD`. Re-committing now would have baked the in-progress
> Phase 2 edits into the "baseline", destroying its purpose as a pre-Phase-2
> rollback point.

---

## 2. What Phase 2 does

Phase 2 adds a **body-relative pose score** to the person↔bag association and
**blends** it with the existing geometric score:

```
score = (1 - blend) * legacy + blend * pose
```

- `legacy`  = proximity + containment + detection confidence (unchanged).
- `pose`    = weighted mean of three cues, renormalised over whichever cues are
  actually available for that person (graceful degradation under occlusion):

  | cue      | weight | meaning                                                    |
  |----------|--------|------------------------------------------------------------|
  | wrist    | 0.4    | proximity of bag centroid to the nearer wrist (the hand)  |
  | zone     | 0.3    | 1.0 if the bag lies inside the body-relative carrying zone|
  | motion   | 0.3    | person and bag move together (shared per-tick velocity)   |

**Implementation locations** (`littering_event_detector.py`):
- `_shoulder_line()` — top anchor of the carrying zone.
- `_mean_step()` — per-tick centroid displacement (px/tick, not px/s).
- `_pose_association_score()` — the three-cue blend with graceful degradation.
- `LitteringEventDetector._update_person_history()` — rolling per-person history.
- `LitteringEventDetector._evaluate_pair()` — blends pose into the final score.

**Safety:** with `POSE_ASSOCIATION_ENABLED = false` (default) or
`POSE_SCORE_BLEND = 0.0`, behaviour is bit-identical to pre-Phase-2. The feature
is therefore a no-op until explicitly enabled in `config/events.yaml`.

---

## 3. Carrying-zone redefinition (the change requested)

**Original plan:** the carrying zone was specified as the vertical band
*"between the shoulders and the knees"*.

**Redefined (this step):** the carrying zone is the vertical band
**from the shoulder line down to the bottom of the person's bounding box**,
horizontally spanning the person's bbox width plus a `0.30` margin.

This was enforced by setting both vertical margins to zero in `config/events.yaml`:

```yaml
POSE_ZONE_VERTICAL_ABOVE: 0.0    # band top == shoulder line
POSE_ZONE_VERTICAL_BELOW: 0.0    # band bottom == bbox bottom
```

So the zone is exactly `shoulder_line <= bag_centroid_y <= bbox_bottom`
(within the horizontal margin). Boundary points are inclusive.

**Verification (smoke test on synthetic geometry):**

| case                              | expected | actual |
|-----------------------------------|----------|--------|
| bag on body (y between sh & bbox) | 1.0      | 1.0    |
| bag above shoulders               | 0.0      | 0.0    |
| bag below bbox bottom             | 0.0      | 0.0    |
| bag exactly at shoulder line      | 1.0      | 1.0    |
| bag exactly at bbox bottom        | 1.0      | 1.0    |
| shoulders missing → fallback top  | 1.0      | 1.0    |

Fallback when shoulders are occluded: `top = bbox_top + 0.45 * person_height`
(upper-body proxy), so the zone still degrades sensibly.

---

## 4. DEVIATION FROM THE ORIGINAL PLAN DUE TO A GENUINE TECHNICAL CONSTRAINT

> **Constraint:** MoveNet (the pose backend at `inference/pose/movenet_pose.py`)
> does **not** expose knee (or hip) keypoints. The only body landmarks it returns
> that survive the `c < 0.2` confidence gate are the **nose, both shoulders, both
> wrists**, and a **derived torso centre**. Knees simply do not exist in its
> output, so they cannot be used to anchor the lower edge of the carrying zone.

**Impact on the original plan:** the brief defined the carrying zone as the band
*"between the shoulders and the knees"*. Because knees are unavailable, the lower
edge of that band could not be placed at the knees as written.

**Resolution adopted:** the band is anchored on the **shoulder line** at the top
and on the **bottom of the person's bounding box** at the bottom. This is the
closest available proxy for *"somewhere on/near the body below the arms"* given
the landmarks MoveNet actually provides. The redefinition is functionally a
strict superset of the intended region (it extends to the feet rather than
stopping at the knees) and is the recommended fix when the pose model cannot
supply knees.

This deviation is also recorded inline in:
- `config/events.yaml` (Phase 2 block comment), and
- `littering_event_detector.py` (`_shoulder_line()` docstring and the
  `pose_zone_vertical_above/below` field comments).

---

## 5. How to enable / verify Phase 2

In `config/events.yaml`:

```yaml
POSE_ASSOCIATION_ENABLED: true     # turn the feature on
POSE_SCORE_BLEND: 0.5             # 0.0 = legacy only, 1.0 = pose only
```

Then re-run the detector on the 5 test videos and compare confirmed events
against the Phase-1 baseline. The pose score is expected to be a tie-breaker in
multi-person frames; the `zone` cue is load-bearing (earlier ablation: removing
`zone` flipped the geometrically-preferred person ranking from correct to 0/8).

---

## 6. Status & next steps (awaiting your review)

- [x] Pixel gate locked to **12px** (not 15px).
- [x] Baseline commit `01e212e` confirmed as a clean pre-Phase-2 rollback point.
- [x] Phase 2 implemented; carrying zone redefined to **shoulders → bbox bottom**.
- [x] Deviation documented (this section 4).
- [ ] **Not yet committed** — Phase 2 changes remain in the working tree for review.
- [ ] Awaiting your decision: enable (`POSE_ASSOCIATION_ENABLED: true`), keep
      disabled, or reject (delete the Phase 2 block).

**Changed files (uncommitted):** `littering_event_detector.py`, `config/events.yaml`.
