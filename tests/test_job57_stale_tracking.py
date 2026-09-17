"""Job-57 class regressions: stale tracking must not manufacture violations.

Covers the general failure class — tracking dropout / stale object state
→ false release/ground → premature irreversible confirmation — with no
video-specific values (no frames, timestamps, or track IDs from M.MOV).
"""
from __future__ import annotations

from littering_event_detector import (
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    LitteringEventDetector,
)


def _person(pid: int, cx: float, cy: float, height: float = 160.0, conf: float = 0.9) -> DetectorPerson:
    half_w = 40.0
    half_h = height / 2.0
    return DetectorPerson(
        track_id=pid,
        bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        confidence=conf,
        keypoints=DetectorKeypoints(
            left_wrist=(cx + 10, cy + 30),
            right_wrist=(cx - 10, cy + 30),
            torso_center=(cx, cy - 20),
        ),
    )


def _bag(bid: int, cx: float, cy: float, conf: float = 0.5) -> DetectorBag:
    return DetectorBag(
        track_id=bid,
        bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
        confidence=conf,
        class_name="trash_bag",
        source="yolo",
        yolo_confirmed=True,
    )


def _fast_config(**over) -> EventDetectorConfig:
    base = dict(
        analysis_fps=10.0,
        min_carried_frames=3,
        min_stationary_frames=4,
        min_departed_frames=2,
        confirmation_grace_frames=2,
        smoothing_window=3,
        stationary_window_frames=3,
        max_pair_age_frames=8,
        min_event_confidence=0.65,
        max_track_gap_frames=3,
    )
    base.update(over)
    return EventDetectorConfig(**base)


def _carry(detector, ticks=6, bag_y=220.0):
    t = 0.0
    for i in range(ticks):
        detector.update([_person(1, 140, 180)], [_bag(10001, 140, bag_y)], t)
        t += 0.1
    return t


def _confirmed(events):
    return [e for e in events if e.confirmed]


def test_job57_A_object_disappears_no_stale_ground_confirm():
    """A: object disappears mid-arc → stale state must NOT confirm."""
    detector = LitteringEventDetector(_fast_config())
    t = _carry(detector)
    events = []
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(12):
        events.extend(detector.update([_person(1, 140, 180)], [], t))
        t += 0.1
    events.extend(detector.finalize())
    assert _confirmed(events) == [], "missing-bag drift must never confirm a violation"

def test_job57_B_different_object_no_uid_transfer():
    """B: old object gone, different object nearby → no stale adoption."""
    detector = LitteringEventDetector(_fast_config())
    t = _carry(detector)
    events = []
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [], t))
        t += 0.1
    for _ in range(8):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(77777, 700, 420)], t))
        t += 0.1
    events.extend(detector.finalize())
    assert _confirmed(events) == [], "rebound adopter must not confirm on stale evidence"


def test_job57_C_stale_separation_no_release():
    """C: a stale AIDM separation latch must not conjure a release."""
    detector = LitteringEventDetector(_fast_config())
    t = _carry(detector)
    events = []
    for y in (280, 340):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    key = next(iter(detector._pairs))
    mem = detector._pairs[key]
    mem.aidm_separated = True
    mem.ever_aidm_attached = True
    for _ in range(8):
        events.extend(detector.update([_person(1, 140, 180)], [], t))
        t += 0.1
    for _ in range(6):
        events.extend(
            detector.update([_person(1, 140, 180)], [_bag(55555, 700, 100)], t)
        )
        t += 0.1
    events.extend(detector.finalize())
    assert _confirmed(events) == [], "stale separation must not trigger release"


def test_job57_D_tracking_uncertainty_no_irreversible_violation():
    """D: jittery never-separated carry must stay a candidate, never confirm."""
    detector = LitteringEventDetector(_fast_config())
    t = 0.0
    events = []
    for _ in range(30):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 200)], t))
        t += 0.1
    events.extend(detector.finalize())
    assert _confirmed(events) == []


def test_job57_E_temporary_ground_plus_recovery_no_violation():
    """E: temporary ground contact followed by recovery → NO VIOLATION.

    Uses a longer abandonment window so the recovery (regrab) happens BEFORE
    the abandonment threshold — exercising the regrab protection path, which
    is the product-correct behavior for 'picked it back up'.
    """
    detector = LitteringEventDetector(_fast_config(min_abandonment_frames=12))
    events = []
    t = _carry(detector)
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1
    # Recovery: the bag is picked back up (returns to carried zone).
    for _ in range(8):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    events.extend(detector.finalize())
    assert _confirmed(events) == [], "temporary grounding + recovery must not confirm"


def test_job57_F_bin_deposit_no_violation():
    """F: a bin deposit with operator zone configured must not confirm."""
    from littering_event_detector import BinZone

    zone = BinZone(
        polygon=((0.55, 0.55), (0.95, 0.55), (0.95, 0.95), (0.55, 0.95)),
        camera_id=None,
        name="bin",
    )
    cfg = _fast_config()
    cfg.bin_zones = (zone,)
    detector = LitteringEventDetector(cfg)
    events = []
    t = 0.0
    for _ in range(6):
        events.extend(
            detector.update(
                [_person(1, 140, 180)], [_bag(10001, 140, 220)], t,
                frame_size=(640, 480),
            )
        )
        t += 0.1
    for _ in range(10):
        events.extend(
            detector.update(
                [_person(1, 140, 180)], [_bag(10001, 448, 336)], t,
                frame_size=(640, 480),
            )
        )
        t += 0.1
    events.extend(detector.finalize())
    assert _confirmed(events) == [], "bin-zone deposit must not confirm"


def test_job57_G_true_ground_abandonment_confirms():
    """G: genuine carry → release → ground → abandon MUST still confirm."""
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1
    for x in (360, 520, 680, 840):
        events.extend(detector.update([_person(1, x, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1
    events.extend(detector.finalize())
    confirmed = _confirmed(events)
    assert len(confirmed) == 1, "true ground abandonment must still confirm"

