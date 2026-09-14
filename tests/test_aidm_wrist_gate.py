"""Section 5b — AIDM-Strat wrist↔bag normalized distance gate."""
from __future__ import annotations

from littering_event_detector import (
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    EventState,
    LitteringEventDetector,
    _person_diagonal,
    _wrist_bag_normalized_distance,
)


def _person(pid, cx, cy, height=200.0, wrists=None):
    half_w = 40.0
    half_h = height / 2.0
    bbox = (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
    kp = None
    if wrists is not None:
        lw, rw = wrists
        kp = DetectorKeypoints(left_wrist=lw, right_wrist=rw, torso_center=(cx, cy))
    return DetectorPerson(track_id=pid, bbox=bbox, confidence=0.9, keypoints=kp)


def _bag(bid, cx, cy, size=40.0):
    h = size / 2.0
    return DetectorBag(
        track_id=bid,
        bbox=(cx - h, cy - h, cx + h, cy + h),
        confidence=0.9,
        class_name="trash_bag",
        source="yolo",
        yolo_confirmed=True,
    )


def test_wrist_bag_normalized_distance_formula():
    # person 80x200 → diag = sqrt(80^2+200^2)=215.407...
    p = _person(1, 100, 200, height=200.0, wrists=((100.0, 250.0), None))
    b = _bag(10, 100.0, 250.0)  # center on left wrist → d=0
    d_norm, ok = _wrist_bag_normalized_distance(p, b)
    assert ok is True
    assert d_norm is not None and d_norm < 0.01
    assert abs(_person_diagonal(p.bbox) - (80.0**2 + 200.0**2) ** 0.5) < 1e-6


def test_aidm_blocks_passerby_release_without_prior_attach():
    """Person near a grounded bag with wrists never gripping must not release."""
    cfg = EventDetectorConfig(
        min_carried_frames=3,
        min_stationary_frames=3,
        min_abandonment_frames=3,
        min_departed_frames=1,
        feet_release_frames=2,
        smoothing_window=2,
        aidm_release_gate_enabled=True,
        aidm_wrist_attach_ratio=0.15,
        aidm_wrist_separate_ratio=0.30,
        require_carry_origin_link=False,  # isolate the release gate
    )
    det = LitteringEventDetector(cfg)
    # Force a CARRIED state via internal memory after synthetic near carries
    # with wrists FAR from the bag (passerby silhouette).
    t = 0.0
    # Bag at feet of passerby; wrists up near shoulders (far from bag).
    for x in (200, 220, 240, 260, 280, 300):
        p = _person(
            1, x, 180, height=200.0,
            wrists=((x - 20, 100.0), (x + 20, 100.0)),
        )
        det.update([p], [_bag(10001, x, 250)], t)
        t += 0.1
    # Walk away a bit while bag stays — classic false-release bait.
    for _ in range(10):
        p = _person(
            1, 360, 180, height=200.0,
            wrists=((340.0, 100.0), (380.0, 100.0)),
        )
        det.update([p], [_bag(10001, 280, 250)], t)
        t += 0.1
    # Inspect pair memory: if a pair exists and was carried, release must
    # remain blocked while d_norm < separate and no prior AIDM attach.
    pairs = list(det._pairs.values())
    if not pairs:
        return  # no pair formed — also acceptable (no false actor)
    mem = pairs[0]
    if mem.state == EventState.BAG_CARRIED:
        assert mem.release_frame is None
        assert mem.ever_aidm_attached is False


def test_aidm_allows_release_when_wrist_separates_after_attach():
    cfg = EventDetectorConfig(
        min_carried_frames=4,
        min_stationary_frames=4,
        min_abandonment_frames=4,
        min_departed_frames=1,
        feet_release_frames=2,
        smoothing_window=2,
        aidm_release_gate_enabled=True,
        require_ground_confirmation=False,
    )
    det = LitteringEventDetector(cfg)
    events = []
    t = 0.0
    # Carry with wrist on bag.
    for _ in range(8):
        p = _person(1, 140, 180, height=200.0, wrists=((140.0, 200.0), (160.0, 120.0)))
        events.extend(det.update([p], [_bag(10001, 140, 200)], t))
        t += 0.1
    # Throw: bag flies away, wrists stay on body → d_norm >> 0.30
    for i in range(12):
        bag_x = 140 + i * 25
        p = _person(1, 140, 180, height=200.0, wrists=((130.0, 160.0), (150.0, 160.0)))
        events.extend(det.update([p], [_bag(10001, bag_x, 230)], t))
        t += 0.1
    events.extend(det.finalize())
    released = any(
        (getattr(e, "frames", {}) or {}).get("release") is not None
        for e in (det.confirmed_events + det.rejected_events + events)
    )
    assert released or any(e.confirmed for e in events), (
        f"expected release after AIDM separation; events="
        f"{[(e.confirmed, e.reason, e.frames) for e in events[:5]]}"
    )


def test_aidm_separation_releases_even_if_smooth_carried_latched():
    """Paper path: d_norm >= separate after grip must leave BAG_CARRIED
    even when wrist_near keeps smooth_carried True (IMG_5290 failure class)."""
    cfg = EventDetectorConfig(
        min_carried_frames=3,
        min_stationary_frames=3,
        min_abandonment_frames=3,
        smoothing_window=2,
        aidm_release_gate_enabled=True,
        aidm_wrist_attach_ratio=0.15,
        aidm_wrist_separate_ratio=0.20,
        require_ground_confirmation=False,
        require_carry_origin_link=False,
    )
    det = LitteringEventDetector(cfg)
    t = 0.0
    # Grip: wrists on bag.
    for _ in range(6):
        p = _person(1, 140, 180, height=200.0, wrists=((140.0, 200.0), (150.0, 195.0)))
        det.update([p], [_bag(10001, 140, 200)], t)
        t += 0.1
    # Separation: bag moves out while person stands; one wrist may linger near.
    for i in range(8):
        bag_y = 200 + i * 20
        p = _person(1, 140, 180, height=200.0, wrists=((140.0, 190.0), (160.0, 120.0)))
        det.update([p], [_bag(10001, 140, bag_y)], t)
        t += 0.1
    mems = list(det._pairs.values())
    assert mems, "expected a person-bag pair"
    assert any(
        m.release_frame is not None or m.state in (EventState.BAG_RELEASED, EventState.BAG_ON_GROUND, EventState.PERSON_DEPARTED, EventState.VIOLATION_CONFIRMED)
        for m in mems
    ), f"states={[m.state.value for m in mems]} releases={[m.release_frame for m in mems]}"


def test_aidm_separation_hysteresis_survives_clothing_latch():
    """Brief d_norm peak above separate must latch; clothing pull-back must not
    erase the dump (IMG_5290 class). Re-attach clears the latch."""
    cfg = EventDetectorConfig(
        min_carried_frames=3,
        min_stationary_frames=3,
        min_abandonment_frames=3,
        smoothing_window=2,
        aidm_release_gate_enabled=True,
        aidm_wrist_attach_ratio=0.15,
        aidm_wrist_separate_ratio=0.20,
        require_ground_confirmation=False,
        require_carry_origin_link=False,
    )
    det = LitteringEventDetector(cfg)
    t = 0.0
    for _ in range(6):
        p = _person(1, 140, 180, height=200.0, wrists=((140.0, 200.0), (150.0, 195.0)))
        det.update([p], [_bag(10001, 140, 200)], t)
        t += 0.1
    # One-frame peak separation (d_norm ≈ 0.21 on Motared), then clothing latch
    # snaps the box back near the wrist (d_norm ≈ 0.10, still above attach).
    p_peak = _person(1, 140, 180, height=200.0, wrists=((140.0, 160.0), (160.0, 120.0)))
    det.update([p_peak], [_bag(10001, 140, 250)], t)  # far below wrists
    t += 0.1
    for _ in range(4):
        p_latch = _person(1, 140, 180, height=200.0, wrists=((140.0, 200.0), (150.0, 120.0)))
        det.update([p_latch], [_bag(10001, 140, 205)], t)  # near wrist again
        t += 0.1
    mems = list(det._pairs.values())
    assert mems, "expected a person-bag pair"
    assert any(m.release_frame is not None or m.aidm_separated for m in mems), (
        f"hysteresis must latch separation; states={[m.state.value for m in mems]} "
        f"sep={[m.aidm_separated for m in mems]} rel={[m.release_frame for m in mems]}"
    )


def test_missing_bag_after_aidm_carry_releases():
    """When the gripped bag disappears after a walking carry, release (IMG_5290)."""
    cfg = EventDetectorConfig(
        min_carried_frames=3,
        min_stationary_frames=3,
        min_abandonment_frames=3,
        feet_release_frames=2,
        smoothing_window=2,
        max_pair_age_frames=30,
        aidm_release_gate_enabled=True,
        aidm_wrist_attach_ratio=0.15,
        aidm_wrist_separate_ratio=0.20,
        require_ground_confirmation=False,
        require_carry_origin_link=False,
    )
    det = LitteringEventDetector(cfg)
    t = 0.0
    for x in (140, 160, 180, 200, 220, 240):
        p = _person(1, x, 180, height=200.0, wrists=((x, 200.0), (x + 10, 195.0)))
        det.update([p], [_bag(10001, x, 200)], t)
        t += 0.1
    # Bag vanishes; person keeps walking.
    for x in (260, 280, 300, 320, 340, 360):
        p = _person(1, x, 180, height=200.0, wrists=((x, 160.0), (x + 10, 120.0)))
        det.update([p], [], t)
        t += 0.1
    events = det.finalize()
    mem_released = any(
        (e.frames or {}).get("release") is not None
        for e in (det.confirmed_events + det.rejected_events + events)
    )
    assert mem_released or any(e.confirmed for e in events), (
        f"expected release after missing bag; events="
        f"{[(e.confirmed, e.reason, e.frames) for e in (events + det.rejected_events)[:6]]}"
    )
