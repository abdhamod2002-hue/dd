"""Tests for the adaptive self-tuning layer (adaptive_tuner.py)."""

from __future__ import annotations

import os

from adaptive_tuner import (
    LEARNING_STEPS,
    AdaptiveEventDetector,
    LearningStore,
    TIER_SPECS,
    build_tier_configs,
)
from littering_event_detector import (
    TUNABLE_THRESHOLD_KEYS,
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    EventState,
    load_event_config,
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


def test_tier_specs_are_bounded_relaxations():
    """min_*/ratio keys may only decrease, max_*/window keys may only
    increase, tier over tier — and every tier value must stay within the
    hard clamps."""
    decreases = {
        "min_carried_frames", "min_stationary_frames", "min_departed_frames",
        "min_abandonment_frames", "min_event_confidence",
        "release_distance_ratio", "release_distance_floor",
        "departure_motion_ratio", "departure_distance_ratio",
        "smoothing_window", "feet_release_frames",
    }
    increases = {
        "release_window_frames", "stationary_grace_frames",
        "confirmation_grace_frames", "max_pair_age_frames",
        "max_fallback_tracker_gap_frames", "stationary_max_pixel_step",
    }
    for prev, nxt in zip(TIER_SPECS, TIER_SPECS[1:]):
        for key, value in nxt.items():
            if key in decreases:
                assert prev.get(key, value) >= value, key
            elif key in increases:
                assert prev.get(key, value) <= value, key


def test_learning_store_advances_and_saturates(tmp_path):
    store = LearningStore(path=str(tmp_path / "learning.json"))
    base = {"total_candidates": 3, "confirmed_violations": 0,
            "rejected_candidates": 3,
            "rejection_reason_counts": {"NOT_ENOUGH_CARRIED_FRAMES": 2}}
    for i in range(6):  # more videos than steps -> must saturate
        store.record(f"v{i}.MOV", base)
    assert store.videos_analyzed == 6
    assert store.step_index()["NOT_ENOUGH_CARRIED_FRAMES"] == len(
        LEARNING_STEPS["NOT_ENOUGH_CARRIED_FRAMES"]
    )
    ov = store.learned_overrides()
    assert ov["min_carried_frames"] == 3
    assert ov["smoothing_window"] == 3
    # reloaded store keeps the learned state
    store2 = LearningStore(path=str(tmp_path / "learning.json"))
    assert store2.learned_overrides()["min_carried_frames"] == 3


def test_tier_configs_deduplicate_and_apply_learning(tmp_path):
    store = LearningStore(path=str(tmp_path / "learning.json"))
    base = {"total_candidates": 1, "confirmed_violations": 0,
            "rejected_candidates": 1,
            "rejection_reason_counts": {"NO_RELEASE_TRANSITION": 1}}
    store.record("a.MOV", base)
    cfgs = build_tier_configs(load_event_config(), store.learned_overrides())
    assert len(cfgs) >= 2
    # tier0 carries the learned overrides, later tiers are at least as relaxed
    assert cfgs[0].release_distance_ratio < load_event_config().release_distance_ratio


def _run_carried_release_ground(detector, carry_ticks=4, ground_ticks=14):
    """carry -> release -> ground; NO departure. Returns confirmed events."""
    events = []
    t = 0.0
    for _ in range(carry_ticks):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(ground_ticks):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1
    events.extend(detector.finalize())
    return [e for e in events if e.confirmed]


def test_adaptive_wrapper_rescues_threshold_brittle_sequence(tmp_path):
    """A sequence the production config rejects (too few carried / stationary
    frames) IS confirmed by the tier ladder — exactly one deduplicated event,
    tagged with the tier that confirmed it."""
    store = LearningStore(path=str(tmp_path / "learning.json"))
    adaptive = AdaptiveEventDetector(load_event_config(), store=store)
    adaptive.learning_video = "brittle.MOV"

    confirmed = _run_carried_release_ground(adaptive, carry_ticks=4)
    assert len(confirmed) == 1
    assert confirmed[0].state == EventState.VIOLATION_CONFIRMED
    assert confirmed[0].evidence.get("adaptive_tier", 0) >= 1

    # tier 0 (production config) must still reject it — learning input.
    strict = AdaptiveEventDetector(load_event_config(), store=store, max_tiers=0)
    assert len(_run_carried_release_ground(strict, carry_ticks=4)) == 0

    # learning recorded with tier attribution
    rec = store._doc["history"]["brittle.MOV"]
    assert rec["tiers_confirmed"]["tier0"] == 0
    assert rec["tiers_confirmed"]["tier2"] == 1


def test_adaptive_wrapper_primary_confirmation_not_duplicated(tmp_path):
    """When tier 0 confirms, the wrapper emits exactly ONE event per physical
    confirmation (no relaxed duplicates)."""
    store = LearningStore(path=str(tmp_path / "learning.json"))
    adaptive = AdaptiveEventDetector(load_event_config(), store=store)
    events = []
    t = 0.0
    for _ in range(8):
        events.extend(adaptive.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    for y in (280, 340, 400):
        events.extend(adaptive.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(10):
        events.extend(adaptive.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1
    for x in (360, 520, 680, 840):
        events.extend(adaptive.update([_person(1, x, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1
    events.extend(adaptive.finalize())
    confirmed = [e for e in events if e.confirmed]
    keys = [(e.person_track_id, round(float(e.timestamps.get("confirmed") or 0), 1)) for e in confirmed]
    assert len(keys) == len(set(keys))  # no duplicate confirmations
    assert any(e.evidence.get("adaptive_tier", 0) == 0 for e in confirmed)


def test_wrapper_learning_disabled_leaves_no_store(tmp_path):
    detector = AdaptiveEventDetector(
        EventDetectorConfig(), store=None, enable_learning=False
    )
    assert detector.store is None
    detector.learning_video = "x.MOV"
    detector.finalize()  # must not create any store file
    assert not os.path.exists(str(tmp_path / "learning.json"))


# --------------------------------------------------------------------------- #
# MASTER_REPAIR_PLAN P0-1b regression: self-matching deduplication
# --------------------------------------------------------------------------- #
def _confirmed_littering_event(event_id: str, person_track_id: int = 7,
                               confirmed_ts: float = 10.0):
    from littering_event_detector import LitteringEvent

    def _make(eid: str) -> LitteringEvent:
        return LitteringEvent(
            event_id=eid,
            person_track_id=person_track_id,
            bag_track_id=99,
            state=EventState.VIOLATION_CONFIRMED,
            confirmed=True,
            reason="test",
            confidence=0.9,
            evidence={},
            frames={},
            timestamps={"confirmed": confirmed_ts},
            bag_class="trash_bag",
            fallback_used=False,
            yolo_reconfirmed=True,
            other_person_closer=False,
        )

    return _make, _make(event_id)


def test_is_duplicate_never_matches_own_acceptance():
    """An event must never be treated as a duplicate of ITS OWN recorded
    acceptance (P0-1b). A different event from the same person inside the
    dedup window must still be treated as a duplicate."""
    adaptive = AdaptiveEventDetector(
        load_event_config(), store=None, enable_learning=False
    )
    _make, ev = _confirmed_littering_event("self123")
    adaptive._accept(ev)
    ts = adaptive._event_ts(ev)
    assert ts is not None
    # own acceptance must not suppress the event itself
    assert adaptive._is_duplicate(ev, ts) is False
    # a DIFFERENT event (same person, same window) is still deduplicated
    other = _make("other456")
    assert adaptive._is_duplicate(other, ts) is True
    # and an event from another person is never suppressed
    stranger = _make("stranger789")
    stranger.person_track_id = 8
    assert adaptive._is_duplicate(stranger, ts) is False


def test_relaxed_tier_event_survives_own_confirmed_events_read(tmp_path):
    """P0-1b end-to-end regression: a tier-1/2-confirmed event that was
    _accept()-ed must survive its very next `confirmed_events` property read
    (which re-runs _dedup_confirmed over the tier detectors' confirmed lists
    and previously found distance=0 against the event's own acceptance)."""
    store = LearningStore(path=str(tmp_path / "learning.json"))
    adaptive = AdaptiveEventDetector(load_event_config(), store=store)
    adaptive.learning_video = "selfmatch.MOV"

    emitted = _run_carried_release_ground(adaptive, carry_ticks=4)
    assert len(emitted) == 1
    ev = emitted[0]
    assert ev.confirmed
    assert ev.evidence.get("adaptive_tier", 0) >= 1  # relaxed-tier confirmed

    # The next property read re-runs _dedup_confirmed() AFTER _accept().
    again = adaptive.confirmed_events
    ids = [e.event_id for e in again]
    assert ev.event_id in ids, "accepted relaxed-tier event was discarded as a duplicate of itself"
    assert ids.count(ev.event_id) == 1, "accepted relaxed-tier event was duplicated"


def test_tunable_threshold_keys_cover_all_adaptive_overrides():
    """P1-8 drift guard: every config key the adaptive layer can override
    (TIER_SPECS relaxations + LEARNING_STEPS learned steps) must be part of
    TUNABLE_THRESHOLD_KEYS, which is what events disclose in
    details['active_thresholds']. A new override key without disclosure
    update would silently hide the effective threshold from operators."""
    override_keys = set()
    for spec in TIER_SPECS:
        override_keys.update(spec.keys())
    for steps in LEARNING_STEPS.values():
        for step in steps:
            override_keys.update(step.keys())
    missing = override_keys - set(TUNABLE_THRESHOLD_KEYS)
    assert not missing, (
        f"adaptive override keys missing from TUNABLE_THRESHOLD_KEYS "
        f"(details['active_thresholds'] disclosure): {sorted(missing)}"
    )

