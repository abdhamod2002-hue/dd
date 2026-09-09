"""P1-5 regression tests: operator-configured bin zones (static cameras).

MASTER_REPAIR_PLAN P1-5: an object placed inside a configured bin zone at
ground height must NOT trigger a ground-litter violation. With
``require_ground_confirmation`` enabled, resting inside the zone suppresses
ground evidence, so the candidate is rejected as BIN_ZONE_DEPOSIT — while the
exact same sequence outside the zone still confirms (no recall regression).
"""

from __future__ import annotations

from littering_event_detector import (
    BinZone,
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    EventState,
    LitteringEventDetector,
)

FRAME_W, FRAME_H = 1280, 720


def _person(pid: int, cx: float, cy: float, height: float = 160.0) -> DetectorPerson:
    half_w, half_h = 40.0, height / 2.0
    return DetectorPerson(
        track_id=pid,
        bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        confidence=0.9,
        keypoints=DetectorKeypoints(
            left_wrist=(cx + 10, cy + 30),
            right_wrist=(cx - 10, cy + 30),
            torso_center=(cx, cy - 20),
        ),
    )


def _bag(bid: int, cx: float, cy: float) -> DetectorBag:
    return DetectorBag(
        track_id=bid,
        bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
        confidence=0.7,
        class_name="trash_bag",
        source="yolo",
        yolo_confirmed=True,
    )


def _cfg(bin_zone: BinZone | None) -> EventDetectorConfig:
    return EventDetectorConfig(
        analysis_fps=10.0,
        min_carried_frames=3,
        min_stationary_frames=4,
        min_departed_frames=2,
        confirmation_grace_frames=2,
        smoothing_window=3,
        stationary_window_frames=3,
        max_pair_age_frames=12,
        min_event_confidence=0.65,
        require_ground_confirmation=True,
        bin_zones=(bin_zone,) if bin_zone else (),
    )


# Resting spot of the dropped bag in the scenario below (pixels):
# ground rest happens at (140, 420) in a 1280x720 frame -> (0.109, 0.583).
BIN_ZONE = BinZone(
    polygon=((0.05, 0.50), (0.20, 0.50), (0.20, 0.70), (0.05, 0.70)),
    camera_id="cam-01",
    name="dumpster",
)


def _run_litter_sequence(detector: LitteringEventDetector):
    """Carry -> release -> ground rest at (140, 420) -> person departs."""
    events = []
    t = 0.0
    size = (FRAME_W, FRAME_H)
    for _ in range(6):  # carry near lower body
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t, frame_size=size))
        t += 0.1
    for y in (280, 340, 400):  # release: bag separates downward
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t, frame_size=size))
        t += 0.1
    for _ in range(6):  # ground rest INSIDE the bin zone footprint
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t, frame_size=size))
        t += 0.1
    for x in (360, 520, 680, 840, 1000, 1160):  # departure
        events.extend(detector.update([_person(1, x, 180)], [_bag(10001, 140, 420)], t, frame_size=size))
        t += 0.1
    events.extend(detector.finalize())
    return events


def test_bag_in_bin_zone_is_not_ground_litter():
    detector = LitteringEventDetector(_cfg(BIN_ZONE), camera_id="cam-01")
    events = _run_litter_sequence(detector)
    confirmed = [e for e in events if e.confirmed]
    assert confirmed == [], "an object resting inside a configured bin zone must never confirm as ground littering"
    rejected = [e for e in events if not e.confirmed]
    assert rejected, "expected the bin-zone deposit to be recorded as a rejection"
    assert any(r.reason == "BIN_ZONE_DEPOSIT" for r in rejected), \
        f"expected BIN_ZONE_DEPOSIT, got {[r.reason for r in rejected]}"
    assert all(r.details["location_status"] == "BIN_ZONE" for r in rejected if r.reason == "BIN_ZONE_DEPOSIT")
    assert all(r.details["bin_zone_frames"] > 0 for r in rejected if r.reason == "BIN_ZONE_DEPOSIT")


def test_same_sequence_outside_zone_still_confirms():
    """Regression guard: the bin zone must not reduce true ground-litter
    recall — the identical sequence resting OUTSIDE the zone still confirms."""
    detector = LitteringEventDetector(_cfg(None), camera_id="cam-01")
    events = _run_litter_sequence(detector)
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1
    assert confirmed[0].state == EventState.VIOLATION_CONFIRMED
    assert confirmed[0].details["location_status"] == "GROUND_CONFIRMED"


def test_zone_of_other_camera_does_not_apply():
    """Per-camera scoping: a zone for cam-02 must not suppress cam-01 events."""
    other_camera_zone = BinZone(
        polygon=((0.05, 0.50), (0.20, 0.50), (0.20, 0.70), (0.05, 0.70)),
        camera_id="cam-02",
    )
    detector = LitteringEventDetector(_cfg(other_camera_zone), camera_id="cam-01")
    events = _run_litter_sequence(detector)
    assert any(e.confirmed for e in events), "another camera's bin zone must not affect this camera"


def test_bin_zone_point_in_polygon_and_bbox_coercion():
    zone = BinZone._coerce({"polygon": [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]})
    assert zone.contains_normalized(0.05, 0.05)
    assert not zone.contains_normalized(0.2, 0.05)
    assert not zone.contains_normalized(0.05, 0.2)
    boxed = BinZone._coerce({"bbox": [0.6, 0.5, 0.85, 0.85], "camera_id": 2})
    assert boxed.camera_id == "2"
    assert boxed.contains_normalized(0.7, 0.6)
    assert not boxed.contains_normalized(0.5, 0.6)
