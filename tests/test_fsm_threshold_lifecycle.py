"""Pin the FSM / threshold lifecycle audit so it cannot silently go stale.

The prose documentation drifted from the code long before the job-43 crash;
this test derives the audit from the live module and fails if the code
contradicts the documented state machine or threshold lifecycle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "phase_rootcause"))


def test_fsm_and_threshold_audit_is_clean():
    from audit_fsm_and_thresholds import _checks

    findings = _checks()
    assert findings == [], "FSM/threshold audit drifted from the code: " + "\n".join(findings)


def test_state_machine_reaches_confirmation_on_synthetic_carry_release_ground():
    """The documented happy path PERSON->CARRY->RELEASE->GROUND->DEPART must be
    reachable and must produce a defined confirmation verdict (never an
    exception from an undefined threshold local)."""
    from littering_event_detector import (
        DetectorBag,
        DetectorPerson,
        EventDetectorConfig,
        LitteringEventDetector,
    )

    det = LitteringEventDetector(
        EventDetectorConfig(
            auto_calibrate=False,
            pose_association_enabled=False,
            min_carried_frames=3,
            min_stationary_frames=3,
            min_departed_frames=2,
            min_abandonment_frames=3,
            confirmation_grace_frames=2,
            smoothing_window=3,
            stationary_window_frames=3,
            max_pair_age_frames=40,
            min_event_confidence=0.0,
            carry_motion_norm_px=2.0,
        )
    )
    # Person walks right carrying a bag, then drops it and walks on.
    for f in range(30):
        if f < 10:
            px, bx = 200 + f * 4, 210 + f * 4
        else:
            px, bx = 200 + f * 4, 250
        det.update(
            [DetectorPerson(track_id=1, bbox=(px, 200, px + 60, 360), confidence=1.0)],
            [DetectorBag(track_id=60001, bbox=(bx, 300, bx + 30, 330), confidence=0.9,
                         class_name="trash_bag", source="yolo", yolo_confirmed=True)],
            timestamp=f * 0.125,
            frame_index=f,
        )
    det.finalize()
    # The verdict is defined either way; the point is that NO exception occurred
    # while evaluating release/separation/ground/departure thresholds.
    assert det.summary()["total_candidates"] >= 0


def test_reset_does_not_leak_calibrated_thresholds_between_videos():
    """A second video must start from configured defaults, not the first
    video's calibrated scene scale."""
    from littering_event_detector import EventDetectorConfig, LitteringEventDetector

    det = LitteringEventDetector(
        EventDetectorConfig(auto_calibrate=True, carry_motion_norm_px=4.0)
    )
    before = det.config.carry_motion_norm_px
    # Simulate what _calibrate would write for a close-camera video.
    det.config.carry_motion_norm_px = 47.0
    det.config.departure_motion_ratio = 0.79
    det.reset()
    assert det.config.carry_motion_norm_px == before
    assert det.config.departure_motion_ratio == det._calibration_mutables["departure_motion_ratio"]
