"""
Comprehensive unit tests for Critical Product Corrections:
1. Physical size limits on handheld carrying (dumpster / large object rejection)
2. Bin vs Ground disposal distinction (no ground confirmation = BIN_DISPOSAL)
3. Stable actor UID binding across tracking churn
4. Evidence validation rejecting corrupt/blank crops
"""
import pytest
import numpy as np

from littering_event_detector import (
    LitteringEventDetector,
    EventDetectorConfig,
    EventState,
    RejectionReason,
    DetectorPerson,
    DetectorBag,
    _PairMemory,
    EventEvidence,
)
from inference.visualization.evidence_package import _is_valid_evidence_crop, _find_person


def test_physical_size_rejects_dumpsters_and_large_objects():
    """Verify that objects larger than 40% of person area or 65000px² cannot be latched as carried."""
    cfg = EventDetectorConfig()
    detector = LitteringEventDetector(config=cfg)

    # Person bbox: [100, 100, 300, 200] -> height=200, width=100, area=20000
    person = DetectorPerson(
        track_id=1,
        bbox=(100.0, 100.0, 200.0, 300.0),
        confidence=0.9,
    )

    # Dumpster bbox: [100, 100, 400, 500] -> height=400, width=300, area=120000
    dumpster = DetectorBag(
        track_id=10,
        bbox=(100.0, 100.0, 400.0, 500.0),
        confidence=0.9,
        class_name="garbage_bag",
    )

    # Evaluate pair
    result = detector._evaluate_pair(
        person=person,
        bag=dumpster,
        timestamp=1.0,
        frame_index=1,
    )
    # Must be None - dumpsters cannot be carried by a human!
    assert result is None, "Dumpster must be rejected as an uncarriable object"


def test_bin_vs_ground_distinction_no_ground_contact():
    """Verify that an object discarded without ground confirmation is classified as BIN_DISPOSAL."""
    cfg = EventDetectorConfig(require_ground_confirmation=True)
    detector = LitteringEventDetector(config=cfg)

    mem = _PairMemory(person_id=1, bag_id=10)
    mem.bag_seen_frames = 20
    mem.person_seen_frames = 20
    mem.carried_frames = 10
    mem.release_frame = 50
    mem.stationary_frames = 10
    mem.ground_evidence_frames = 0
    mem.bin_zone_frames = 0
    mem.confidence_sum = 18.0
    mem.confidence_count = 20

    evidence = EventEvidence(
        carry_score=0.9,
        release_score=0.9,
        stationary_score=0.9,
        departure_score=0.9,
        association_score=0.9,
        detection_score=0.9,
        confidence=0.95,
    )

    reason = detector._rejection_reason(mem, evidence)
    assert reason == RejectionReason.BIN_DISPOSAL.value


def test_evidence_validation_rejects_corrupted_or_blank_crops():
    """Verify that blank, black, or degenerate crops are flagged invalid."""
    # Blank/zero size
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    assert not _is_valid_evidence_crop(empty)

    # Too small
    tiny = np.ones((5, 5, 3), dtype=np.uint8) * 128
    assert not _is_valid_evidence_crop(tiny)

    # Pitch black
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert not _is_valid_evidence_crop(black)

    # Uniform color (no texture/std)
    uniform = np.ones((100, 100, 3), dtype=np.uint8) * 120
    assert not _is_valid_evidence_crop(uniform)

    # Real textured image
    np.random.seed(42)
    textured = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
    assert _is_valid_evidence_crop(textured)


def test_find_person_resolves_by_person_uid_across_track_churn():
    """Verify that _find_person resolves an actor by person_uid even when ByteTrack track_id churns."""
    p1 = {"track_id": 1, "person_uid": 5, "bbox": [10, 10, 50, 50]}
    p2 = {"track_id": 2, "person_uid": 7, "bbox": [60, 60, 90, 90]}
    p3_churned = {"track_id": 99, "person_uid": 5, "bbox": [12, 12, 52, 52]}

    # Record at release time has track_id=99, but person_uid=5
    record = {"persons": [p2, p3_churned]}

    # Search by original track_id=1 and person_uid=5
    found = _find_person(record, track_id=1, person_uid=5)
    assert found is not None
    assert found["track_id"] == 99
    assert found["person_uid"] == 5


def test_pipeline_event_exposes_event_timestamp_not_timestamp():
    """Regression: evidence_assembly dedup must use event_timestamp, not .timestamp."""
    from inference.pipeline import PipelineEvent

    ev = PipelineEvent(
        event_id="abc",
        camera_id="cam-1",
        person_track_id=1,
        object_track_id=2,
        object_type="red_waste_bag",
        confidence=0.9,
        event_timestamp=147.2,
        state_history=[],
    )
    assert not hasattr(ev, "timestamp")
    assert float(ev.event_timestamp) == 147.2


def test_no_physical_separation_rejects_clothing_latch():
    """Hip-attached color blob that never left the person box must not confirm."""
    cfg = EventDetectorConfig(require_ground_confirmation=True)
    detector = LitteringEventDetector(config=cfg)

    mem = _PairMemory(person_id=95, bag_id=60070)
    mem.bag_seen_frames = 20
    mem.person_seen_frames = 20
    mem.carried_frames = 10
    mem.release_frame = 4025
    mem.stationary_frames = 10
    mem.ground_evidence_frames = 5
    mem.bin_zone_frames = 0
    mem.ever_contained = True
    mem.max_post_release_norm_distance = 0.45  # person walked away from static blob
    mem.separated_frames = 1
    mem.max_bag_displacement_px = 3.0  # object barely moved
    mem.release_person_height = 400.0
    mem.confidence_sum = 18.0
    mem.confidence_count = 20

    evidence = EventEvidence(
        carry_score=0.9,
        release_score=0.9,
        stationary_score=0.9,
        departure_score=0.9,
        association_score=0.9,
        detection_score=0.9,
        confidence=0.95,
    )

    reason = detector._rejection_reason(mem, evidence)
    assert reason == RejectionReason.NO_PHYSICAL_SEPARATION.value


def test_physical_separation_allows_ground_litter_confirm():
    """Real discard with post-release separation clears the separation gate."""
    cfg = EventDetectorConfig(require_ground_confirmation=True)
    detector = LitteringEventDetector(config=cfg)

    mem = _PairMemory(person_id=303, bag_id=10310)
    mem.bag_seen_frames = 20
    mem.person_seen_frames = 20
    mem.carried_frames = 8
    mem.release_frame = 6420
    mem.stationary_frames = 10
    mem.ground_evidence_frames = 6
    mem.bin_zone_frames = 0
    mem.ever_contained = True
    mem.max_post_release_norm_distance = 0.35
    mem.separated_frames = 4
    mem.max_bag_displacement_px = 80.0
    mem.release_person_height = 400.0
    mem.confidence_sum = 18.0
    mem.confidence_count = 20

    evidence = EventEvidence(
        carry_score=0.9,
        release_score=0.9,
        stationary_score=0.9,
        departure_score=0.9,
        association_score=0.9,
        detection_score=0.9,
        confidence=0.95,
    )

    reason = detector._rejection_reason(mem, evidence)
    assert reason is None


def test_fast_drop_confirm_with_separation_below_full_carry_count():
    """Quick throw may confirm with slightly fewer than min_carried_frames."""
    cfg = EventDetectorConfig(
        require_ground_confirmation=True,
        min_carried_frames=6,
    )
    detector = LitteringEventDetector(config=cfg)

    mem = _PairMemory(person_id=1, bag_id=2)
    mem.bag_seen_frames = 20
    mem.person_seen_frames = 20
    mem.carried_frames = 4  # max(2, 6-2) = 4 fast-drop floor
    mem.release_frame = 100
    mem.stationary_frames = 10
    mem.ground_evidence_frames = 5
    mem.ever_contained = True
    mem.max_post_release_norm_distance = 0.40
    mem.separated_frames = 3
    mem.max_bag_displacement_px = 90.0
    mem.release_person_height = 400.0
    mem.confidence_sum = 18.0
    mem.confidence_count = 20

    evidence = EventEvidence(
        carry_score=0.7,
        release_score=1.0,
        stationary_score=1.0,
        departure_score=1.0,
        association_score=0.9,
        detection_score=0.9,
        confidence=0.9,
    )

    reason = detector._rejection_reason(mem, evidence)
    assert reason is None


def test_adaptive_flush_suppresses_unseparated_relaxed_confirm():
    """Relaxed tiers must not emit clothing-latch confirmations."""
    from adaptive_tuner import AdaptiveEventDetector
    from littering_event_detector import LitteringEvent, EventState

    det = AdaptiveEventDetector(EventDetectorConfig(), enable_learning=False)
    ev = LitteringEvent(
        event_id="latch01",
        person_track_id=95,
        bag_track_id=60070,
        state=EventState.VIOLATION_CONFIRMED,
        confirmed=True,
        reason="VIOLATION_CONFIRMED",
        confidence=0.88,
        evidence={},
        frames={"carry_start": 3945, "release": 4025, "ground": 4041, "departure": 4081, "confirmed": 4073},
        timestamps={"confirmed": 100.0, "departure": 99.5, "release": 98.0, "carry_start": 95.0},
        bag_class="red_waste_bag",
        fallback_used=True,
        yolo_reconfirmed=False,
        other_person_closer=False,
        details={
            "max_post_release_norm_distance": 0.45,
            "separated_frames": 1,
            "ever_contained": True,
            "max_bag_displacement_px": 2.0,
        },
    )
    assert not AdaptiveEventDetector._event_has_physical_separation(ev)
    det._pending_relaxed.append((1, ev, None))
    out = det._flush_pending(None, final=True)
    assert out == []
    assert det.confirmed_events == []


def test_end_of_stream_walkaway_confirms_instead_of_person_did_not_depart():
    """Short clip: release+ground then person keeps walking — finalize confirms."""
    cfg = EventDetectorConfig(
        require_ground_confirmation=True,
        min_carried_frames=4,
        min_stationary_frames=8,
        min_abandonment_frames=8,
        min_departed_frames=2,
        min_event_confidence=0.5,
    )
    detector = LitteringEventDetector(config=cfg)

    mem = _PairMemory(person_id=1, bag_id=10)
    mem.bag_seen_frames = 20
    mem.person_seen_frames = 20
    mem.carried_frames = 8
    mem.release_frame = 209
    mem.ground_frame = 249
    mem.ground_evidence_frames = 3
    mem.stationary_frames = 2  # clip cut before full stationary window
    mem.departed_frames = 1
    mem.max_departure_ratio = 0.6
    mem.departure_frame = None
    mem.ever_contained = True
    mem.max_post_release_norm_distance = 0.35
    mem.separated_frames = 3
    mem.max_bag_displacement_px = 90.0
    mem.release_person_height = 400.0
    mem.release_person_centroid = (300.0, 900.0)
    mem.last_frame = 272
    mem.last_timestamp = 4.5
    mem.confidence_sum = 18.0
    mem.confidence_count = 20
    mem.state = EventState.BAG_ON_GROUND

    # Precondition: without end-of-stream credit this would be PERSON_DID_NOT_DEPART
    evidence = EventEvidence(
        carry_score=0.9,
        release_score=1.0,
        stationary_score=0.3,
        departure_score=0.6,
        association_score=0.9,
        detection_score=0.9,
        confidence=0.85,
    )
    assert (
        detector._rejection_reason(mem, evidence)
        == RejectionReason.PERSON_DID_NOT_DEPART.value
    )

    out = detector._finalize_pair(mem)
    assert len(out) == 1
    assert out[0].confirmed is True
    assert out[0].reason == "VIOLATION_CONFIRMED"
    assert mem.end_of_stream_walkaway is True
    assert mem.departure_frame == 272
