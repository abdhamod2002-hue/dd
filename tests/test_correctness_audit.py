"""P1-4 known-limitation tests (MASTER_REPAIR_PLAN Section 21).

The production FSM enforces ONE bag per person: a person carrying two
objects simultaneously only ever gets a pair memory for the higher-scoring
(primary) object; the second object is demoted before pair creation and is
therefore structurally invisible to the event engine.

These tests PIN that documented behavior:
  * exactly one pair memory exists for a two-object actor;
  * releasing the secondary object emits no event and does not disturb the
    primary pair (the P1-4 acceptance scenario "P1 carries W1 + W2, releases
    W2" is NOT supported: W2 can never become an event candidate).

If a future change adds true two-object support, these tests are expected
to be deliberately rewritten — not silently deleted.
"""

from __future__ import annotations

from littering_event_detector import (
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    EventState,
    LitteringEventDetector,
    _PairInfo,
    _PairMemory,
)


def _cfg() -> EventDetectorConfig:
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
    )


def _person(pid, cx, cy, height=160.0, conf=0.9, with_wrists=True):
    half_w = 40.0
    half_h = height / 2.0
    kp = None
    if with_wrists:
        # Wrists near torso center so a mid-body bag can latch as carried
        # (stationary bags without wrist evidence cannot zone-carry).
        kp = DetectorKeypoints(
            left_wrist=(cx - 10.0, cy),
            right_wrist=(cx + 10.0, cy),
            torso_center=(cx, cy),
            left_shoulder=(cx - 20.0, cy - 40.0),
            right_shoulder=(cx + 20.0, cy - 40.0),
        )
    return DetectorPerson(
        track_id=pid,
        bbox=(cx - half_w, cy - half_h, cx + half_w, cy + half_h),
        confidence=conf,
        keypoints=kp,
    )


def _bag(tid, cx, cy, conf=0.9):
    return DetectorBag(
        track_id=tid,
        bbox=(cx - 25, cy - 25, cx + 25, cy + 25),
        confidence=conf,
        class_name="plastic bottle",
        source="yolo",
        yolo_confirmed=True,
    )


FRAME_SIZE = (1280, 720)


def _ticks(det, persons, bags, n, t0=0.0, dt=0.1):
    events = []
    t = t0
    for _ in range(n):
        events.extend(det.update(list(persons), list(bags), t, frame_size=FRAME_SIZE))
        t += dt
    return events, t


def test_two_object_actor_has_exactly_one_pair_memory():
    """P1-4 invariant: one person + two simultaneous bags => exactly ONE pair.

    The higher-scoring (closer, wrist-adjacent) bag wins the primary
    association; the second is demoted before pair creation.
    """
    det = LitteringEventDetector(_cfg())
    person = _person(1, 640, 180)
    bag_a = _bag(30001, 640, 180)   # mid-torso carry band (above ground plane)
    bag_b = _bag(30002, 720, 185)   # also near the person, weaker
    _ticks(det, [person], [bag_a, bag_b], 6)[0]

    pairs_for_person = [k for k in det._pairs if k[0] == 1]
    assert len(pairs_for_person) == 1, (
        "one-bag-per-person invariant violated: person 1 has pair memories "
        f"{pairs_for_person}"
    )
    mem = det._pairs[pairs_for_person[0]]
    assert mem.bag_id == 30001, "primary (wrist-adjacent) bag must win the association"
    assert mem.state == EventState.BAG_CARRIED


def test_secondary_object_release_is_invisible_to_event_engine():
    """P1-4 scenario: actor carries W1(primary) + W2, then releases W2.

    Expected (documented limitation): W2 produces NO event candidate, and
    the W1 pair stays undisturbed in BAG_CARRIED. The FSM never sees W2's
    release as a littering arc for this actor.
    """
    det = LitteringEventDetector(_cfg())
    person = _person(1, 640, 180)
    bag_a = _bag(30001, 640, 180)
    bag_b = _bag(30002, 720, 185)

    # Phase 1 — carry both objects.
    _, t = _ticks(det, [person], [bag_a, bag_b], 6)
    pairs_before = [k for k in det._pairs if k[0] == 1]
    assert len(pairs_before) == 1
    assert det._pairs[pairs_before[0]].state == EventState.BAG_CARRIED

    # Phase 2 — W2 is put down on the ground while W1 stays in hand.
    # W2 rests stationary well below the person (ground context) and far
    # enough that it is no longer a carry association for this actor.
    bag_b_ground = _bag(30002, 720, 420)
    emitted, _ = _ticks(det, [person], [bag_a, bag_b_ground], 14, t0=t)

    assert emitted == [], (
        "the demoted secondary object's release must be invisible to the "
        f"FSM, but events were emitted: {[ (e.person_track_id, e.reason) for e in emitted ]}"
    )
    # Still exactly one pair, still carrying the primary object.
    pairs_after = [k for k in det._pairs if k[0] == 1]
    assert len(pairs_after) == 1
    mem = det._pairs[pairs_after[0]]
    assert mem.bag_id == 30001
    assert mem.state == EventState.BAG_CARRIED, (
        f"primary pair must be undisturbed, got {mem.state}"
    )


def _pair_info(norm_distance=0.5, **overrides):
    info = _PairInfo(
        person_id=1,
        bag_id=10001,
        timestamp=1.0,
        frame_index=10,
        distance=100.0,
        norm_distance=norm_distance,
        person_centroid=(100.0, 100.0),
        person_height=160.0,
        near=False,
        carried=False,
        stationary=True,
        departed=True,
        score=0.5,
        containment=0.0,
        bag_confidence=0.9,
        bag_class="trash_bag",
        fallback=False,
        yolo_confirmed=True,
        bag_uid=1,
        pose_score=None,
        pose_components={},
        legacy_score=0.5,
        bag_centroid=(200.0, 200.0),
        bag_below_feet=False,
        near_ground_plane=True,
        in_bin_zone=False,
        person_moving=True,
        moves_with_person=False,
        wrist_near=False,
    )
    for k, v in overrides.items():
        setattr(info, k, v)
    return info


def test_release_window_frames_is_consulted_p2_1():
    """P2-1 wiring: the windowed distance-growth release test must be BOUNDED
    by release_window_frames (most recent ticks only), not the full history.

    History: an early rise, then a flat/slightly-falling tail. The FULL-history
    test would keep 'growing' forever; the bounded window must not."""
    cfg = _cfg()
    cfg = EventDetectorConfig(**{**cfg.to_dict(), "release_window_frames": 4})
    det = LitteringEventDetector(cfg)
    mem = _PairMemory(person_id=1, bag_id=10001, bag_uid=1)
    mem.carried_frames = 10
    # Early rise (5 -> 40), then a falling/flat tail inside the recent window.
    for d in (5.0, 40.0, 30.0, 20.0, 10.0):
        mem.distances.append(d)
    info = _pair_info(norm_distance=0.5)
    # distance_increasing=False, so ONLY the windowed-growth criterion could
    # fire; with the bounded window (last 4: 40,30,20,10) it must not.
    assert det._release_detected(mem, info, smooth_carried=False, distance_increasing=False) is False
    # Sanity: the same tail RISING must fire (window growth works).
    mem.distances.clear()
    for d in (5.0, 5.0, 5.0, 20.0, 30.0, 40.0):
        mem.distances.append(d)
    assert det._release_detected(mem, info, smooth_carried=False, distance_increasing=False) is True


def test_fallback_gap_gate_p2_1_wiring():
    """P2-1 wiring: max_fallback_tracker_gap_frames gates confirmation — a pair
    coasting on a non-YOLO tracker longer than the cap is rejected as
    BAG_NOT_DETECTED (design principle #4). DORMANT in production today (only
    yolo-source bags form pairs); the gate is the safety net."""
    cfg = EventDetectorConfig(**{**_cfg().to_dict(), "max_fallback_tracker_gap_frames": 4})
    det = LitteringEventDetector(cfg)
    # bag_seen_frames must be >0, otherwise _rejection_reason's earlier
    # "never detected at all" check (line 1) short-circuits to BAG_NOT_DETECTED
    # before the fallback-gap gate under test is ever reached, which would
    # make every assertion below pass/fail for the wrong reason.
    mem = _PairMemory(person_id=1, bag_id=10001, bag_uid=1, bag_seen_frames=10)
    evidence = det._compute_evidence(mem)
    # Gap within the cap -> no fallback rejection.
    mem.fallback_gap_frames = 4
    assert det._rejection_reason(mem, evidence) != "BAG_NOT_DETECTED"
    # Gap beyond the cap -> BAG_NOT_DETECTED (before any other gate result).
    mem.fallback_gap_frames = 5
    assert det._rejection_reason(mem, evidence) == "BAG_NOT_DETECTED"
    # YOLO reconfirmation resets the streak.
    mem.fallback_gap_frames = 0
    assert det._rejection_reason(mem, evidence) != "BAG_NOT_DETECTED"


def test_static_low_conf_ground_clutter_never_enters_pair_memory():
    """2026-09-10 clutter gate: a low-confidence (< min_carry_origin_confidence)
    object resting on the ground plane that is NOT near/carried by the person
    must never enter FSM pair memory — even though it is within the loose
    association radius. This is the NESCAFE-tin-on-the-floor case."""
    det = LitteringEventDetector(_cfg())
    person = _person(1, 640, 180)   # bbox (600,100)-(680,260)
    # Bag at (700,320): associated (dist 152px <= 160px person-height radius)
    # but NOT near, NOT carried, below the feet line, and low confidence.
    clutter = _bag(90001, 700, 320, conf=0.30)
    _ticks(det, [person], [clutter], 8)
    assert det._pairs == {}, (
        f"static low-conf ground clutter formed pair memories: {list(det._pairs)}"
    )
    # Control: the same geometry with healthy confidence DOES enter the FSM.
    det2 = LitteringEventDetector(_cfg())
    solid = _bag(90001, 700, 320, conf=0.90)
    _ticks(det2, [person], [solid], 8)
    assert det2._pairs, "high-confidence associated bag must still form a pair"


def test_carry_requires_wrist_link_when_pose_available():
    """CARRY-ORIGIN CONSTRAINT: with wrist keypoints available, an object in
    the carry zone can only become CARRIED if a wrist was ever spatially
    linked to it. A bag riding in the zone without any hand contact (e.g.
    clutter passing through the zone projection) stays BAG_NEAR_PERSON."""
    det = LitteringEventDetector(_cfg())
    # Wrists far from the bag (right wrist up by the head).
    kp_far = DetectorKeypoints(left_wrist=(560, 120), right_wrist=(660, 110))
    person = _person(1, 640, 180)
    person.keypoints = kp_far
    # Bag inside the carry zone (chest height, above the feet line).
    bag = _bag(91001, 640, 190, conf=0.9)
    _ticks(det, [person], [bag], 10)
    states = [m.state for m in det._pairs.values()]
    assert states and all(s != EventState.BAG_CARRIED for s in states), (
        f"bag without any wrist linkage must NOT reach CARRIED, got {states}"
    )
    # Control: with the wrist ON the bag, the carry proceeds.
    det2 = LitteringEventDetector(_cfg())
    kp_near = DetectorKeypoints(left_wrist=(640, 190), right_wrist=(660, 110))
    person2 = _person(1, 640, 180)
    person2.keypoints = kp_near
    bag2 = _bag(91001, 640, 190, conf=0.9)
    _ticks(det2, [person2], [bag2], 10)
    states2 = [m.state for m in det2._pairs.values()]
    assert states2 and any(s == EventState.BAG_CARRIED for s in states2), (
        f"wrist-linked bag must reach CARRIED, got {states2}"
    )
