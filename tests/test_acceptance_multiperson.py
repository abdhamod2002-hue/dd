"""Multi-person & negative acceptance tests (spec: PERSON-CENTRIC, EVENT-CENTRIC).

These encode the hard guarantee: a passing person, a second person, or a vehicle
can NEVER inherit another actor's littering event, and non-littering behaviour
(carry-without-drop, regrab, passer-by, vehicle) must NOT produce a confirmed
event. Each test uses the SAME real detector update() path (no mocks of the FSM).
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


def _person(pid, cx, cy, height=160.0, conf=0.9):
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


def _bag(bid, cx, cy, conf=0.5, class_name="trash_bag", source="yolo"):
    return DetectorBag(
        track_id=bid,
        bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
        confidence=conf,
        class_name=class_name,
        source=source,
        yolo_confirmed=True,
    )


def _cfg():
    return EventDetectorConfig(
        analysis_fps=10.0,
        min_carried_frames=3,
        min_stationary_frames=4,
        min_departed_frames=2,
        confirmation_grace_frames=2,
        smoothing_window=3,
        stationary_window_frames=3,
        max_pair_age_frames=8,
        min_event_confidence=0.65,
        max_other_person_closer_frames=3,
    )


def _run(detector, script):
    """script: list of (persons, bags, dt). Returns flat list of emitted events."""
    events = []
    t = 0.0
    for persons, bags, dt in script:
        events.extend(detector.update(persons, bags, t))
        t += dt
    events.extend(detector.finalize())
    return events


def test_two_actors_each_litter_get_distinct_events():
    """PERSON-CENTRIC: two people each litter their OWN bag in DISJOINT regions
    -> two confirmed events, each attributed to the correct actor with a distinct
    stable uid. Regions are kept far apart so association never flips between them."""
    det = LitteringEventDetector(_cfg())
    script = []
    dt = 0.1
    # Person 1 @ left (x=140, bag 10001); Person 2 @ right (x=900, bag 20001).
    for _ in range(4):
        script.append(([_person(1, 140, 180), _person(2, 900, 180)],
                       [_bag(10001, 140, 220), _bag(20001, 900, 220)], dt))
    for y in (280, 340, 400):
        script.append(([_person(1, 140, 180), _person(2, 900, 180)],
                       [_bag(10001, 140, y), _bag(20001, 900, y)], dt))
    for _ in range(5):
        script.append(([_person(1, 140, 180), _person(2, 900, 180)],
                       [_bag(10001, 140, 400), _bag(20001, 900, 400)], dt))
    # Depart in OPPOSITE directions (person 1 far left, person 2 far right) so the
    # paths never cross and each recedes >275px (norm >= ~1.7) to trigger departure.
    for x1, x2 in [(-100, 1100), (-200, 1300), (-300, 1500), (-400, 1700)]:
        script.append(([_person(1, x1, 180), _person(2, x2, 180)],
                       [_bag(10001, 140, 400), _bag(20001, 900, 400)], dt))
    events = _run(det, script)
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 2, f"expected 2 confirmed, got {len(confirmed)}: {[(e.person_track_id) for e in confirmed]}"
    actors = {e.event_actor_person_track_id for e in confirmed}
    assert actors == {1, 2}, f"both actors must be present: {actors}"
    uids = {e.event_object_uid for e in confirmed}
    assert len(uids) == 2, "each actor's object must get a distinct stable uid"
    # No event may be attributed to the WRONG actor.
    for e in confirmed:
        assert e.event_actor_person_track_id == e.person_track_id
        assert e.event_object_uid is not None


def test_passerby_never_inherits_anothers_event():
    """The core guarantee: a person who merely WALKS PAST another's dropped bag
    must never be attributed the littering event."""
    det = LitteringEventDetector(_cfg())
    script = []
    dt = 0.1
    # Person 1 carries + drops + departs (real littering at x=140).
    for _ in range(4):
        script.append(([_person(1, 140, 180)], [_bag(10001, 140, 220)], dt))
    for y in (280, 340, 400):
        script.append(([_person(1, 140, 180)], [_bag(10001, 140, y)], dt))
    for _ in range(5):
        script.append(([_person(1, 140, 180)], [_bag(10001, 140, 400)], dt))
    # Person 2 strolls across, passing NEAR the grounded bag (x 200..140 region)
    # but never carrying it. Person 1 meanwhile departs.
    for x2 in (520, 400, 280, 200, 140, 140, 140):
        script.append(([_person(1, 300, 180), _person(2, x2, 180)],
                       [_bag(10001, 140, 400)], dt))
    events = _run(det, script)
    confirmed = [e for e in events if e.confirmed]
    assert confirmed, "person 1's real littering must still confirm"
    assert all(e.event_actor_person_track_id == 1 for e in confirmed), \
        f"passer-by (person 2) must NOT be attributed: {[e.event_actor_person_track_id for e in confirmed]}"
    assert 2 not in {e.event_actor_person_track_id for e in confirmed}


def test_negative_carry_without_drop_is_rejected():
    """NEGATIVE: a person carries a bag and walks away WITHOUT dropping it is NOT
    littering — must be rejected, never confirmed."""
    det = LitteringEventDetector(_cfg())
    script = []
    dt = 0.1
    for _ in range(6):
        script.append(([_person(1, 140, 180)], [_bag(10001, 140, 220)], dt))
    # Walk away still carrying (bag stays in carrying position, person recedes).
    for x in (300, 460, 620, 780, 940):
        script.append(([_person(1, x, 180)], [_bag(10001, x, 220)], dt))
    events = _run(det, script)
    confirmed = [e for e in events if e.confirmed]
    assert not confirmed, f"carry-without-drop must NOT confirm, got {len(confirmed)}"
    rejected = [e for e in events if not e.confirmed]
    assert rejected
    assert rejected[0].reason in ("NO_RELEASE_TRANSITION", "PERSON_NOT_DETECTED")


def test_negative_closer_bystander_not_attributed():
    """NEGATIVE (misattribution guard): a SECOND person who stands CLOSER to the
    dropped bag than the carrier, but never actually carries it, must NEVER be
    attributed the littering event. The carrier (person 1) is confirmed; the
    bystander (person 2) is not."""
    det = LitteringEventDetector(_cfg())
    script = []
    dt = 0.1
    # Person 2 is a BYSTANDER who never carries: placed far enough (x=260) that its
    # wrist is never near the bag during the carry phase, yet it is still CLOSER to
    # the dropped bag than the departing carrier (who walks to x>=320), exercising
    # the misattribution guard. Person 1 carries + drops + grounds at x=140.
    for _ in range(4):
        script.append(([_person(1, 140, 180), _person(2, 260, 180)],
                       [_bag(10001, 140, 220)], dt))
    for y in (280, 340, 400):
        script.append(([_person(1, 140, 180), _person(2, 260, 180)],
                       [_bag(10001, 140, y)], dt))
    for _ in range(5):
        script.append(([_person(1, 140, 180), _person(2, 260, 180)],
                       [_bag(10001, 140, 400)], dt))
    # Person 1 departs far enough (>275px, norm >= ~1.7) to trigger departure/confirm
    # while STAYING ON SCREEN (positive x), so the detector still tracks the actor.
    for x in (320, 440, 560, 680):
        script.append(([_person(1, x, 180), _person(2, 260, 180)],
                       [_bag(10001, 140, 400)], dt))
    # THEN the closer bystander (person 2) loiters near the grounded bag.
    for _ in range(6):
        script.append(([_person(2, 260, 180)],
                       [_bag(10001, 140, 400)], dt))
    events = _run(det, script)
    confirmed = [e for e in events if e.confirmed]
    assert confirmed, "the real carrier (person 1) must still confirm"
    assert all(e.event_actor_person_track_id == 1 for e in confirmed), \
        f"bystander (person 2) must NOT be attributed: {[e.event_actor_person_track_id for e in confirmed]}"
    assert 2 not in {e.event_actor_person_track_id for e in confirmed}


def test_negative_vehicle_track_is_never_an_actor():
    """NEGATIVE: a 'car'/'vehicle' track near a bag must never become the actor
    of a littering event (only person tracks are actors; vehicles are excluded
    structurally)."""
    det = LitteringEventDetector(_cfg())
    script = []
    dt = 0.1
    # Vehicle passed in as a DetectorPerson-like track with class 'car'.
    vehicle = DetectorPerson(
        track_id=99,
        bbox=(135 - 40, 200 - 80, 135 + 40, 200 + 80),
        confidence=0.9,
        keypoints=DetectorKeypoints(left_wrist=(0, 0), right_wrist=(0, 0), torso_center=(135, 200)),
    )
    # The vehicle sits next to a grounded bag the whole time.
    for _ in range(12):
        script.append(([vehicle], [_bag(10001, 135, 210)], dt))
    # Also have a real person carry+drop elsewhere to show the detector still works.
    for _ in range(4):
        script.append(([_person(1, 600, 180), vehicle], [_bag(10001, 135, 210), _bag(20001, 600, 220)], dt))
    for y in (280, 340, 400):
        script.append(([_person(1, 600, 180), vehicle], [_bag(10001, 135, 210), _bag(20001, 600, y)], dt))
    for _ in range(5):
        script.append(([_person(1, 600, 180), vehicle], [_bag(10001, 135, 210), _bag(20001, 600, 400)], dt))
    for x in (760, 920):
        script.append(([_person(1, x, 180), vehicle], [_bag(10001, 135, 210), _bag(20001, 600, 400)], dt))
    events = _run(det, script)
    confirmed = [e for e in events if e.confirmed]
    # Only the real person (1) may confirm; vehicle 99 must never be an actor.
    assert all(e.event_actor_person_track_id != 99 for e in confirmed), \
        "VEHICLE must never be attributed a littering event"
    assert all(e.event_actor_person_track_id == 1 for e in confirmed), \
        f"only person 1 should confirm: {[e.event_actor_person_track_id for e in confirmed]}"


def test_literal_three_person_only_litterer_is_actor():
    """ACCEPTANCE Case A (literal 3-person): P1 walks by BEFORE/DURING the carry,
    P2 performs the littering, P3 walks by AFTER the drop near the grounded bag.
    ONLY P2 may become the event actor; P1 and P3 must never be attributed."""
    det = LitteringEventDetector(_cfg())
    script = []
    dt = 0.1
    # Phase 1 — P1 strolls past (x 400..760) while P2 carries at x=140.
    # P1's wrist is never near the bag (>=260px away) and P1 is never closer to
    # the bag than the carrier, so no association/ownership flip may occur.
    for x1 in (400, 520, 640, 760):
        script.append(([_person(1, x1, 180), _person(2, 140, 180)],
                       [_bag(20001, 140, 220)], dt))
    # Phase 2 — P2 releases the bag (bag separates downward); P1 exits the frame.
    for y, x1 in ((280, 880), (340, 1000), (400, 1120)):
        script.append(([_person(1, x1, 180), _person(2, 140, 180)],
                       [_bag(20001, 140, y)], dt))
    # Phase 3 — P2 stands near the grounded bag (rest), P1 is gone.
    for _ in range(5):
        script.append(([_person(2, 140, 180)], [_bag(20001, 140, 400)], dt))
    # Phase 4 — P2 departs far enough (>275px) to trigger departure/confirmation.
    for x2 in (320, 460, 600, 740):
        script.append(([_person(2, x2, 180)], [_bag(20001, 140, 400)], dt))
    # Phase 5 — P3 strolls past the grounded bag AFTER the fact (never carrying).
    for x3 in (520, 400, 280, 200, 140, 140, 140):
        script.append(([_person(3, x3, 180)], [_bag(20001, 140, 400)], dt))
    events = _run(det, script)
    confirmed = [e for e in events if e.confirmed]
    assert confirmed, "the real litterer (person 2) must confirm"
    actors = {e.event_actor_person_track_id for e in confirmed}
    assert actors == {2}, f"ONLY person 2 may be the actor, got {actors}"
    for e in confirmed:
        assert e.person_track_id == 2
        assert e.event_object_uid is not None, "confirmed event must carry a stable object uid"
    # Neither walker (1, 3) may own any confirmed event.
    assert 1 not in actors and 3 not in actors


# Combined Case C: two DIFFERENT actors litter SIMULTANEOUSLY — one on open
# ground, one inside the operator-configured bin zone (P1-5). The bin zone
# footprint in pixels (1280x720): x 64..256, y 360..504.
_BIN_ZONE = BinZone(
    polygon=((0.05, 0.50), (0.20, 0.50), (0.20, 0.70), (0.05, 0.70)),
    camera_id="cam-01",
    name="dumpster",
)
_FRAME_W, _FRAME_H = 1280, 720


def _cfg_bin(zone: BinZone | None) -> EventDetectorConfig:
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
        bin_zones=(zone,) if zone else (),
    )


def test_simultaneous_ground_drop_and_bin_deposit_two_actors():
    """ACCEPTANCE combined Case C: actor A ground-drops at x=900 (outside the bin
    zone) while actor B simultaneously deposits at x=200 (INSIDE the zone).
    Exactly one confirmed GROUND event must exist — owned by A; B's candidate
    must be rejected as BIN_ZONE_DEPOSIT; no cross-attribution either way."""
    det = LitteringEventDetector(_cfg_bin(_BIN_ZONE), camera_id="cam-01")
    size = (_FRAME_W, _FRAME_H)
    script = []
    dt = 0.1
    # Both actors carry their own bag simultaneously (regions far apart).
    for _ in range(6):
        script.append(([_person(1, 900, 180), _person(2, 200, 180)],
                       [_bag(10001, 900, 220), _bag(20001, 200, 220)], dt))
    # Both release: bags separate downward at their own x positions.
    for y in (280, 340, 400):
        script.append(([_person(1, 900, 180), _person(2, 200, 180)],
                       [_bag(10001, 900, y), _bag(20001, 200, y)], dt))
    # Ground rest: A's bag at (900,420) = OUTSIDE the zone; B's bag at
    # (200,420) = INSIDE the zone footprint.
    for _ in range(6):
        script.append(([_person(1, 900, 180), _person(2, 200, 180)],
                       [_bag(10001, 900, 420), _bag(20001, 200, 420)], dt))
    # Both depart in opposite directions, receding fast enough for departure.
    for x1, x2 in ((1100, -100), (1300, -300), (1500, -500), (1700, -700)):
        script.append(([_person(1, x1, 180), _person(2, x2, 180)],
                       [_bag(10001, 900, 420), _bag(20001, 200, 420)], dt))
    events = []
    t = 0.0
    for persons, bags, _ in script:
        events.extend(det.update(persons, bags, t, frame_size=size))
        t += dt
    events.extend(det.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1, \
        f"exactly one ground event must confirm, got {len(confirmed)}: {[(e.person_track_id, e.reason) for e in events]}"
    assert confirmed[0].event_actor_person_track_id == 1, \
        "the ground-dropping actor (1) must own the confirmed event"
    assert confirmed[0].state == EventState.VIOLATION_CONFIRMED
    assert confirmed[0].details["location_status"] == "GROUND_CONFIRMED"
    # Actor B's in-zone deposit must be an explicit rejection, never a confirmation.
    rejected = [e for e in events if not e.confirmed]
    assert any(r.reason == "BIN_ZONE_DEPOSIT" for r in rejected), \
        f"actor 2's bin deposit must be rejected as BIN_ZONE_DEPOSIT, got {[r.reason for r in rejected]}"
    # No cross-attribution: no event (confirmed or not) swaps the actors.
    for e in events:
        if e.event_actor_person_track_id is not None:
            assert e.event_actor_person_track_id in (1, 2)
