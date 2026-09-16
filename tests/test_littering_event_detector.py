"""Tests for the temporal littering event detector."""

from __future__ import annotations

import os

from littering_event_detector import (
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    EventState,
    LitteringEventDetector,
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


def _bag(bid: int, cx: float, cy: float, conf: float = 0.5, source: str = "yolo", yolo_confirmed: bool = True) -> DetectorBag:
    return DetectorBag(
        track_id=bid,
        bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
        confidence=conf,
        class_name="trash_bag",
        source=source,
        yolo_confirmed=yolo_confirmed,
    )


def _fast_config() -> EventDetectorConfig:
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
    )


def test_positive_littering_sequence_confirms():
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0

    # Carry: bag near lower body for several frames.
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1

    # Release: bag moves downward and separates.
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1

    # Ground: bag stationary.
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1

    # Departure: person moves far away.
    for x in (360, 520, 680, 840):
        events.extend(detector.update([_person(1, x, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1

    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1
    event = confirmed[0]
    assert event.state == EventState.VIOLATION_CONFIRMED
    assert event.person_track_id == 1
    assert event.bag_track_id == 10001
    assert event.confidence >= 0.65
    assert event.frames["carry_start"] is not None
    assert event.frames["release"] is not None
    assert event.frames["ground"] is not None
    assert event.frames["departure"] is not None


def test_abandonment_confirms_when_person_stays():
    """A bag carried, released, and left on the ground (person stays) is a
    valid littering event via ABANDONMENT — the object is left behind and not
    reclaimed. This is intentional: littering does not require the person to
    walk far away. The regrab path still protects the 'picks it back up' case.
    """
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0

    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(12):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1

    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1
    assert confirmed[0].state == EventState.VIOLATION_CONFIRMED
    assert confirmed[0].frames["ground"] is not None


def test_fallback_only_full_sequence_confirms():
    """REPAIR-P0-02: color-sourced bags (HSV path) MAY confirm with the
    fallback confidence discount. Novelty / detected_object remain proposal-only.
    """
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0

    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)],
                                      [_bag(10001, 140, 220, source="color", yolo_confirmed=False)], t))
        t += 0.1
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)],
                                      [_bag(10001, 140, y, source="color", yolo_confirmed=False)], t))
        t += 0.1
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)],
                                      [_bag(10001, 140, 420, source="color", yolo_confirmed=False)], t))
        t += 0.1
    for x in (360, 520, 680, 840):
        events.extend(detector.update([_person(1, x, 180)],
                                      [_bag(10001, 140, 420, source="color", yolo_confirmed=False)], t))
        t += 0.1

    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1, "color-sourced full littering sequence must confirm (discounted)"
    ev = confirmed[0]
    assert ev.fallback_used is True
    assert ev.confidence >= 0.55


def test_novelty_alone_never_confirms():
    """Novelty proposals must never become littering events by themselves."""
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    nov = DetectorBag(
        track_id=10001,
        bbox=(125, 205, 155, 235),
        confidence=0.9,
        class_name="detected_object",
        source="novelty",
        yolo_confirmed=False,
    )
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [nov], t))
        t += 0.1
    for y in (280, 340, 400):
        nov = DetectorBag(
            track_id=10001,
            bbox=(125, y - 15, 155, y + 15),
            confidence=0.9,
            class_name="detected_object",
            source="novelty",
            yolo_confirmed=False,
        )
        events.extend(detector.update([_person(1, 140, 180)], [nov], t))
        t += 0.1
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [nov], t))
        t += 0.1
    for x in (360, 520, 680, 840):
        events.extend(detector.update([_person(1, x, 180)], [nov], t))
        t += 0.1
    events.extend(detector.finalize())
    assert not any(e.confirmed for e in events)


def test_carry_only_incomplete_rejected():
    """Carry that never transitions to a release is honestly rejected with
    NO_RELEASE_TRANSITION (the clip ended mid-carry)."""
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    events.extend(detector.finalize())
    rejected = [e for e in events if not e.confirmed]
    assert rejected
    assert rejected[0].reason == "NO_RELEASE_TRANSITION"


def test_rebind_across_id_churn_confirms():
    """The HSV fallback tracker re-births the same bag under a new track id.
    The detector must preserve the (person, bag) temporal evidence across the
    id change and still confirm the event."""
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    # Carry with id 60001.
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(60001, 140, 220)], t))
        t += 0.1
    # One frame where the fallback tracker loses the bag entirely.
    events.extend(detector.update([_person(1, 140, 180)], [], t))
    t += 0.1
    # Bag re-appears under a new id 60002, continues the same physical motion.
    for y in (280, 340, 400):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(60002, 140, y)], t))
        t += 0.1
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(60002, 140, 420)], t))
        t += 0.1
    for x in (360, 520, 680, 840):
        events.extend(detector.update([_person(1, x, 180)], [_bag(60002, 140, 420)], t))
        t += 0.1
    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1


def test_non_carried_bag_not_confirmed():
    """A bag sitting on the ground that is merely associated to a person who
    never carries it must NOT confirm (no carry/release transition)."""
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    for _ in range(20):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 420)], t))
        t += 0.1
    events.extend(detector.finalize())
    assert not any(e.confirmed for e in events)


def test_summary_counts_real_events():
    detector = LitteringEventDetector(_fast_config())
    t = 0.0
    for _ in range(6):
        detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t)
        t += 0.1
    detector.finalize()
    summary = detector.summary()
    assert summary["confirmed_violations"] == 0
    assert summary["rejected_candidates"] >= 0
    assert summary["total_candidates"] == summary["confirmed_violations"] + summary["rejected_candidates"]


def test_load_event_config_accepts_uppercase_yaml(tmp_path):
    path = os.path.join(str(tmp_path), "events.yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("ANALYSIS_FPS: 12.0\nMIN_CARRIED_FRAMES: 5\nMIN_EVENT_CONFIDENCE: 0.7\n")
    cfg = load_event_config(path)
    assert cfg.analysis_fps == 12.0
    assert cfg.min_carried_frames == 5
    assert cfg.min_event_confidence == 0.7


def test_put_down_then_regrab_reclassified_picked_back_up():
    """Regrab protection: a bag carried, released, grounded, then PICKED BACK
    UP and kept must NOT confirm a violation, and must be explicitly
    reclassified as PICKED_BACK_UP (NOT_ABANDONED) at finalization so the
    avoided false positive is auditable.

    This is the detector-level counterpart of
    test_put_down_then_regrab_does_not_confirm (pipeline level) — it additionally
    asserts the explicit reclassification rather than a silent drop.

    NOTE: the bag is set down NEAR the person (within the departure threshold of
    the release baseline) so the ABANDONMENT path is active, not the far-away
    DEPARTURE path. Feeding the detector a bag far below the person would instead
    read as "departed" and confirm (the pipeline's association layer filters such
    far bags out before they reach the detector, which is why the pipeline test
    keeps the bag at y=420 without confirming).
    """
    detector = LitteringEventDetector(_fast_config())
    events: list = []
    t = 0.0
    # Carry: bag held at the carry position (distance 40 -> norm_distance 0.25).
    for _ in range(8):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    # Release: bag moved progressively downward so the detector sees a sustained
    # "not carried + distance increasing" release (a single-step drop would be
    # swallowed by the smoothing window). Release fires around y=280.
    for y in (240, 260, 280):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    # Ground briefly (enough for BAG_ON_GROUND, not long enough to abandon).
    for _ in range(3):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 300)], t))
        t += 0.1
    # Regrab immediately: lift well above the release centroid (y≈280 → y=160)
    # with wrists ON the bag so AIDM attach (d_norm <= 0.15) allows reclaim.
    for y in (240, 200, 160, 150, 150):
        p = _person(1, 140, 180)
        p.keypoints = DetectorKeypoints(
            left_wrist=(140.0, float(y)),
            right_wrist=(150.0, float(y) - 10.0),
            torso_center=(140.0, 160.0),
        )
        events.extend(detector.update([p], [_bag(10001, 140, y)], t))
        t += 0.1
    # Keep carrying it (held above the ground plane, wrists still gripping)
    for _ in range(6):
        p = _person(1, 140, 180)
        p.keypoints = DetectorKeypoints(
            left_wrist=(140.0, 160.0),
            right_wrist=(150.0, 150.0),
            torso_center=(140.0, 160.0),
        )
        events.extend(detector.update([p], [_bag(10001, 140, 160)], t))
        t += 0.1

    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 0, "regrab must revert; no littering event expected"

    finalized = detector.finalize()
    rejected = [e for e in finalized if not e.confirmed]
    picked = [
        e
        for e in rejected
        if e.state == EventState.PICKED_BACK_UP and e.reason == "PICKED_BACK_UP"
    ]
    assert picked, "reclaimed pair must be reclassified as PICKED_BACK_UP (NOT_ABANDONED)"
    assert len(picked) == 1
    # The reclassification must not be a generic NO_RELEASE_TRANSITION drop.
    assert all(e.reason != "NO_RELEASE_TRANSITION" for e in picked)


def test_feet_putdown_outside_strict_band_releases():
    """REPAIR-P0-01: bag centroid in near_ground_plane band but outside the
    old 0.05·ph bag_below_feet band must unlatch carried and allow release
    while the person stands still (the IMG_5117 failure class).
    """
    # Person cy=180, h=160 → feet y=260.
    # near_ground_plane (0.40): y >= 260 - 64 = 196
    # bag_below_feet (0.05):   y >= 260 - 8  = 252
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0

    # Carry above the ground-plane band (y=185 < 196) with wrist nearby.
    for _ in range(6):
        events.extend(
            detector.update([_person(1, 140, 180)], [_bag(10001, 140, 185)], t)
        )
        t += 0.1

    # Put-down into the 0.05–0.40 gap (y=230): near_ground True, strict feet False.
    # Move wrists away so wrist_near cannot keep carried latched.
    standing = DetectorPerson(
        track_id=1,
        bbox=(100.0, 100.0, 180.0, 260.0),
        confidence=0.9,
        keypoints=DetectorKeypoints(
            left_wrist=(170, 120),
            right_wrist=(110, 120),
            torso_center=(140, 160),
        ),
    )
    for _ in range(8):
        events.extend(detector.update([standing], [_bag(10001, 140, 230)], t))
        t += 0.1

    # Abandonment window while person stays near the grounded bag.
    for _ in range(12):
        events.extend(detector.update([standing], [_bag(10001, 140, 230)], t))
        t += 0.1

    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1, (
        f"expected confirmation for feet put-down in ground-plane gap; "
        f"got confirmed={len(confirmed)} events={[ (e.state, e.reason) for e in events ]}"
    )
    assert confirmed[0].frames["release"] is not None
    assert confirmed[0].frames["carry_start"] is not None


def test_wrist_near_grounded_bag_still_releases():
    """REPAIR-P0-01b: after a real carry, a bag resting on the ground plane
    must release even if wrists remain geometrically near the bag (standing
    over a put-down) — the failure mode that survived the first margin fix.
    """
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    # Carry above ground plane.
    for _ in range(6):
        events.extend(
            detector.update([_person(1, 140, 180)], [_bag(10001, 140, 185)], t)
        )
        t += 0.1
    # Put-down at y=230 with wrists still near the bag (default _person wrists).
    for _ in range(15):
        events.extend(
            detector.update([_person(1, 140, 180)], [_bag(10001, 140, 230)], t)
        )
        t += 0.1
    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1, (
        f"wrist-near grounded put-down must confirm; got "
        f"{[(e.state, e.reason, e.evidence) for e in events if not e.confirmed][:3]}"
    )


def test_desync_putdown_while_bag_stays_in_ground_band():
    """REPAIR-P0-01c: bag held in the ground-plane band for the whole carry
    (never ever_off_ground) must still release when the person walked while
    carrying and the bag later becomes stationary / unsynced.
    """
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    # Walk while carrying bag at y=220 (inside ground band for this geometry).
    for x in (140, 160, 180, 200, 220, 240, 260, 280):
        events.extend(
            detector.update([_person(1, x, 180)], [_bag(10001, x, 220)], t)
        )
        t += 0.1
    # Stop: bag stays put, person stands near it.
    for _ in range(15):
        events.extend(
            detector.update([_person(1, 280, 180)], [_bag(10001, 220, 230)], t)
        )
        t += 0.1
    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 1, (
        f"desync put-down must confirm; got "
        f"{[(e.state, e.reason) for e in events]}"
    )


def test_passerby_near_grounded_bag_no_event():
    """REPAIR-P0-01 regression: standing next to a static grounded bag without
    prior carry must not invent a littering event.
    """
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    standing = DetectorPerson(
        track_id=1,
        bbox=(100.0, 100.0, 180.0, 260.0),
        confidence=0.9,
        keypoints=DetectorKeypoints(
            left_wrist=(170, 120),
            right_wrist=(110, 120),
            torso_center=(140, 160),
        ),
    )
    for _ in range(20):
        events.extend(detector.update([standing], [_bag(10001, 140, 230)], t))
        t += 0.1
    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert len(confirmed) == 0


def test_carry_only_still_rejected_as_no_release():
    """A carry with no release must remain a generic NO_RELEASE_TRANSITION
    rejection (not PICKED_BACK_UP) — reclassification only applies when an
    actual reclamation occurred."""
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0
    for _ in range(6):
        events.extend(detector.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    finalized = detector.finalize()
    rejected = [e for e in finalized if not e.confirmed]
    assert rejected
    assert rejected[0].reason == "NO_RELEASE_TRANSITION"
    assert rejected[0].state != EventState.PICKED_BACK_UP


def test_tracker_id_churn_keeps_single_stable_object_identity():
    """ROOT-CAUSE FIX (spec #11): when the underlying tracker churns a physical
    bag's id many times across an event (the real D:\\22 failure mode), the
    detector must still track ONE stable pair + ONE stable bag_uid for the actor,
    and must still confirm the littering event. A naive bag-id-keyed pair would
    split one physical bag into many identities and either fragment the evidence
    or amplify a transient blip into a false positive.
    """
    detector = LitteringEventDetector(_fast_config())
    events = []
    t = 0.0

    # --- Carry (with an id churn mid-carry) ---
    for bid, y in [(10001, 220)] * 4 + [(10002, 220)] * 4:
        events.extend(detector.update([_person(1, 140, 180)], [_bag(bid, 140, y)], t))
        t += 0.1
    # --- Release (with another id churn) ---
    for bid, y in [(10002, 280), (10003, 340), (10003, 400)]:
        events.extend(detector.update([_person(1, 140, 180)], [_bag(bid, 140, y)], t))
        t += 0.1
    # --- Ground (bag stationary, id churns again) ---
    for bid in [10003, 10004, 10005, 10004, 10005, 10004]:
        events.extend(detector.update([_person(1, 140, 180)], [_bag(bid, 140, 400)], t))
        t += 0.1
    # --- Departure (person recedes from the grounded bag) ---
    for x in (300, 460, 620, 780):
        events.extend(detector.update([_person(1, x, 180)], [_bag(10005, 140, 400)], t))
        t += 0.1

    # Exactly one pair for the actor, anchored to a single stable bag_uid.
    # Phase 1: the pair key's object slot is now the AUTHORITATIVE stable
    # object uid from the object-identity manager (spatial/temporal matching
    # across the churn above), not the legacy per-person counter (1). The
    # guarantee under test is "ONE pair + ONE constant uid", not the key
    # format; we additionally verify the uid survived every churned raw id.
    keys = [k for k in detector._pairs.keys() if k[0] == 1]
    assert len(keys) == 1, f"expected a single stable pair for actor 1, got {keys}"
    stable_uid = detector._pairs[keys[0]].bag_uid
    assert stable_uid is not None
    assert stable_uid == keys[0][1], "pair key must be anchored by the stable object uid"

    # The event must confirm and carry that SAME stable uid (not a churned id).
    confirmed = [e for e in events if e.confirmed]
    assert confirmed, "littering event must still confirm under tracker id churn"
    assert confirmed[0].person_track_id == 1
    assert confirmed[0].bag_uid == stable_uid
    # The live track id may have churned, but the event's identity is stable.
    assert confirmed[0].bag_track_id in (10001, 10002, 10003, 10004, 10005)
    # Phase C — authoritative actor/object record is FROZEN at carry time and
    # survives the churn that follows. The carry-time object id is 10001 (the id
    # present when the continuous temporal association first locked).
    assert confirmed[0].event_actor_person_track_id == 1
    assert confirmed[0].event_object_track_id == 10001, "object id must be frozen at carry, not a later churned id"
    assert confirmed[0].event_object_uid == stable_uid


def test_static_container_with_id_churn_never_confirms():
    """RCM-01 (forensic corrective plan): a large STATIC object (a container)
    detected as a semantic waste class, whose raw tracker id churns every
    tick exactly like the real detector does, must NEVER be attributed as
    carried-then-abandoned just because a person walks past it. This is the
    general form of the M.MOV/IMG_5613 failure class: a static container
    inheriting carry-then-litter status from a passer-by, caused by motion
    history that reset every id churn and therefore never proved the object
    was actually stationary. No video-specific data is used here — this is
    a synthetic id-churn walk-past, not a recorded clip.
    """
    cfg = EventDetectorConfig(
        analysis_fps=10.0,
        min_carried_frames=3,
        min_stationary_frames=4,
        min_departed_frames=2,
        confirmation_grace_frames=2,
        smoothing_window=3,
        stationary_window_frames=3,
        max_pair_age_frames=40,
        min_event_confidence=0.5,
        min_abandonment_frames=6,
    )
    detector = LitteringEventDetector(cfg)
    events = []
    t = 0.0
    raw_id_cycle = [9001, 9002, 9003, 9004]
    for f in range(40):
        # Person has NO pose keypoints (forces the no-pose carry-origin
        # fallback path, which is the one that used to accept motion
        # CORRELATION as carry evidence).
        px = 50.0 + f * 22.0
        person = DetectorPerson(
            track_id=1, bbox=(px - 40.0, 220.0, px + 40.0, 380.0), confidence=0.9
        )
        # Static container: fixed real-world position, but its RAW track id
        # churns every tick — exactly what a tracker does on the SAME
        # physical object across re-detections.
        bag = DetectorBag(
            track_id=raw_id_cycle[f % 4],
            bbox=(465.0, 270.0, 535.0, 330.0),
            confidence=0.85,
            class_name="garbage_bag",
            source="yolo",
            yolo_confirmed=True,
        )
        events.extend(detector.update([person], [bag], t, frame_index=f))
        t += 0.1
    events.extend(detector.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert confirmed == [], (
        "a static container must never be confirmed as carried/abandoned "
        "litter merely because a person walked past it, regardless of "
        "tracker id churn on the container itself"
    )
    # The container must be recognised as ONE stable physical object across
    # its own id churn (object-identity is working correctly here — the bug
    # this test guards against is in the FSM's motion/carry reasoning, not
    # in uid assignment).
    assert set(detector.last_object_uid_map.values()) == {
        next(iter(detector.last_object_uid_map.values()))
    }


def test_active_thresholds_disclosed_in_event_details():
    """P1-8: every event must disclose the EFFECTIVE runtime thresholds that
    produced it (details['active_thresholds']), so operators never have to
    guess YAML vs learned overrides."""
    cfg = EventDetectorConfig(
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
        release_distance_ratio=0.17,  # non-default value must be reflected
    )
    detector = LitteringEventDetector(cfg)
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

    emitted = [e for e in events if e.details.get("active_thresholds")]
    assert emitted, "every event must carry the active_thresholds disclosure"
    for e in emitted:
        at = e.details["active_thresholds"]
        assert at["min_carried_frames"] == 3
        assert at["release_distance_ratio"] == 0.17, (
            "disclosure must show the EFFECTIVE (possibly non-default) value"
        )
        assert at["min_event_confidence"] == 0.65
        assert "release_distance_floor" in at

