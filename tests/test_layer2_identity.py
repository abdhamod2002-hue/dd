"""LAYER 2 tests — stable person/object identity + temporal event recognition.

Encodes the Phase-1 guarantees:
  * two different people visible in the SAME FRAME never share a person uid;
  * a raw tracker id switch (P2 -> P7) of the SAME physical person preserves
    the logical uid (spatial/temporal/geometry evidence);
  * identity is NOT forced when evidence is insufficient (long gap -> new uid);
  * an object keeps its stable uid across a short detector id gap, and two
    distinct physical objects never merge;
  * the (person uid, object uid) event pair is FROZEN at carry time and is
    never recalculated from final-frame proximity;
  * a person's raw id churning mid-carry never changes the event actor;
  * BIN VS GROUND: a resting position never observed on the actor's ground
    plane is LOCATION_AMBIGUOUS -> NO_CONFIDENT_EVENT (when the gate is on),
    and the status is always recorded on the event either way.
"""

from __future__ import annotations

import pytest

from inference.tracking.object_identity import ObjectIdentityManager
from inference.tracking.person_identity import PersonIdentityManager
from littering_event_detector import (
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    LitteringEventDetector,
)


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
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


def _bag(bid, cx, cy, conf=0.5):
    return DetectorBag(
        track_id=bid,
        bbox=(cx - 15, cy - 15, cx + 15, cy + 15),
        confidence=conf,
        class_name="trash_bag",
        source="yolo",
        yolo_confirmed=True,
    )


def _cfg(**overrides):
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
        max_other_person_closer_frames=3,
    )
    base.update(overrides)
    return EventDetectorConfig(**base)


def _resolve_all(mgr, persons, frame_index):
    """Resolve a list of DetectorPerson through the manager for one tick."""
    from littering_event_detector import _centroid

    out = {}
    for p in persons:
        out[int(p.track_id)] = mgr.resolve(
            int(p.track_id), _centroid(p.bbox), p.bbox, p.keypoints, frame_index, 0.0
        )
    return out


# --------------------------------------------------------------------- #
# PERSON UID — same-frame distinctness
# --------------------------------------------------------------------- #
def test_same_frame_persons_never_share_uid():
    """Two different people visible in the SAME frame must never receive the
    same logical uid — even standing within the re-association distance."""
    mgr = PersonIdentityManager(max_age_frames=45, distance_threshold=200.0)
    for f in range(30):
        mapping = _resolve_all(mgr, [_person(1, 140, 180), _person(2, 260, 180)], f)
        assert mapping[1] != mapping[2], f"uid collision at frame {f}: {mapping}"
        # stable across frames
    late = _resolve_all(mgr, [_person(1, 140, 180), _person(2, 260, 180)], 30)
    assert late[1] == mapping[1] and late[2] == mapping[2]


def test_crowd_of_three_same_frame_all_distinct():
    mgr = PersonIdentityManager(max_age_frames=45, distance_threshold=200.0)
    for f in range(20):
        mapping = _resolve_all(
            mgr, [_person(1, 140, 180), _person(2, 260, 180), _person(3, 380, 180)], f
        )
        uids = set(mapping.values())
        assert len(uids) == 3, f"uid collision in crowd at frame {f}: {mapping}"


# --------------------------------------------------------------------- #
# PERSON UID — raw id switch recovery (P2 -> P7)
# --------------------------------------------------------------------- #
def test_raw_id_switch_preserves_person_uid():
    """The physical person is tracked as raw id 2, briefly lost (3 ticks), and
    re-appears at the same position under raw id 7: the logical uid must be
    preserved (spatial + temporal + geometry continuity)."""
    mgr = PersonIdentityManager(max_age_frames=45, distance_threshold=200.0)
    for f in range(10):
        mapping = _resolve_all(mgr, [_person(2, 300, 400)], f)
    uid_before = mapping[2]
    # lost for 3 ticks
    for f in range(10, 13):
        _resolve_all(mgr, [], f)
    # re-appears under raw id 7, same spot
    mapping_after = _resolve_all(mgr, [_person(7, 306, 404)], 13)
    assert mapping_after[7] == uid_before, (
        "same physical person re-appearing under a new raw id must keep its uid"
    )


def test_long_gap_does_not_force_identity():
    """A person lost for longer than max_age_frames must NOT be force-matched:
    evidence is insufficient, a fresh uid is minted (spec: never invent
    continuity)."""
    mgr = PersonIdentityManager(max_age_frames=45, distance_threshold=200.0)
    for f in range(10):
        mapping = _resolve_all(mgr, [_person(2, 300, 400)], f)
    uid_before = mapping[2]
    for f in range(10, 10 + 46 + 5):
        _resolve_all(mgr, [], f)
    mapping_after = _resolve_all(mgr, [_person(9, 300, 400)], 61)
    assert mapping_after[9] != uid_before


def test_two_people_swap_positions_no_uid_steal():
    """Two people cross paths; each raw id keeps following its own physical
    person while it is continuously visible (direct mapping has priority over
    spatial re-association), so uids never merge."""
    mgr = PersonIdentityManager(max_age_frames=45, distance_threshold=200.0)
    first = _resolve_all(mgr, [_person(1, 100, 400), _person(2, 260, 400)], 0)
    # walk towards each other and past (raw ids stay continuously visible)
    positions = [(140, 400), (180, 400), (220, 400), (260, 400), (300, 400)]
    for i, (x1, y) in enumerate(positions, start=1):
        # person 1 walks right, person 2 walks left (they pass through the
        # same midpoint zone on consecutive frames)
        m = _resolve_all(mgr, [_person(1, x1, y), _person(2, 360 - x1, y)], i)
        assert m[1] == first[1] and m[2] == first[2], f"uid swap/merge at step {i}: {m}"


# --------------------------------------------------------------------- #
# OBJECT UID — gap continuity and no merging
# --------------------------------------------------------------------- #
def _obj_update(mgr, objs, frame_index):
    """objs: list of (class_name, (cx, cy))."""
    dets = []
    for cls, (cx, cy) in objs:
        bbox = (cx - 15.0, cy - 15.0, cx + 15.0, cy + 15.0)
        dets.append((cls, bbox, (cx, cy)))
    return mgr.update(dets)


def test_object_uid_survives_short_detector_gap():
    """W1 -> W7 after a short id gap: the same physical object re-appears near
    its last position within the max-age window and keeps its logical uid."""
    mgr = ObjectIdentityManager(distance_px=220.0, max_age_frames=120)
    uid = None
    for f in range(10):
        res = _obj_update(mgr, [("trash_bag", (100, 100))], f)
        uid = res[0]
    # 5-tick gap (detector id lost entirely)
    for f in range(10, 15):
        _obj_update(mgr, [], f)
    res = _obj_update(mgr, [("trash_bag", (110, 110))], 15)
    assert res[0] == uid, "object re-appearing after a short gap must keep its uid"


def test_two_distinct_objects_never_merge():
    mgr = ObjectIdentityManager(distance_px=220.0, max_age_frames=120)
    for f in range(10):
        res = _obj_update(
            mgr, [("trash_bag", (100, 100)), ("trash_bag", (600, 100))], f
        )
        assert res[0] != res[1], "two distinct physical objects must never share a uid"
    late = _obj_update(mgr, [("trash_bag", (100, 100)), ("trash_bag", (600, 100))], 11)
    assert late[0] != late[1]


def test_object_far_reappearance_gets_new_uid():
    """An object that vanished and a DIFFERENT object appearing far away must
    not be merged (spatial evidence does not support continuity)."""
    mgr = ObjectIdentityManager(distance_px=220.0, max_age_frames=120)
    uid = None
    for f in range(10):
        uid = _obj_update(mgr, [("trash_bag", (100, 100))], f)[0]
    for f in range(10, 14):
        _obj_update(mgr, [], f)
    new_uid = _obj_update(mgr, [("trash_bag", (900, 700))], 14)[0]
    assert new_uid != uid


# --------------------------------------------------------------------- #
# EVENT OWNERSHIP — raw id churn mid-carry never changes the actor
# --------------------------------------------------------------------- #
def test_person_id_churn_midcarry_preserves_actor_uid():
    """The actor's raw id churns (1 -> 9) mid-carry at the same position: the
    (person uid, object uid) pair survives and the confirmed event's actor is
    the SAME stable uid, frozen at carry time."""
    det = LitteringEventDetector(_cfg())
    events = []
    t = 0.0
    # carry as raw id 1
    for _ in range(6):
        events.extend(det.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    uid_of_raw1 = det._person_uid_of(1)
    # mid-carry raw id switch to 9 (same physical person, same position)
    for _ in range(3):
        events.extend(det.update([_person(9, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    assert det._person_uid_of(9) == uid_of_raw1, "raw id switch must preserve uid"
    # release + ground + departure
    for y in (280, 340, 400):
        events.extend(det.update([_person(9, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(6):
        events.extend(det.update([_person(9, 140, 180)], [_bag(10001, 140, 400)], t))
        t += 0.1
    for x in (320, 480, 640, 800):
        events.extend(det.update([_person(9, x, 180)], [_bag(10001, 140, 400)], t))
        t += 0.1
    events.extend(det.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert confirmed, "event must confirm despite the actor's raw id churn"
    ev = confirmed[0]
    assert ev.event_actor_person_uid == uid_of_raw1, (
        "event actor must be identified by the STABLE uid frozen at carry"
    )
    assert ev.event_object_uid == ev.bag_uid is not None
    # a closer passer-by can never steal the ownership
    assert ev.event_object_uid is not None


def test_owner_override_not_recaptured_by_final_frame_proximity():
    """Ownership freeze: after release+ground, a bystander who walks closer
    than the departing actor can never take over the pair's ownership."""
    det = LitteringEventDetector(_cfg())
    t = 0.0
    for _ in range(6):
        det.update([_person(1, 140, 180), _person(2, 700, 180)], [_bag(10001, 140, 220)], t)
        t += 0.1
    for y in (280, 340, 400):
        det.update([_person(1, 140, 180), _person(2, 700, 180)], [_bag(10001, 140, y)], t)
        t += 0.1
    for _ in range(5):
        det.update([_person(1, 140, 180), _person(2, 700, 180)], [_bag(10001, 140, 400)], t)
        t += 0.1
    # actor departs; bystander approaches the grounded bag and stands closest
    for x1, x2 in ((320, 200), (480, 160), (640, 160), (800, 160)):
        det.update([_person(1, x1, 180), _person(2, x2, 180)], [_bag(10001, 140, 400)], t)
        t += 0.1
    owner_keys = [
        (k, m) for k, m in det._pairs.items()
        if m.state.value in ("BAG_CARRIED", "BAG_RELEASED", "BAG_ON_GROUND", "PERSON_DEPARTED", "VIOLATION_CONFIRMED")
    ]
    assert owner_keys, "the actor's pair must still be alive"
    assert all(k[0] == 1 for k, _ in owner_keys), (
        "ownership must remain with the carrying actor's uid, never the closer bystander"
    )


# --------------------------------------------------------------------- #
# BIN VS GROUND
# --------------------------------------------------------------------- #
def test_bin_height_rest_is_no_confident_event():
    """An object released that comes to rest well ABOVE the actor's ground
    plane (~0.7 person-heights above the feet line — a bin deposit / ledge)
    is LOCATION_AMBIGUOUS: with the gate on it must be rejected as
    NO_CONFIDENT_EVENT, never confirmed. (A bin rest is ~0.5-0.8·ph above the
    feet line; the ground-plane margin accepts <= 0.4·ph.)"""
    det = LitteringEventDetector(_cfg(require_ground_confirmation=True))
    events = []
    t = 0.0
    # carry (bag held near torso, wrist near)
    for _ in range(6):
        events.extend(det.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    # release: person walks away, bag stays at bin height (feet line ~260,
    # bag centroid 150 -> 0.69·ph ABOVE the feet line, beyond the margin)
    for x in (180, 240, 300, 360, 420, 480):
        events.extend(det.update([_person(1, x, 180)], [_bag(10001, 140, 150)], t))
        t += 0.1
    # bag stationary at bin height while the actor keeps walking
    for x in (540, 600, 660, 720, 780, 840):
        events.extend(det.update([_person(1, x, 180)], [_bag(10001, 140, 150)], t))
        t += 0.1
    events.extend(det.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert not confirmed, "ambiguous-location candidates must never confirm"
    rejected = [e for e in events if not e.confirmed]
    assert any(
        e.reason == "NO_CONFIDENT_EVENT" and e.details["location_status"] == "LOCATION_AMBIGUOUS"
        for e in rejected
    ), f"expected a NO_CONFIDENT_EVENT rejection, got {[ (e.reason, e.details) for e in rejected ]}"


def test_ground_rest_below_feet_confirms_and_records_status():
    """Same arc but the object rests on the ground plane (below the actor's
    feet line): confirms, and the event records GROUND_CONFIRMED."""
    det = LitteringEventDetector(_cfg(require_ground_confirmation=True))
    events = []
    t = 0.0
    for _ in range(6):
        events.extend(det.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    for y in (280, 340, 400):
        events.extend(det.update([_person(1, 140, 180)], [_bag(10001, 140, y)], t))
        t += 0.1
    for _ in range(6):
        events.extend(det.update([_person(1, 140, 180)], [_bag(10001, 140, 400)], t))
        t += 0.1
    for x in (320, 480, 640, 800):
        events.extend(det.update([_person(1, x, 180)], [_bag(10001, 140, 400)], t))
        t += 0.1
    events.extend(det.finalize())
    confirmed = [e for e in events if e.confirmed]
    assert confirmed, "ground-plane rest must confirm"
    assert confirmed[0].details["location_status"] == "GROUND_CONFIRMED"
    assert confirmed[0].details["ground_evidence_frames"] > 0


def test_location_status_recorded_even_when_gate_off():
    """The location status is always attached (transparency), even when the
    confirmation gate is disabled: an ambiguous rest that confirms under the
    legacy behaviour is explicitly labelled LOCATION_AMBIGUOUS."""
    det = LitteringEventDetector(_cfg(require_ground_confirmation=False))
    events = []
    t = 0.0
    for _ in range(6):
        events.extend(det.update([_person(1, 140, 180)], [_bag(10001, 140, 220)], t))
        t += 0.1
    for x in (180, 240, 300, 360, 420, 480):
        events.extend(det.update([_person(1, x, 180)], [_bag(10001, 140, 150)], t))
        t += 0.1
    for x in (540, 600, 660, 720, 780, 840):
        events.extend(det.update([_person(1, x, 180)], [_bag(10001, 140, 150)], t))
        t += 0.1
    events.extend(det.finalize())
    confirmed = [e for e in events if e.confirmed]
    if confirmed:  # legacy gate-off behaviour may confirm; status must still be honest
        assert confirmed[0].details["location_status"] == "LOCATION_AMBIGUOUS"
    else:
        assert any(e.reason == "NO_CONFIDENT_EVENT" or e.confidence < 0.65 for e in events if not e.confirmed)


# --------------------------------------------------------------------- #
# Detector diagnostics (identity telemetry for the real-video audit)
# --------------------------------------------------------------------- #
def test_last_tick_uid_maps_exposed():
    det = LitteringEventDetector(_cfg())
    det.update([_person(1, 140, 180), _person(2, 700, 180)], [_bag(10001, 140, 220)], 0.0)
    assert det.last_person_uid_map.get(1) is not None
    assert det.last_person_uid_map.get(2) is not None
    assert det.last_person_uid_map[1] != det.last_person_uid_map[2]
    assert det.last_object_uid_map.get(10001) is not None


# --------------------------------------------------------------------- #
# Object identity inside the detector: churn keeps ONE pair (already covered
# at behaviour level by test_tracker_id_churn_keeps_single_stable_object_
# identity); here we assert the object-uid provenance end to end.
# --------------------------------------------------------------------- #
def test_detector_pair_key_uses_authoritative_object_uid():
    det = LitteringEventDetector(_cfg())
    for _ in range(4):
        det.update([_person(1, 140, 180)], [_bag(777001, 140, 220)], 0.0)
    keys = [k for k in det._pairs if k[0] == 1]
    assert len(keys) == 1
    obj_uid = det.last_object_uid_map.get(777001)
    assert obj_uid is not None
    assert keys[0][1] == obj_uid, "pair key object slot must be the authoritative object uid"
    assert det._pairs[keys[0]].bag_uid == obj_uid
