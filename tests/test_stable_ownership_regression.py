"""Regression: stable ownership fields must be correctly populated at _make_event.

Covers the lifecycle PersonIdentityManager -> person_uid -> PairMemory
-> BAG_CARRIED freeze -> _advance_pair -> _make_event -> to_dict.

Four cases:
  - normal single-person single-object
  - person raw-id switch mid-carry (same physical actor)
  - object raw-id switch mid-arc (same physical bag, tracker churn)
  - multi-person scene (bystander must not steal ownership)

Each asserts:
  * event_actor_person_uid is populated and equals the authoritative stable person uid
  * event_object_uid is populated and equals the authoritative stable object uid
  * carry_* raw track ids preserve the raw ids frozen at carry time
  * bag_uid equals event_object_uid (pair key identity)
  * no fallback copying of raw id into stable field when stable available
"""
from littering_event_detector import LitteringEventDetector, DetectorPerson, DetectorBag, EventDetectorConfig

def _cfg(**overrides):
    base = dict(
        auto_calibrate=False,
        pose_association_enabled=False,
        min_carried_frames=3,
        min_stationary_frames=3,
        min_departed_frames=2,
        min_abandonment_frames=3,
        confirmation_grace_frames=2,
        smoothing_window=3,
        stationary_window_frames=3,
        max_pair_age_frames=20,
        min_event_confidence=0.6,
        carry_motion_norm_px=2.0,
    )
    base.update(overrides)
    return EventDetectorConfig(**base)

def _person(pid, x, y, w=60, h=160):
    return DetectorPerson(track_id=pid, bbox=(x, y, x+w, y+h), confidence=1.0)

def _bag(bid, x, y, w=30, h=30, conf=0.9):
    return DetectorBag(track_id=bid, bbox=(x, y, x+w, y+h), confidence=conf, class_name="trash_bag", source="yolo", yolo_confirmed=True)

def _run(det, person_ids, bag_ids):
    t=0.0
    for f in range(len(person_ids)):
        pid=person_ids[f]; bid=bag_ids[f]
        if f < 10:
            px=200+f*2; py=200; bx=px+10; by=py+80
        elif f < 14:
            px=200+f*2; py=200; bx=400; by=380
        else:
            px=200+f*5; py=200; bx=400; by=380
        persons=[_person(pid, px, py)]
        bags=[_bag(bid, bx, by)]
        det.update(persons, bags, timestamp=t, frame_index=f)
        t+=0.125
    det.finalize()
    # inspector may call finalize; collect
    evs=[e for e in det.confirmed_events if e.confirmed]
    return evs

def _assert_stable(ev, expected_person_raw, expected_bag_raw_at_carry):
    d=ev.to_dict()
    # raw preservation
    assert d["person_track_id"] == expected_person_raw, f"raw person_track_id {d['person_track_id']} != {expected_person_raw}"
    # stable must be populated and distinct from raw when raw!=stable (raw 35 -> uid 1)
    assert d["event_actor_person_uid"] is not None, "event_actor_person_uid must not be None"
    assert d["event_object_uid"] is not None, "event_object_uid must not be None"
    assert d["bag_uid"] is not None, "bag_uid must not be None"
    assert d["event_actor_person_track_id"] is not None
    assert d["event_object_track_id"] is not None
    # authoritative: stable belongs to same physical entity, not invented
    # person stable comes from PersonIdentityManager
    assert isinstance(d["event_actor_person_uid"], int) and d["event_actor_person_uid"] > 0
    assert isinstance(d["event_object_uid"], int) and d["event_object_uid"] >= 100001
    # carry-frozen raw ids must equal raw at carry time, not later churned ids
    assert d["event_actor_person_track_id"] == expected_person_raw
    assert d["event_object_track_id"] == expected_bag_raw_at_carry
    # bag_uid must equal event_object_uid (same logical object)
    assert d["bag_uid"] == d["event_object_uid"]
    # stable person uid must NOT be a copy of raw when they differ
    # Use raw 35 -> stable 1, so they differ: ensure stable != raw
    if expected_person_raw != d["event_actor_person_uid"]:
        assert d["event_actor_person_uid"] != expected_person_raw or expected_person_raw==d["event_actor_person_uid"] # dummy, allow equality for tiny raw=1 case but here 35!=1 so must hold
        assert d["event_actor_person_uid"] != expected_person_raw, "stable uid must not be raw id copy"

def test_normal_event_stable_ownership():
    det=LitteringEventDetector(_cfg())
    evs=_run(det, [35]*22, [110023]*22)
    assert evs, "normal event must confirm"
    _assert_stable(evs[0], 35, 110023)
    # prove object uid is authoritative from ObjectIdentityManager (100001 range)
    assert evs[0].to_dict()["event_object_uid"] >= 100001

def test_person_id_switch_preserves_stable_actor():
    det=LitteringEventDetector(_cfg())
    # switch raw 35 -> 36 mid-carry, same position => same stable uid
    evs=_run(det, [35]*6 + [36]*16, [110023]*22)
    assert evs
    ev=evs[0]
    d=ev.to_dict()
    # stable uid frozen at carry (raw 35 era) must still be uid 1
    assert d["event_actor_person_uid"] == 1
    assert d["event_actor_person_track_id"] == 35  # frozen raw at carry, not 36
    # verify manager maps both raws to same uid
    assert det._person_uid_of(35) == det._person_uid_of(36) == 1

def test_object_id_switch_preserves_stable_object():
    det=LitteringEventDetector(_cfg())
    # same physical bag, tracker churns raw 110023 -> 110138
    evs=_run(det, [35]*22, [110023]*8 + [110138]*14)
    assert evs
    ev=evs[0]
    d=ev.to_dict()
    # stable object uid must be one logical id across churn
    assert d["event_object_uid"] is not None and d["bag_uid"] is not None
    assert d["event_object_uid"] == d["bag_uid"]
    # raw frozen at carry is 110023, not later 110138
    assert d["event_object_track_id"] == 110023
    assert d["bag_track_id"] == 110023
    # object uid should still be in authoritative range, not per-person fallback
    assert d["event_object_uid"] >= 100001

def test_multi_person_bystander_cannot_steal():
    det=LitteringEventDetector(_cfg())
    t=0.0
    for f in range(22):
        if f < 10:
            px=200+f*2; py=200; bx=px+10; by=py+80
        elif f < 14:
            px=200+f*2; py=200; bx=400; by=380
        else:
            px=200+f*5; py=200; bx=400; by=380
        persons=[_person(35, px, py), _person(99, 410, 210)]
        bags=[_bag(110023, bx, by)]
        det.update(persons, bags, timestamp=t, frame_index=f)
        t+=0.125
    det.finalize()
    evs=[e for e in det.confirmed_events if e.confirmed]
    assert evs
    d=evs[0].to_dict()
    assert d["event_actor_person_uid"] == det._person_uid_of(35)
    assert d["event_actor_person_uid"] != det._person_uid_of(99)
    _assert_stable(evs[0], 35, 110023)
