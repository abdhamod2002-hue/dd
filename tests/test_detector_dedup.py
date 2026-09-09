"""
Phase 3 Step 10 — detector-dedup regression tests (no camera, no models).

Covers the 7 required cases:
1. same object from two YOLO models → one detection
2. same object from same model → one detection
3. two distinct nearby objects → remain two
4. overlapping people → remain separate
5. semantic YOLO waste + HSV proposal → semantic wins
6. novelty proposal never becomes semantic waste
7. dedup does not change event logic (idempotent, field-preserving,
   person/object groups never merged)
"""

from __future__ import annotations

import pytest

from inference.detection.yolo_detector import (
    Detection,
    TrackedDetection,
    boxes_duplicate,
    deduplicate_detections,
    deduplicate_tracked,
)


def _det(cls, conf, bbox, is_person=None):
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return Detection(cls, conf, tuple(bbox), (cx, cy))


def _trk(tid, cls, conf, bbox, is_person, source="yolo"):
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return TrackedDetection(tid, cls, conf, tuple(bbox), (cx, cy),
                            is_person, source)


# Real duplicate pair: IMG_5306 src=4816 (two Garbage-Bag boxes, IoU=0.21,
# big holds 89.8% of small's area; small has HIGHER confidence).
BAG_SMALL = [300.7, 1097.5, 478.4, 1286.1]   # conf 0.2723
BAG_BIG = [318.9, 1091.9, 617.7, 1552.5]     # conf 0.2582


def test_1_cross_model_same_bag_collapses_to_one():
    a = _trk(3, "bottle", 0.45, (100, 100, 200, 220), False, "yolo")
    b = _trk(7, "Garbage Bag", 0.31, (105, 105, 195, 215), False, "yolo")
    out = deduplicate_tracked([a, b])
    assert len(out) == 1
    assert out[0].confidence == pytest.approx(0.45)  # higher conf survives


def test_2_same_model_partial_overlap_collapses_to_one():
    # The exact src=4816 failure the old asymmetric rule missed.
    a = _trk(11, "Garbage Bag", 0.2723, BAG_SMALL, False, "yolo")
    b = _trk(12, "Garbage Bag", 0.2582, BAG_BIG, False, "yolo")
    assert boxes_duplicate(tuple(BAG_SMALL), tuple(BAG_BIG))
    out = deduplicate_tracked([a, b])
    assert len(out) == 1
    assert out[0].track_id == 11
    # Order-independent: big-first input gives the same survivor.
    out2 = deduplicate_tracked([b, a])
    assert len(out2) == 1 and out2[0].track_id == 11


def test_3_two_distinct_nearby_objects_remain_two():
    a = _trk(1, "Garbage Bag", 0.30, (100, 100, 160, 170), False, "yolo")
    b = _trk(2, "Garbage Bag", 0.29, (200, 110, 260, 180), False, "yolo")
    assert not boxes_duplicate(a.bbox, b.bbox)
    out = deduplicate_tracked([a, b])
    assert len(out) == 2


def test_4_overlapping_people_remain_separate():
    # One person behind another: IoU ~0.3, containment both ways < 0.7.
    a = _trk(1, "person", 0.80, (500, 900, 620, 1300), True, "yolo")
    b = _trk(2, "person", 0.75, (560, 920, 680, 1320), True, "yolo")
    assert not boxes_duplicate(a.bbox, b.bbox)
    out = deduplicate_tracked([a, b])
    assert len(out) == 2
    # Person-vs-object overlap is NEVER a duplicate.
    bag = _trk(3, "Garbage Bag", 0.60, (540, 910, 660, 1310), False, "yolo")
    out2 = deduplicate_tracked([a, bag])
    assert len(out2) == 2


def test_5_semantic_yolo_beats_hsv_proposal_even_at_lower_conf():
    yolo = _trk(5, "Garbage Bag", 0.30, (100, 100, 200, 220), False, "yolo")
    hsv = _trk(50001, "waste_bag", 0.90, (102, 102, 198, 218), False, "color")
    out = deduplicate_tracked([hsv, yolo])  # hsv first: must still lose
    assert len(out) == 1
    assert out[0].source == "yolo"
    assert out[0].class_name == "Garbage Bag"


def test_6_novelty_never_overrides_semantic_waste():
    yolo = _trk(5, "Garbage Bag", 0.28, (100, 100, 200, 220), False, "yolo")
    nov = _trk(100001, "detected_object", 0.95, (103, 103, 197, 217),
               False, "novelty")
    out = deduplicate_tracked([nov, yolo])
    assert len(out) == 1
    assert out[0].source == "yolo"
    # Two overlapping novelty proposals still dedup among themselves.
    nov2 = _trk(100002, "detected_object", 0.80, (104, 104, 196, 216),
                False, "novelty")
    out2 = deduplicate_tracked([nov, nov2])
    assert len(out2) == 1


def test_8_overlapping_color_proposals_collapse_among_themselves():
    # The color tracker emits every established track each frame; overlapping
    # fragments on the same pile must collapse (persons never affected).
    a = _trk(50001, "color_candidate_yellow", 0.60, (100, 100, 200, 220),
             False, "color")
    b = _trk(50002, "color_candidate_yellow", 0.55, (105, 105, 195, 215),
             False, "color")
    p = _trk(1, "person", 0.80, (100, 100, 200, 220), True, "yolo")
    out = deduplicate_tracked([a, b, p])
    assert len(out) == 2  # one color survivor + the person
    assert sum(1 for t in out if t.source == "color") == 1
    assert any(t.is_person for t in out)


def test_7_dedup_is_event_logic_neutral():
    # Idempotent: second pass changes nothing (stable tracker input).
    dets = [
        _trk(1, "person", 0.8, (500, 900, 620, 1300), True),
        _trk(2, "Garbage Bag", 0.30, (540, 910, 600, 1000), False),
        _trk(3, "Garbage Bag", 0.26, (542, 912, 598, 998), False),
    ]
    once = deduplicate_tracked(dets)
    twice = deduplicate_tracked(once)
    assert [t.track_id for t in once] == [t.track_id for t in twice]
    # Survivors keep every field byte-identical (no silent rewrites that
    # could perturb identity/ownership downstream).
    for t in once:
        src = next(d for d in dets if d.track_id == t.track_id)
        assert (t.bbox, t.confidence, t.class_name, t.source,
                t.is_person) == (src.bbox, src.confidence, src.class_name,
                                  src.source, src.is_person)
    # Detection-level entry point behaves the same.
    dd = [_det("person", 0.8, (500, 900, 620, 1300)),
          _det("Garbage Bag", 0.30, (540, 910, 600, 1000)),
          _det("Garbage Bag", 0.26, (542, 912, 598, 998))]
    assert len(deduplicate_detections(dd)) == 2
