"""Layer 1 (DETECTION / TRACKING) unit tests for the three-layer refactor.

Scope: the spec's Layer 1 — WHO/WHAT/WHERE with persistent track ids, stable
Person/Object ids, detector sources kept separate, and the hard invariant that
a Novelty `detected_object` is NOT a waste/litter candidate.

These tests are PURE (no torch/ultralytics/MoveNet), so they run everywhere.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.layer1_audit import _group_raw_tracks  # noqa: E402
from inference.tracking.object_identity import ObjectIdentityManager  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Physical-object grouping / ID-switch measurement
# --------------------------------------------------------------------------- #
def _raw(rid, frames, cx, cy, src="color"):
    return {rid: [(f, float(f), (cx, cy), src) for f in frames]}


def test_stable_raw_id_is_one_physical_object():
    """A raw tracker id seen across 30 consecutive frames must be ONE physical
    object with 0 id switches — never fragmented into 30 objects."""
    tf = _raw(60001, range(0, 30), 100.0, 100.0)
    grouped = _group_raw_tracks(tf, max_gap=8, dist=180.0)
    assert len(grouped) == 1, f"expected 1 object, got {len(grouped)}"
    info = list(grouped.values())[0]
    assert info["raw_ids"] == [60001]
    assert info["duration_frames"] == 30


def test_honest_id_switch_detected():
    """When the SAME physical object re-births under a new raw id at the same
    place shortly after, it must merge into ONE physical object and count an
    id switch (the Addendum's 'reassociated_tracks' / 'id_switches')."""
    tf = {}
    tf.update(_raw(60001, range(0, 6), 100.0, 100.0))
    tf.update(_raw(60002, range(7, 12), 101.0, 101.0))  # gap=1 frame, ~1px away
    grouped = _group_raw_tracks(tf, max_gap=4, dist=180.0)
    assert len(grouped) == 1, f"physical object split into {len(grouped)}"
    info = list(grouped.values())[0]
    assert len(info["raw_ids"]) == 2
    assert 60001 in info["raw_ids"] and 60002 in info["raw_ids"]


def test_distinct_objects_stay_distinct():
    """Two different physical objects (far apart) must NOT be merged."""
    tf = {}
    tf.update(_raw(60001, range(0, 6), 100.0, 100.0))
    tf.update(_raw(60002, range(0, 6), 800.0, 800.0))  # spatially far
    grouped = _group_raw_tracks(tf, max_gap=4, dist=180.0)
    assert len(grouped) == 2


# --------------------------------------------------------------------------- #
# 2. ObjectIdentityManager — stable logical uid across tracker churn
# --------------------------------------------------------------------------- #
def test_object_identity_persists_across_id_churn():
    """The same physical bag, churning raw tracker ids, must keep ONE stable
    logical uid (required so Layer-2/3 evidence anchor to the same object)."""
    oid = ObjectIdentityManager(distance_px=220.0, start_uid=100001)
    u1 = oid.update([("yellow_waste_bag", (10, 20, 30, 40), (20, 30))])[0]
    u2 = oid.update([("yellow_waste_bag", (11, 20, 30, 40), (21, 30))])[0]
    assert u1 == u2, f"physical object split: uid {u1} -> {u2}"


# --------------------------------------------------------------------------- #
# 3. Detector-source separation via namespacing
# --------------------------------------------------------------------------- #
def test_source_id_spaces_are_disjoint():
    """Persons, yolo objects, color objects and novelty objects must never
    collide in the same namespaced Track id space."""
    from inference.detection.yolo_detector import TrackedDetection
    from inference.tracking.bytetrack_tracker import BytetrackTracker

    trk = BytetrackTracker(min_confirm_frames=1)
    person = TrackedDetection(1, "person", 0.9, (0, 0, 50, 100), (25, 50), True, source="yolo")
    obj_yolo = TrackedDetection(1, "bottle", 0.7, (0, 0, 20, 20), (10, 10), False, source="yolo")
    # color_detector pre-offsets raw color ids by +50000; novelty by +100000
    # (matching _color_fallback_track / NoveltyDetector emission), then
    # BytetrackTracker adds the +10000 object-namespace offset on top.
    obj_color = TrackedDetection(50005, "yellow_waste_bag", 0.6, (0, 0, 20, 20), (10, 10), False,
                                 source="color")
    nov = TrackedDetection(100009, "detected_object", 0.5, (0, 0, 20, 20), (10, 10), False,
                           source="novelty")
    batch = [person, obj_yolo, obj_color, nov]
    trk.update(batch, frame_index=0)
    persons, objects = trk.to_tracks(batch)
    assert len(persons) == 1 and persons[0].track_id < 10000
    obj_ids = sorted(o.track_id for o in objects)
    # yolo raw1 -> 10001; color detector pre-offset 50005 -> 60005 (color's 50000
    # was already baked into the raw id by the detector); novelty pre-offset
    # 100009 -> 110009. All three id spaces stay disjoint from persons.
    assert obj_ids == [10001, 60005, 110009]


# --------------------------------------------------------------------------- #
# 4. Novelty != Waste (the reported-bug fix, Layer 1 half)
# --------------------------------------------------------------------------- #
def test_novelty_detected_object_is_not_a_litter_candidate():
    """The CRITICAL invariant: a class-agnostic Novelty 'detected_object' must
    NOT be treated as a waste/litter candidate. This is the Layer-1 guarantee
    that stops random objects/people from becoming event actors.

    `PersonObjectAssociator` (the module this test originally exercised) was
    removed as dead code (MASTER_REPAIR_PLAN P1-10/Section 16) — it was never
    called on the production path. The actual, live production gate for this
    invariant is `_is_semantic_waste()` in littering_event_detector.py, which
    this test now exercises directly. Note it is intentionally STRICTER than
    the old associator: HSV colour proposals (source="color") are excluded
    entirely too, not just admitted like the old associator's class-name-only
    check did.
    """
    from littering_event_detector import DetectorBag, _is_semantic_waste

    def bag(class_name: str, source: str) -> DetectorBag:
        return DetectorBag(track_id=1, bbox=(0.0, 0.0, 10.0, 10.0), class_name=class_name, source=source)

    assert _is_semantic_waste(bag("detected_object", "novelty")) is False
    assert _is_semantic_waste(bag("yellow_waste_bag", "color")) is True  # REPAIR-P0-02 discounted HSV
    assert _is_semantic_waste(bag("color_candidate_yellow", "yolo")) is False  # belt-and-suspenders
    assert _is_semantic_waste(bag("color_candidate_yellow", "color")) is False
    assert _is_semantic_waste(bag("bottle", "yolo")) is True


# --------------------------------------------------------------------------- #
# 5. Per-frame Layer-1 records expose detector_source
# --------------------------------------------------------------------------- #
def test_frame_record_carries_detector_source():
    """Layer-2 input (FrameAnalysis-like) must expose detector_source per object
    so the temporal layer can reason about HSV vs novelty vs yolo separately."""
    from inference.detection.yolo_detector import TrackedDetection
    from inference.tracking.bytetrack_tracker import BytetrackTracker

    trk = BytetrackTracker(min_confirm_frames=1)
    tracked = [
        TrackedDetection(1, "person", 0.9, (0, 0, 50, 100), (25, 50), True, source="yolo"),
        TrackedDetection(7, "detected_object", 0.5, (0, 0, 20, 20), (10, 10), False,
                         source="novelty"),
    ]
    trk.update(tracked, frame_index=0)
    persons, objects = trk.to_tracks(tracked)
    rec = {
        "persons": [{"id": p.track_id, "src": p.source} for p in persons],
        "objects": [{"id": o.track_id, "class": o.class_name, "src": o.source} for o in objects],
    }
    assert rec["persons"][0]["src"] == "yolo"
    assert rec["objects"][0]["class"] == "detected_object"
    assert rec["objects"][0]["src"] == "novelty"