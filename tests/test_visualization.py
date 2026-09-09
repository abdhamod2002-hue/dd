"""Tests for the visualization data contract and renderer.

These tests verify that the visual layer consumes real production objects
and does not change frame dimensions or invent entities.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from inference.association.person_object_assoc import Keypoints, Track
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.visualization import build_frame_analysis, render_analysis_frame
from inference.visualization.evidence_package import (
    _find_object,
    write_event_evidence_package,
)


class _State:
    def __init__(self, value: str):
        self.value = value


class _Pair:
    def __init__(self, person_id: int, bag_id: int, state: str):
        self.person_id = person_id
        self.bag_id = bag_id
        self.state = _State(state)

    def mean_association_score(self) -> float:
        return 0.8


class _Detector:
    def __init__(self, pairs):
        self._pairs = pairs


def _person(track_id: int = 1, bbox=(100, 100, 220, 360), conf: float = 0.91) -> Track:
    kp = Keypoints(
        left_wrist=(150, 220),
        right_wrist=(170, 225),
        left_shoulder=(140, 160),
        right_shoulder=(180, 160),
        torso_center=(160, 160),
        nose=(160, 130),
        nose_confidence=0.82,
    )
    return Track(track_id=track_id, class_name="person", centroid=(160, 230), bbox=bbox, keypoints=kp, confidence=conf)


def _bag(track_id: int = 60003, bbox=(150, 210, 190, 250), conf: float = 0.77) -> Track:
    return Track(track_id=track_id, class_name="yellow_waste_bag", centroid=(170, 230), bbox=bbox, confidence=conf, source="color")


def test_build_frame_analysis_uses_real_tracks_and_detector_state():
    tracker = BytetrackTracker()
    from inference.detection.yolo_detector import TrackedDetection

    tracked = [
        TrackedDetection(track_id=1, class_name="person", confidence=0.91, bbox=(100, 100, 220, 360), centroid=(160, 230), is_person=True),
        TrackedDetection(track_id=50003, class_name="yellow_waste_bag", confidence=0.77, bbox=(150, 210, 190, 250), centroid=(170, 230), is_person=False, source="color"),
    ]
    tracker.update(tracked, frame_index=10)
    detector = _Detector({(1, 60003): _Pair(1, 60003, "BAG_CARRIED")})

    analysis = build_frame_analysis(
        [_person()],
        [_bag()],
        detector,
        tracker,
        timestamp=1.25,
        frame_number=10,
        source_fps=59.9,
        analysis_fps=8.0,
        video_name="IMG_5117.MOV",
    )

    assert analysis.frame_number == 10
    assert len(analysis.persons) == 1
    assert len(analysis.objects) == 1
    assert analysis.persons[0].track_id == 1
    assert analysis.objects[0].track_id == 60003
    assert analysis.objects[0].source == "color"
    assert analysis.persons[0].head_visible is True
    assert analysis.persons[0].trail
    assert analysis.states == ["BAG_CARRIED"]
    assert analysis.associations[0].state == "BAG_CARRIED"
    assert analysis.hud["persons"] == 1
    assert analysis.hud["objects"] == 1


def test_render_analysis_frame_preserves_dimensions_and_is_copy():
    import cv2
    import numpy as np

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    original = frame.copy()
    analysis = build_frame_analysis(
        [_person()],
        [_bag()],
        _Detector({(1, 60003): _Pair(1, 60003, "BAG_CARRIED")}),
        None,
        timestamp=2.0,
        frame_number=20,
        source_fps=30.0,
        analysis_fps=8.0,
        video_name="test.mp4",
    )
    rendered = render_analysis_frame(frame, analysis)

    assert rendered.shape == original.shape
    assert np.array_equal(frame, original)
    assert rendered.sum() > original.sum()


def test_render_analysis_frame_supports_portrait_dimensions():
    import numpy as np

    frame = np.zeros((1920, 1080, 3), dtype=np.uint8)
    analysis = build_frame_analysis([_person()], [_bag()], _Detector({}), None, 1.0, 1)
    rendered = render_analysis_frame(frame, analysis)
    assert rendered.shape == (1920, 1080, 3)


def test_analysis_markers_use_real_timestamps():
    from backend.routers.analysis import _build_markers

    markers = _build_markers(
        first_timestamp=100.0,
        person_first_seen={1: (10, 100.5)},
        object_first_seen={60003: (20, 101.0)},
        confirmed_event_dicts={
            "abc": {
                "event_id": "abc",
                "person_track_id": 1,
                "bag_track_id": 60003,
                "frames": {"carry_start": 30, "release": 50, "ground": 70, "departure": 90, "confirmed": 90},
                "timestamps": {"carry_start": 101.5, "release": 102.5, "ground": 103.5, "departure": 104.5, "confirmed": 104.5},
            }
        },
    )
    labels = [m["label"] for m in markers]
    assert "PERSON #1 FIRST SEEN" in labels
    assert "OBJECT #60003 FIRST SEEN" in labels
    assert "RELEASE" in labels
    assert "GROUND" in labels
    assert "DEPARTURE" in labels
    assert "EVENT" in labels
    release = next(m for m in markers if m["label"] == "RELEASE")
    assert release["timestamp"] == pytest.approx(2.5)
    assert release["frame"] == 50


def test_evidence_package_writes_real_crops_and_clip(tmp_path):
    import cv2
    import numpy as np

    analyzed = tmp_path / "analyzed.mp4"
    h, w = 480, 640
    writer = cv2.VideoWriter(str(analyzed), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened()
    for i in range(90):
        frame = np.full((h, w, 3), 30 + i, dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (220, 360), (0, 255, 0), 3)
        cv2.rectangle(frame, (150, 210), (190, 250), (0, 255, 255), 3)
        writer.write(frame)
    writer.release()

    records = []
    for i in range(90):
        records.append({
            "frame_number": i,
            "timestamp": i / 30.0,
            "persons": [{"track_id": 1, "bbox": [100, 100, 220, 360]}],
            "objects": [{"track_id": 60003, "bbox": [150, 210, 190, 250]}],
        })

    event = {
        "event_id": "abc",
        "person_track_id": 1,
        "bag_track_id": 60003,
        "frames": {"carry_start": 10, "release": 30, "ground": 45, "departure": 60, "confirmed": 60},
        "timestamps": {"carry_start": 10 / 30.0, "release": 30 / 30.0, "ground": 45 / 30.0, "departure": 60 / 30.0, "confirmed": 60 / 30.0},
        "confidence": 0.95,
    }
    out_dir = tmp_path / "event"
    pkg = write_event_evidence_package(
        analyzed_video_path=str(analyzed),
        target_dir=str(out_dir),
        event=event,
        frame_records=records,
        job_id=1,
        original_filename="test.mp4",
        source_fps=30.0,
        pre_seconds=1.0,
        post_seconds=1.0,
    )

    assert pkg["snapshot"] and os.path.exists(pkg["snapshot"])
    assert pkg["person"] and os.path.exists(pkg["person"])
    assert pkg["waste"] and os.path.exists(pkg["waste"])
    assert pkg["clip"] and os.path.exists(pkg["clip"])
    meta_path = tmp_path / "event" / "metadata.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["event"]["person_track_id"] == 1
    assert meta["event"]["bag_track_id"] == 60003


def test_find_object_prefers_stable_uid_over_churned_track_id():
    """Phase C/event-centric: when a physical bag is reborn under many tracker
    ids, evidence must locate it by the STABLE object_uid, not the churned id.

    At the GROUND frame the bag appears under a NEW churned id (99999) while the
    frozen event_object_track_id (60003) is no longer present; resolving by the
    stable uid=1 must still find it.
    """
    ground_record = {
        "objects": [
            {"track_id": 99999, "object_uid": 1, "bbox": [100, 100, 120, 120]},
            {"track_id": 55555, "object_uid": 2, "bbox": [200, 200, 220, 220]},  # a DIFFERENT bag
        ]
    }
    # event_object_track_id is stale (60003) but event_object_uid is stable (1)
    found = _find_object(ground_record, 60003, 1)
    assert found is not None
    assert found["track_id"] == 99999, "must resolve to the CURRENT churned id via uid, not the stale one"
    assert found["object_uid"] == 1
    # a DIFFERENT bag (uid 2) is never confused with the event object (uid 1)
    other = _find_object(ground_record, 55555, 2)
    assert other is not None and other["object_uid"] == 2 and other["track_id"] == 55555


def test_evidence_anchors_to_stable_uid_across_churn(tmp_path):
    """End-to-end event-centric anchoring: the object track id churns between the
    carry frame and the ground frame, but a stable object_uid ties them together
    so the waste crop is taken from the GROUND frame (the explanatory still), not
    from a stale/incorrect id."""
    import cv2
    import numpy as np

    analyzed = tmp_path / "analyzed.mp4"
    h, w = 480, 640
    writer = cv2.VideoWriter(str(analyzed), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(90):
        frame = np.full((h, w, 3), 30 + i, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    # carry frame (10): object id 60003, uid 1
    # ground frame (45): object id CHURNED to 99999, uid 1 (same physical bag)
    records = []
    for i in range(90):
        obj = None
        if i == 10:
            obj = {"track_id": 60003, "object_uid": 1, "bbox": [150, 210, 190, 250]}
        elif i == 45:
            obj = {"track_id": 99999, "object_uid": 1, "bbox": [300, 300, 340, 360]}
        records.append({
            "frame_number": i,
            "timestamp": i / 30.0,
            "persons": [{"track_id": 1, "bbox": [100, 100, 220, 360]}],
            "objects": [obj] if obj else [],
        })

    event = {
        "event_id": "abc",
        "confirmed": True,
        "person_track_id": 1,
        "bag_track_id": 60003,
        # Phase C authoritative ids: frozen carry id + stable uid
        "event_actor_person_track_id": 1,
        "event_object_track_id": 60003,
        "event_object_uid": 1,
        "frames": {"carry_start": 10, "release": 30, "ground": 45, "departure": 60, "confirmed": 60},
        "timestamps": {"carry_start": 10 / 30.0, "release": 30 / 30.0, "ground": 45 / 30.0, "departure": 60 / 30.0, "confirmed": 60 / 30.0},
        "confidence": 0.95,
    }
    out_dir = tmp_path / "event"
    pkg = write_event_evidence_package(
        analyzed_video_path=str(analyzed),
        target_dir=str(out_dir),
        event=event,
        frame_records=records,
        job_id=1,
        original_filename="test.mp4",
        source_fps=30.0,
        pre_seconds=1.0,
        post_seconds=1.0,
    )
    assert pkg["waste"] and os.path.exists(pkg["waste"])
    meta = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    # The best explanatory frame must be the GROUND frame (45), and the object
    # resolved there must be the uid-1 object (track 99999 at that frame), proving
    # the evidence anchored by stable uid rather than the stale event_object_track_id.
    assert meta["snapshot_frame_number"] == 45
    best_obj = _find_object(meta["best_frame_record"], 60003, 1)
    assert best_obj is not None and best_obj["track_id"] == 99999 and best_obj["object_uid"] == 1
