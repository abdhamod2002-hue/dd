"""Section 4 — track confirmation + conf floors for visual noise."""
from __future__ import annotations

from inference.detection.yolo_detector import TrackedDetection, YoloDetector
from inference.tracking.bytetrack_tracker import BytetrackTracker


def _td(tid: int, person: bool = True, conf: float = 0.9):
    return TrackedDetection(
        track_id=tid,
        class_name="person" if person else "Garbage Bag",
        confidence=conf,
        bbox=(10.0, 10.0, 50.0, 80.0),
        centroid=(30.0, 45.0),
        is_person=person,
    )


def test_yolo_default_conf_floors_are_point_four():
    det = YoloDetector(litter_weights="nonexistent.pt", color_fallback=False)
    assert det.person_conf == 0.4
    assert det.litter_conf == 0.4
    assert det.bag_conf == 0.4
    assert det.iou == 0.5


def test_tentative_tracks_excluded_until_confirm_hits():
    tr = BytetrackTracker(min_confirm_frames=3)
    # First 2 observations: still tentative
    for i in range(2):
        tr.update([_td(7)], frame_index=i)
        persons, objects = tr.to_tracks([_td(7)])
        assert persons == []
        assert objects == []
        assert tr.is_confirmed(7) is False
    # 3rd hit confirms
    tr.update([_td(7)], frame_index=2)
    persons, objects = tr.to_tracks([_td(7)])
    assert len(persons) == 1
    assert persons[0].track_id == 7
    assert tr.is_confirmed(7) is True


def test_proposals_hidden_by_default(monkeypatch):
    from inference.visualization.supervision_annotator import _draw_proposals_enabled

    monkeypatch.delenv("MOTARED_DRAW_PROPOSALS", raising=False)
    assert _draw_proposals_enabled() is False
    monkeypatch.setenv("MOTARED_DRAW_PROPOSALS", "1")
    assert _draw_proposals_enabled() is True
