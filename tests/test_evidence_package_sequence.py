"""P1-2 regression tests: carry/release/ground sequence stills.

MASTER_REPAIR_PLAN P1-2: the sequence images were orphaned artifacts — written
(or silently not written) by ``write_event_evidence_package`` but never stored
in DB columns, served, or rendered. These tests pin the writer side:

1. when both actor and object bboxes are present at the recorded FSM frames,
   carry.jpg / release.jpg / ground.jpg are actually written and their paths
   are returned (so the API layer can persist them);
2. when a bbox is missing, the failure is LOGGED, not silent (operator
   visibility into how often sequence images fail).
"""

from __future__ import annotations

import logging

import cv2
import pytest

from inference.visualization.evidence_package import write_event_evidence_package


@pytest.fixture()
def tiny_video(tmp_path):
    """A 10-frame 320x240 mp4v video (browser playability not under test)."""
    import numpy as np

    path = tmp_path / "src.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    assert writer.isOpened()
    for i in range(10):
        frame = np.full((240, 320, 3), 60 + i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


def _frame_records():
    """Frames 4/5/6 all contain the actor (track 7) and object (uid 100004)."""
    person = {"track_id": 7, "bbox": [40, 60, 120, 220]}
    obj = {"track_id": 9001, "object_uid": 100004, "bbox": [150, 180, 210, 230],
           "class_name": "yellow_waste_bag"}
    return [
        {"frame_number": f, "timestamp": f / 10.0,
         "persons": [dict(person)], "objects": [dict(obj)]}
        for f in (4, 5, 6)
    ]


def _event():
    return {
        "event_id": "p1_2_test",
        "person_track_id": 7,
        "bag_track_id": 9001,
        "event_actor_person_track_id": 7,
        "event_object_track_id": 9001,
        "event_object_uid": 100004,
        "frames": {"carry_start": 4, "release": 5, "ground": 6, "confirmed": 6},
        "timestamps": {"carry_start": 0.4, "confirmed": 0.6},
        "object_type": "yellow_waste_bag",
        "confidence": 0.9,
    }


def test_sequence_images_written_when_bboxes_present(tiny_video, tmp_path):
    pkg = write_event_evidence_package(
        original_video_path=tiny_video,
        target_dir=str(tmp_path / "ev"),
        event=_event(),
        frame_records=_frame_records(),
        job_id=1,
        original_filename="unit.mp4",
        source_fps=10.0,
        pre_seconds=0.1,
        post_seconds=0.1,
        analyzed_video_path=tiny_video,
    )
    for key in ("carry", "release", "ground"):
        path = pkg.get(key)
        assert path, f"{key} sequence image was not returned by the package writer"
        assert tmp_path.joinpath("ev", f"{key}.jpg").exists(), f"{key}.jpg not written"
    metadata = (tmp_path / "ev" / "metadata.json")
    assert metadata.exists()
    import json

    files = json.loads(metadata.read_text(encoding="utf-8"))["files"]
    for key in ("carry", "release", "ground"):
        assert files[key], f"metadata.files.{key} is None"


def test_missing_bbox_is_logged_not_silent(tiny_video, tmp_path, caplog):
    records = _frame_records()
    # remove the object from the release frame -> release.jpg must fail
    records[1]["objects"] = []
    with caplog.at_level(logging.INFO, logger="inference.visualization.evidence_package"):
        pkg = write_event_evidence_package(
            original_video_path=tiny_video,
            target_dir=str(tmp_path / "ev2"),
            event=_event(),
            frame_records=records,
            job_id=1,
            original_filename="unit.mp4",
            source_fps=10.0,
            pre_seconds=0.1,
            post_seconds=0.1,
            analyzed_video_path=tiny_video,
        )
    assert pkg.get("release") is None
    assert any("RELEASE" in r.getMessage() or "sequence image" in r.getMessage()
               for r in caplog.records), "missing-bbox failure was silent"
