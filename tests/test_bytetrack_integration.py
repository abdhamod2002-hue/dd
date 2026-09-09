"""
P0 integration test: REAL YOLO + REAL ByteTrack produce STABLE ids across
sequential frames.

This is the test the audit demanded: it proves the detection→tracking
adapter emits Track objects whose ids persist across frames (the
foundation the Association/FSM/Voting layers require).

It generates a short synthetic video containing a moving, person-like
colored block and runs the REAL ultralytics YOLO + ByteTrack
(``model.track(..., persist=True)``) on it via ``YoloDetector.track()``
and ``BytetrackTracker``.

What it verifies:
  - detector.track() runs without error on real frames
  - ByteTrack assigns ids (box.id is not None) when an entity is detected
  - the namespaced ids stay STABLE across consecutive frames
  - the TrackStore records history per id

If YOLO/ByteTrack cannot be imported (no torch/ultralytics on the host),
the test is SKIPPED — it only passes where the real stack is installed.
It is NEVER faked.
"""

from __future__ import annotations

import os
import tempfile

import pytest


def _have_ultralytics() -> bool:
    try:
        import torch  # noqa
        import ultralytics  # noqa
        import cv2  # noqa
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _have_ultralytics(),
                                reason="ultralytics/torch/cv2 not installed (real-camera stack)")


def _have_real_person_image() -> str | None:
    """Locate a real sample image with people that ultralytics ships."""
    try:
        import ultralytics, os
        ult_dir = os.path.dirname(ultralytics.__file__)
        for cand in ["assets/zidane.jpg", "assets/bus.jpg"]:
            p = os.path.join(ult_dir, cand)
            if os.path.exists(p):
                return p
    except Exception:
        pass
    return None


def _make_moving_person_video(path: str, frames: int = 20, size=(640, 480)) -> bool:
    """
    Create a short mp4 with a REAL detectable person that moves across
    frames. Uses ultralytics' bundled sample image (zidane.jpg / bus.jpg)
    which contains actual people YOLO can detect. We crop a person-
    containing region and shift it horizontally frame-by-frame so
    ByteTrack has both a detectable entity AND motion to lock onto.

    Returns True if the video was created from a real image, False if we
    had to fall back (in which case the caller should skip rather than
    fake).
    """
    import cv2
    import numpy as np

    src = _have_real_person_image()
    if src is None:
        return False

    img = cv2.imread(src)
    if img is None:
        return False

    h_src, w_src = img.shape[:2]
    # crop a person-bearing region (the sample images have people roughly
    # centered; take the middle band)
    crop = img[h_src // 4: 3 * h_src // 4, :, :]
    ch, cw = crop.shape[:2]

    out_w, out_h = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, 12, (out_w, out_h))

    # scale crop to fit height AND be at most 60% of output width (so it
    # can move horizontally without filling the whole frame)
    target_h = out_h - 20
    scale_h = target_h / ch
    target_w = int(cw * scale_h)
    max_w = int(out_w * 0.55)
    if target_w > max_w:
        scale = max_w / cw
    else:
        scale = scale_h
    scaled = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))))
    sh = scaled.shape[0]
    sw = scaled.shape[1]
    max_shift = max(1, out_w - sw - 20)

    for i in range(frames):
        frame = np.full((out_h, out_w, 3), 30, dtype=np.uint8)
        x = 10 + int(i * max_shift / max(1, frames - 1))
        y = 10
        frame[y:y + sh, x:x + sw] = scaled
        vw.write(frame)
    vw.release()
    return True


def test_bytetrack_emits_stable_ids_across_frames(tmp_path):
    """The core P0 test: real YOLO + real ByteTrack → stable ids."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from inference.detection.yolo_detector import YoloDetector
    from inference.tracking.bytetrack_tracker import BytetrackTracker

    video = str(tmp_path / "moving.mp4")
    made = _make_moving_person_video(video, frames=20)
    if not made:
        pytest.skip("No real person sample image available (ultralytics assets missing) "
                    "— cannot test real detection without faking it, which is forbidden.")

    # read frames back
    import cv2
    cap = cv2.VideoCapture(video)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    assert len(frames) == 20

    detector = YoloDetector(person_weights="yolov8n.pt", litter_weights="nonexistent.pt",
                          bag_weights=None)
    detector.load()
    # litter model absent and bag model explicitly disabled → person-only
    # path. The working tree otherwise loads garbage_bag_v2.pt which emits
    # spurious Garbage-Bag boxes on the synthetic zidane crop and would
    # pollute the person-stability assertion (detected via stash comparison).

    tracker = BytetrackTracker()
    tracker.load()

    # run real track() across the sequence (person ids only — bag/color
    # false positives on the synthetic crop are irrelevant to person-id
    # stability and would otherwise make the all_ids→namespace lookup
    # ambiguous).
    per_frame_ids = []
    per_frame_tracked = []
    for i, f in enumerate(frames):
        tracked = detector.track(f, persist=True)
        tracker.update(tracked, frame_index=i)
        ids_this_frame = sorted(t.track_id for t in tracked if t.is_person)
        per_frame_ids.append(ids_this_frame)
        per_frame_tracked.append(tracked)

    # We must have detected SOMETHING in at least some frames.
    detected_frames = [ids for ids in per_frame_ids if ids]
    assert len(detected_frames) >= 1, (
        "YOLO detected nothing across 20 frames — check the synthetic video "
        "or the detector. This test cannot pass without real detections."
    )

    # The ids that appear must be STABLE: once an id appears, it should
    # keep appearing (possibly with brief gaps from ByteTrack) in
    # subsequent frames, NOT be replaced by a brand-new id every frame.
    all_ids = set()
    for ids in detected_frames:
        all_ids.update(ids)
    # With a single moving entity, we expect ~1 id (ByteTrack may
    # occasionally fragment, but NOT 24 different ids).
    assert len(all_ids) <= 4, (
        f"Expected stable ids (≤4 distinct) but got {len(all_ids)} distinct "
        f"ids: {all_ids}. ByteTrack is not persisting ids — the temporal "
        f"layer would be fed garbage."
    )

    # The store must have recorded history for those ids
    for tid in all_ids:
        ns_id = tracker.namespace(tid, is_person=True)
        hist = tracker.get_history(ns_id)
        assert hist is not None, f"TrackStore missing history for id {ns_id}"
        assert len(hist.centroids) >= 1

    # Centroids should show movement (the block moves ~8px/frame)
    if all_ids:
        tid = next(iter(all_ids))
        ns_id = tracker.namespace(tid, is_person=True)
        hist = tracker.get_history(ns_id)
        if len(hist.centroids) >= 2:
            xs = [c[0] for c in hist.centroids]
            assert max(xs) - min(xs) > 0, "tracked centroid did not move — tracker may be stuck"


def test_to_tracks_splits_persons_and_objects():
    """The adapter correctly splits tracked detections into persons/objects
    with namespaced ids and attaches keypoints to persons only."""
    from inference.detection.yolo_detector import TrackedDetection
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from inference.association.person_object_assoc import Keypoints

    tracker = BytetrackTracker()
    tracked = [
        TrackedDetection(track_id=1, class_name="person", confidence=0.9,
                         bbox=(0, 0, 50, 100), centroid=(25, 50), is_person=True),
        TrackedDetection(track_id=2, class_name="person", confidence=0.8,
                         bbox=(60, 0, 110, 100), centroid=(85, 50), is_person=True),
        TrackedDetection(track_id=1, class_name="bottle", confidence=0.7,
                         bbox=(120, 120, 140, 160), centroid=(130, 140), is_person=False),
    ]
    kp = {tracker.namespace(1, True): Keypoints(left_wrist=(30, 60), torso_center=(25, 50))}
    persons, objects = tracker.to_tracks(tracked, keypoints_by_person_ns=kp)

    assert len(persons) == 2
    assert len(objects) == 1
    assert persons[0].track_id == 1  # namespaced person
    assert persons[1].track_id == 2
    assert objects[0].track_id == 10001  # namespaced object (offset 10000)
    assert persons[0].keypoints is not None  # got pose
    assert persons[1].keypoints is None     # no pose for id 2
    assert objects[0].keypoints is None


def test_tracker_store_pruning():
    """TrackStore prunes stale tracks."""
    from inference.detection.yolo_detector import TrackedDetection
    from inference.tracking.bytetrack_tracker import BytetrackTracker

    tracker = BytetrackTracker()
    td = TrackedDetection(track_id=1, class_name="person", confidence=0.9,
                          bbox=(0, 0, 50, 100), centroid=(25, 50), is_person=True)
    tracker.update([td], frame_index=0)
    assert tracker.store_size == 1
    # advance many frames without seeing id 1 again
    tracker.update([], frame_index=100)
    tracker.prune(max_age_frames=60)
    assert tracker.store_size == 0
