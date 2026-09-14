"""Section 6 — LiveCameraReader + session hygiene unit tests (no real camera)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np

from inference.capture.live_camera_reader import LiveCameraReader
from inference.capture.live_hygiene import LiveHygieneConfig, LiveSessionHygiene, reset_live_trackers
from inference.evidence.face_evidence import FaceEvidenceCapture


class _FakeCap:
    def __init__(self, n: int = 5, fail_after: int | None = None):
        self.n = n
        self.i = 0
        self.fail_after = fail_after
        self.released = False

    def isOpened(self):
        return not self.released

    def read(self):
        if self.fail_after is not None and self.i >= self.fail_after:
            return False, None
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)

    def set(self, *_a, **_k):
        return True

    def release(self):
        self.released = True


def test_live_reader_keeps_latest_and_ring_buffer():
    frames_made = {"n": 0}

    class EndlessCap(_FakeCap):
        def read(self):
            frames_made["n"] += 1
            return True, np.zeros((48, 64, 3), dtype=np.uint8)

    fake = EndlessCap()
    reader = LiveCameraReader(
        0,
        buffer_seconds=2.0,
        target_fps=30,
        reconnect_delay=0.05,
        max_reconnect_delay=0.1,
    )
    reader._open_capture = lambda: fake  # type: ignore
    reader.start()
    deadline = time.time() + 2.0
    latest = None
    while time.time() < deadline:
        latest = reader.get_latest_frame()
        if latest is not None and latest.frame_index >= 3:
            break
        time.sleep(0.02)
    reader.stop()

    assert latest is not None
    assert latest.frame_index >= 3
    assert len(reader.ring_buffer) >= 1
    clip = reader.get_buffered_clip(2.0, around_ts=latest.timestamp)
    assert isinstance(clip, list)
    assert frames_made["n"] >= 4


def test_hygiene_empty_scene_triggers_after_window():
    h = LiveSessionHygiene(
        LiveHygieneConfig(
            empty_scene_seconds=0.2,
            max_session_seconds=3600.0,
            min_reset_interval_seconds=0.0,
        )
    )
    t0 = 1000.0
    assert h.observe(person_count=0, now=t0) is False
    assert h.observe(person_count=0, now=t0 + 0.25) is True
    h.mark_reset(now=t0 + 0.25)
    assert h.observe(person_count=0, now=t0 + 0.30) is False


def test_hygiene_hourly_triggers():
    h = LiveSessionHygiene(
        LiveHygieneConfig(
            empty_scene_seconds=999.0,
            max_session_seconds=1.0,
            min_reset_interval_seconds=0.0,
        )
    )
    t0 = h._session_started
    assert h.observe(person_count=2, now=t0 + 0.5) is False
    assert h.observe(person_count=2, now=t0 + 1.1) is True


def test_reset_live_trackers_calls_reset_hooks():
    det = MagicMock()
    trk = MagicMock()
    ed = MagicMock()
    nov = MagicMock()
    reset_live_trackers(detector=det, tracker_ns=trk, event_detector=ed, novelty=nov)
    det.reset_tracking.assert_called_once()
    trk.reset.assert_called_once()
    ed.reset.assert_called_once()
    nov.reset.assert_called_once()


def test_face_quality_prefers_level_shoulders():
    face = FaceEvidenceCapture(backend="region")
    crop = np.random.randint(40, 200, (80, 60, 3), dtype=np.uint8)
    frontal = face.frontality_score(
        {"left_shoulder": (100.0, 200.0), "right_shoulder": (180.0, 201.0)}
    )
    profile = face.frontality_score(
        {"left_shoulder": (100.0, 200.0), "right_shoulder": (120.0, 260.0)}
    )
    assert frontal > profile
    q = face.frame_quality_score(
        crop,
        {"h": 80, "conf": 1.0},
        keypoints={"left_shoulder": (100.0, 200.0), "right_shoulder": (180.0, 200.0)},
    )
    assert 0.0 <= q <= 1.0
