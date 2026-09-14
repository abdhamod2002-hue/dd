"""Unit tests for CFR normalize + runtime determinism (sections 2–3)."""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from backend.services.video_normalizer import ensure_cfr_source, normalize_video
from inference.runtime_determinism import (
    configure_determinism,
    determinism_enabled_from_env,
)
from adaptive_tuner import AdaptiveEventDetector
from littering_event_detector import load_event_config


def test_determinism_enabled_defaults_on(monkeypatch):
    monkeypatch.delenv("MOTARED_DETERMINISTIC", raising=False)
    assert determinism_enabled_from_env() is True
    monkeypatch.setenv("MOTARED_DETERMINISTIC", "0")
    assert determinism_enabled_from_env() is False


def test_configure_determinism_sets_env_and_seeds():
    report = configure_determinism(0)
    assert report.get("configured") is True
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert os.environ.get("PYTHONHASHSEED") == "0"


def test_normalize_video_invokes_ffmpeg_cfr(tmp_path, monkeypatch):
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake")
    dst = tmp_path / "out_CFR.mp4"

    def _fake_run(cmd, capture_output=True, text=True, timeout=3600):
        # Mimic ffmpeg success by writing the output path (last arg).
        Path(cmd[-1]).write_bytes(b"cfr")
        return mock.Mock(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(
        "backend.services.video_normalizer._ffmpeg_exe",
        lambda: "ffmpeg",
    )
    monkeypatch.setattr(
        "backend.services.video_normalizer.subprocess.run",
        _fake_run,
    )
    out = normalize_video(src, dst, target_fps=30)
    assert Path(out).is_file()
    assert Path(out).read_bytes() == b"cfr"


def test_ensure_cfr_skip_env(tmp_path, monkeypatch):
    src = tmp_path / "orig.mp4"
    src.write_bytes(b"x")
    monkeypatch.setenv("MOTARED_SKIP_CFR", "1")
    out = ensure_cfr_source(src, tmp_path)
    assert Path(out).resolve() == src.resolve()


def test_adaptive_detector_freezes_learning_writes_when_deterministic(tmp_path):
    from adaptive_tuner import LearningStore

    store_path = tmp_path / "learning.json"
    store = LearningStore(path=str(store_path))
    det = AdaptiveEventDetector(
        load_event_config(),
        store=store,
        deterministic=True,
    )
    assert det.freeze_learning_writes is True
    det.learning_video = "same.mp4"
    before = store_path.read_text(encoding="utf-8") if store_path.exists() else ""
    det.finalize()
    after = store_path.read_text(encoding="utf-8") if store_path.exists() else ""
    assert after == before
