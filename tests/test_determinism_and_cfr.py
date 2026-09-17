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


def test_deterministic_inference_reads_pin_not_live(tmp_path, monkeypatch):
    """P0-A: mutating live learning.json must not change deterministic tier-0."""
    import json
    from adaptive_tuner import LearningStore, build_tier_configs, resolve_inference_overrides

    pin = tmp_path / "inference_pin.json"
    live = tmp_path / "learning.json"
    pin.write_text(
        json.dumps(
            {
                "version": 1,
                "step_index": {"NOT_ENOUGH_CARRIED_FRAMES": 0},
                "rejection_reasons": {},
                "history": {},
            }
        ),
        encoding="utf-8",
    )
    live.write_text(
        json.dumps(
            {
                "version": 1,
                "step_index": {"NOT_ENOUGH_CARRIED_FRAMES": 3},
                "rejection_reasons": {},
                "history": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("adaptive_tuner.INFERENCE_PIN_PATH", str(pin))
    monkeypatch.delenv("MOTARED_LEARNING_SOURCE", raising=False)

    store = LearningStore(path=str(live))
    pinned = resolve_inference_overrides(store, deterministic=True) or {}
    live_ov = store.learned_overrides()
    # Live is fully stepped (min_carried=3); pin step 0 applies nothing.
    assert live_ov.get("min_carried_frames") == 3
    assert "min_carried_frames" not in pinned

    det = AdaptiveEventDetector(
        load_event_config(),
        store=store,
        deterministic=True,
        max_tiers=0,
    )
    assert det.learning_source == "pin"
    pin_cfg = build_tier_configs(
        load_event_config(), LearningStore(str(pin)).learned_overrides() or None, 0
    )[0]
    assert det.tier_configs[0].min_carried_frames == pin_cfg.min_carried_frames
    # Live store alone would have lowered min_carried — must not leak in.
    live_cfg = build_tier_configs(load_event_config(), live_ov, 0)[0]
    assert det.tier_configs[0].min_carried_frames != live_cfg.min_carried_frames


def test_deterministic_ignores_live_learning_source_env(tmp_path, monkeypatch):
    """P0-A: MOTARED_LEARNING_SOURCE=live must not reopen the read leak."""
    import json
    from adaptive_tuner import AdaptiveEventDetector, LearningStore

    pin = tmp_path / "inference_pin.json"
    live = tmp_path / "learning.json"
    pin.write_text(
        json.dumps({"version": 1, "step_index": {}, "rejection_reasons": {}, "history": {}}),
        encoding="utf-8",
    )
    live.write_text(
        json.dumps(
            {
                "version": 1,
                "step_index": {"NOT_ENOUGH_CARRIED_FRAMES": 3},
                "rejection_reasons": {},
                "history": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("adaptive_tuner.INFERENCE_PIN_PATH", str(pin))
    monkeypatch.setenv("MOTARED_LEARNING_SOURCE", "live")
    det = AdaptiveEventDetector(
        load_event_config(), store=LearningStore(path=str(live)), deterministic=True
    )
    assert det.learning_source == "pin"
    assert "min_carried_frames" not in (det.summary()["adaptive"]["learned_overrides"] and {})
    assert det.tier_configs[0].min_carried_frames == load_event_config().min_carried_frames


def test_deterministic_triple_run_identical_with_live_mutation(tmp_path, monkeypatch):
    """P0-A acceptance: 3x same ticks pinned -> identical confirms; live edits ignored."""
    import json
    from adaptive_tuner import LearningStore
    from tests.test_adaptive_tuner import _run_carried_release_ground

    pin = tmp_path / "inference_pin.json"
    live = tmp_path / "learning.json"
    pin.write_text(
        json.dumps({"version": 1, "step_index": {}, "rejection_reasons": {}, "history": {}}),
        encoding="utf-8",
    )
    live.write_text(
        json.dumps({"version": 1, "step_index": {}, "rejection_reasons": {}, "history": {}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("adaptive_tuner.INFERENCE_PIN_PATH", str(pin))
    monkeypatch.delenv("MOTARED_LEARNING_SOURCE", raising=False)
    results = []
    for i in range(3):
        if i == 1:
            # Mutate LIVE step_index mid-sequence: pinned runs must not notice.
            live.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "step_index": {
                            "NOT_ENOUGH_CARRIED_FRAMES": 3,
                            "NO_RELEASE_TRANSITION": 3,
                        },
                        "rejection_reasons": {},
                        "history": {},
                    }
                ),
                encoding="utf-8",
            )
        det = AdaptiveEventDetector(
            load_event_config(), store=LearningStore(path=str(live)), deterministic=True
        )
        assert det.learning_source == "pin"
        confirmed = _run_carried_release_ground(det, carry_ticks=6)
        confirmed_t = sorted(round(e.timestamps.get("confirmed", 0.0), 3) for e in confirmed)
        results.append((len(confirmed), confirmed_t, det.tier_configs[0].min_carried_frames))
    assert results[0] == results[1] == results[2]
    assert results[0][1], "synthetic sequence must confirm on all 3 pinned runs"


def test_shared_file_pipeline_matches_upload_defaults():
    """P0-B: CLI file mode builds the same loop config as upload/frozen paths."""
    from scripts.run_pipeline import FILE_MODE_DEFAULTS, build_shared_file_pipeline

    cfg = build_shared_file_pipeline(analysis_fps=8.0, camera_id="cam-9", deterministic=True)
    assert cfg.buffer_seconds == FILE_MODE_DEFAULTS["buffer_seconds"]
    assert cfg.analysis_fps == 8.0
    assert cfg.auto_tune is True
    assert cfg.deterministic is True
    assert cfg.camera_id == "cam-9"


def test_fit_report_json_preserves_stage_attribution_when_truncated():
    """Regression (IMG_5291 job 43): a truncation must NOT drop stage data.

    Dropping ``stages``/``pipeline_telemetry`` made the dashboard fall back to
    "Stage 1 — VIDEO INPUT FAILED" while the true failure was Stage 9.
    """
    from backend.routers.analysis import PIPELINE_STAGES, PipelineStageTracker, _fit_report_json
    import json

    tracker = PipelineStageTracker()
    tracker.complete_stage("video_input", "Decoded 942 frames")
    tracker.current_stage_id = "temporal_event_detection"
    tracker.fail_stage("temporal_event_detection", "name 'sep_floor' is not defined", frame_idx=870)
    telemetry = tracker.build_telemetry(
        job_id=43,
        status="failed",
        processed_frames=870,
        total_frames=942,
        error_message="name 'sep_floor' is not defined",
    )
    # Bloat the report so the serializer is forced through its rungs.
    telemetry["timeline"] = [{"i": i, "pad": "x" * 120} for i in range(120)]
    telemetry["event_detector"] = {
        "config": {k: 1.0 for k in range(200)},
        "confirmed_violations": [{"pad": "y" * 200} for _ in range(10)],
        "rejected_candidates": [{"pad": "z" * 200} for _ in range(10)],
        "summary": {},
    }

    raw = _fit_report_json(telemetry)
    assert len(raw) <= 4096
    payload = json.loads(raw)

    assert payload.get("pipeline_telemetry", {}).get("current_stage_id") == "temporal_event_detection"
    assert payload["pipeline_telemetry"]["current_step"] == 9
    assert payload["pipeline_telemetry"]["current_stage_status"].lower() == "failed"
    stages = payload.get("stages")
    assert stages, "stage matrix must survive truncation"
    by_id = {s["id"]: s for s in stages}
    assert by_id["temporal_event_detection"]["status"] == "FAILED"
    assert by_id["video_input"]["status"] == "COMPLETED"
    assert by_id["evidence_assembly"]["status"] == "SKIPPED"
    assert len(stages) == len(PIPELINE_STAGES)


def test_failure_forensics_writes_context_and_real_frame(tmp_path, monkeypatch):
    """Fault-injection test for the crash forensics (IMG_5291 class of failure).

    A synthetic-but-real frame is drawn and the handler is invoked with a real
    exception, so the assertions cover the actual mechanism: JSON context +
    annotated failure image + FSM pair snapshot + thresholds. This does NOT
    reproduce the original job-43 crash (that handler did not exist then, so
    no forensics were captured for it) — it proves the new path works.
    """
    import numpy as np
    import json
    import backend.routers.analysis as an

    monkeypatch.setattr(an, "REPO_ROOT", tmp_path)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    class _P:
        track_id = 1
        bbox = (10, 20, 60, 200)
        confidence = 0.9
        keypoints = None

    class _B:
        track_id = 60013
        bbox = (70, 150, 110, 190)
        confidence = 0.5
        class_name = "black_waste_bag"
        source = "yolo"
        object_uid = 100003

    from littering_event_detector import DetectorBag, DetectorPerson, EventDetectorConfig, LitteringEventDetector

    det = LitteringEventDetector(EventDetectorConfig())
    person = DetectorPerson(track_id=1, bbox=(10, 20, 60, 200), confidence=0.9)
    bag = DetectorBag(
        track_id=60013, bbox=(70, 150, 110, 190), confidence=0.5,
        class_name="black_waste_bag", source="yolo", yolo_confirmed=True,
        object_uid=100003,
    )
    det.update([person], [bag], 1.0, 1)

    out = an._write_failure_forensics(
        job_id=999,
        video_name="IMG_5291.MOV",
        stage_id="temporal_event_detection",
        exc=NameError("name 'sep_floor' is not defined"),
        traceback_text="Traceback (most recent call last): ...",
        frame_idx=870,
        timestamp=29.0,
        source_fps=30.0,
        frame=frame,
        persons=[person],
        objects=[bag],
        detector=det,
        pair_states=an._pair_forensics(det),
    )

    assert out["stage"] == "temporal_event_detection"
    assert out["exception_type"] == "NameError"
    assert out["frame_index"] == 870
    assert out["timestamp_sec"] == 29.0
    assert out["source_fps"] == 30.0
    assert out["entities"]["objects"][0]["object_uid"] is not None  # assigned stable UID
    assert out["entities"]["objects"][0]["class_name"] == "black_waste_bag"
    assert out["pairs"], "FSM pair snapshot must be captured"
    assert out["pairs"][0]["person_track_id"] == 1
    assert out["pairs"][0]["person_uid"] is not None
    assert out["detector"]["active_thresholds"]["min_carried_frames"] >= 1

    img = out["frame_image"]
    assert img, "a real annotated failure image must be written"
    p = Path(img)
    assert p.is_file() and p.stat().st_size > 0
    import cv2

    read = cv2.imread(str(p))
    assert read is not None and read.shape == frame.shape
    ctx = Path(out["context_path"])
    assert ctx.is_file()
    ctx_data = json.loads(ctx.read_text(encoding="utf-8"))
    assert ctx_data["exception"] == "name 'sep_floor' is not defined"
    assert ctx_data["stage"] == "temporal_event_detection"
    assert ctx_data["timestamp_sec"] == 29.0
    assert ctx_data["nearest_frame"] is False


def test_configure_determinism_reports_strict_mode(monkeypatch):
    monkeypatch.delenv("MOTARED_DETERMINISTIC", raising=False)
    report = configure_determinism(0)
    if report.get("torch"):
        assert report.get("torch_deterministic_warn_only") is False
