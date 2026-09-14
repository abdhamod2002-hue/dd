"""Run the same CFR-normalized clip 3 times and compare FSM fingerprints.

Usage (inside littering-backend container or host venv with models):

  python project_audit/check_determinism_3runs.py \\
      backend/uploaded_videos/20260821_180002_WhatsApp\\ Video\\ 2026-08-21\\ at\\ 5.15.03\\ PM.mp4

Exit 0 only when all three fingerprints match.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _fingerprint(confirmed, rejected, frames_n: int) -> dict:
    conf = [
        {
            "person": getattr(e, "person_track_id", None),
            "bag": getattr(e, "bag_track_id", None),
            "reason": getattr(e, "reason", None),
            "frames": getattr(e, "frames", None),
            "conf": round(float(getattr(e, "confidence", 0.0) or 0.0), 4),
        }
        for e in confirmed
    ]
    rej = [
        {
            "person": getattr(e, "person_track_id", None),
            "reason": getattr(e, "reason", None),
            "frames": getattr(e, "frames", None),
        }
        for e in rejected
    ]
    payload = {"frames": frames_n, "confirmed": conf, "rejected": rej}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return {
        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "n_confirmed": len(conf),
        "n_rejected": len(rej),
        "payload": payload,
    }


def _run_once(video_path: Path) -> dict:
    from backend.services.video_normalizer import ensure_cfr_source
    from inference.capture.camera_source import VideoFileSource
    from inference.detection.novelty_detector import NoveltyConfig, NoveltyDetector
    from inference.detection.yolo_detector import YoloDetector
    from inference.pipeline import InferencePipeline, PipelineConfig
    from inference.pose.movenet_pose import MovenetPose
    from inference.runtime_determinism import configure_determinism
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from scripts.run_pipeline import build_tracks_real

    configure_determinism(0)
    with tempfile.TemporaryDirectory(prefix="motared_cfr_") as td:
        cfr = Path(ensure_cfr_source(video_path, td, stem="source_CFR"))
        source = VideoFileSource(str(cfr))
        assert source.open(), f"cannot open {cfr}"

        detector = YoloDetector()
        detector.load()
        detector.reset_tracking()
        tracker = BytetrackTracker()
        tracker.load()
        nov_cfg = NoveltyConfig.from_yaml()
        nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
        movenet = MovenetPose()
        movenet.load()

        pipe = InferencePipeline(
            PipelineConfig(
                buffer_seconds=8.0,
                analysis_fps=8.0,
                camera_id="determinism-check",
                post_backend_url=None,
                auto_tune=True,
                deterministic=True,
            )
        )
        pipe.event_detector.reset()

        frame_n = 0
        last_ts = 0.0
        for pkt in source:
            frame_n += 1
            last_ts = float(pkt.timestamp)
            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker, frame_n - 1,
                run_pose=run_pose, nov=nov,
            )
            pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)

        source.release()
        pipe.finalize(last_ts)
        ed = pipe.event_detector
        return _fingerprint(ed.confirmed_events, ed.rejected_events, frame_n)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: check_determinism_3runs.py <video>", file=sys.stderr)
        return 2
    video = Path(sys.argv[1])
    if not video.is_file():
        print(f"missing video: {video}", file=sys.stderr)
        return 2

    fps = []
    for i in range(3):
        print(f"=== run {i + 1}/3 ===", flush=True)
        fp = _run_once(video)
        print(
            f"run{i + 1}: frames={fp['payload']['frames']} "
            f"confirmed={fp['n_confirmed']} rejected={fp['n_rejected']} "
            f"sha={fp['sha256'][:16]}",
            flush=True,
        )
        fps.append(fp)

    hashes = [f["sha256"] for f in fps]
    ok = hashes[0] == hashes[1] == hashes[2]
    out = {
        "match": ok,
        "hashes": hashes,
        "runs": [{"n_confirmed": f["n_confirmed"], "n_rejected": f["n_rejected"], "sha256": f["sha256"]} for f in fps],
    }
    print(json.dumps(out, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
