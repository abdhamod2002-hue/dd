"""Container-safe production runner for one video (matches analysis.py path)."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
os.environ.setdefault("TFHUB_CACHE_DIR", str(REPO / "models" / "movenet"))

from inference.capture.camera_source import VideoFileSource  # noqa: E402
from inference.detection.yolo_detector import YoloDetector  # noqa: E402
from inference.detection.novelty_detector import NoveltyDetector  # noqa: E402
from inference.pose.movenet_pose import MovenetPose  # noqa: E402
from inference.tracking.bytetrack_tracker import BytetrackTracker  # noqa: E402
from inference.pipeline import InferencePipeline, PipelineConfig  # noqa: E402
from scripts.run_pipeline import build_tracks_real  # noqa: E402


def main(video: Path) -> int:
    print("VIDEO", video, "exists", video.exists(), flush=True)
    # Fresh learning store so polluted learning/learning.json overrides cannot
    # silently change release geometry for this proof run.
    from adaptive_tuner import AdaptiveEventDetector, LearningStore
    from littering_event_detector import load_event_config
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="proof_learn_"))
    store = LearningStore(path=str(tmp / "learning.json"))

    source = VideoFileSource(str(video))
    if not source.open():
        raise RuntimeError(f"failed to open {video}")
    detector = YoloDetector()
    detector.load()
    tracker = BytetrackTracker()
    movenet = MovenetPose()
    try:
        movenet.load()
    except Exception:
        pass
    try:
        nov = NoveltyDetector()
    except Exception:
        nov = None
    pipe = InferencePipeline(PipelineConfig(analysis_fps=8.0, auto_tune=True))
    # Replace adaptive detector with a clean-store instance (same tiers, no
    # cross-video learning pollution).
    pipe.event_detector = AdaptiveEventDetector(load_event_config(), store=store)
    pipe.event_detector.learning_video = video.name

    frame_idx = 0
    t0 = time.time()
    while True:
        pkt = source.read()
        if pkt is None:
            break
        tracked = detector.track(pkt.frame, persist=True)
        run_pose = pipe.should_analyze(float(pkt.timestamp))
        persons, objects = build_tracks_real(
            pkt.frame, tracked, movenet, tracker, frame_idx,
            run_pose=run_pose, nov=nov,
        )
        pipe.process_frame(pkt.frame, float(pkt.timestamp), persons, objects)
        frame_idx += 1

    for _ in pipe.finalize():
        pass

    ed = pipe.event_detector
    summary = ed.summary()
    conf = [e.to_dict() for e in ed.confirmed_events]
    rej = [e.to_dict() for e in ed.rejected_events]
    out = {
        "video": str(video),
        "frames": frame_idx,
        "elapsed_sec": round(time.time() - t0, 2),
        "detector": type(ed).__name__,
        "summary": summary,
        "confirmed": conf,
        "rejected": [
            {
                "reason": r.get("reason"),
                "state": r.get("state"),
                "frames": r.get("frames"),
                "evidence": r.get("evidence"),
                "fallback_used": r.get("fallback_used"),
                "event_actor_person_uid": r.get("event_actor_person_uid"),
                "event_object_uid": r.get("event_object_uid"),
            }
            for r in rej
        ],
    }
    out_path = REPO / "project_audit" / "_diag_5117_result.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("WROTE", out_path, flush=True)
    print(
        "confirmed", len(conf), "rejected", len(rej),
        "reasons", summary.get("rejection_reason_counts"),
        flush=True,
    )
    if conf:
        c = conf[0]
        print(
            "CONFIRMED frames", c.get("frames"),
            "uids", c.get("event_actor_person_uid"), c.get("event_object_uid"),
            "conf", c.get("confidence"),
            flush=True,
        )
    elif rej:
        print("REJECT0", out["rejected"][0], flush=True)
    return 0 if conf else 1


if __name__ == "__main__":
    vid = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/app/backend/uploaded_videos/20260831_104342_IMG_5117.MOV"
    )
    raise SystemExit(main(vid))
