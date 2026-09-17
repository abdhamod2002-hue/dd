"""Host oracle re-run for IMG_5290 (full video, clean learning store)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
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
from adaptive_tuner import AdaptiveEventDetector, LearningStore  # noqa: E402
from littering_event_detector import load_event_config  # noqa: E402


def main() -> int:
    video = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\22\IMG_5290.MOV")
    mode = sys.argv[2] if len(sys.argv) > 2 else "POSTFIX5"
    out_path = REPO / "project_audit" / f"_oracle_IMG_5290_{mode.lower()}.json"

    tmp = Path(tempfile.mkdtemp(prefix="oracle5290_learn_"))
    store = LearningStore(path=str(tmp / "learning.json"))

    source = VideoFileSource(str(video))
    if not source.open():
        raise RuntimeError(f"failed to open {video}")
    fps = float(getattr(source, "fps", 0) or 59.97)

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
        if frame_idx % 200 == 0:
            print(f"frame {frame_idx}", flush=True)

    for _ in pipe.finalize():
        pass

    ed = pipe.event_detector
    summary = ed.summary()
    conf = [e.to_dict() for e in ed.confirmed_events]
    rej = [e.to_dict() for e in ed.rejected_events]

    def rel_times(frames: dict) -> dict:
        out = {}
        for k, v in (frames or {}).items():
            if v is None:
                out[k] = None
            else:
                out[k] = round(float(v) / fps, 3)
        return out

    payload = {
        "mode": mode,
        "video": str(video),
        "frames": frame_idx,
        "fps": fps,
        "elapsed": round(time.time() - t0, 2),
        "summary": summary,
        "confirmed": conf,
        "confirmed_rel_times": [rel_times(c.get("frames")) for c in conf],
        "rejected": [
            {
                "reason": r.get("reason"),
                "state": r.get("state"),
                "frames": r.get("frames"),
                "rel_times": rel_times(r.get("frames")),
                "bag_class": r.get("bag_class"),
                "evidence": r.get("evidence"),
                "uids": [
                    r.get("event_actor_person_uid"),
                    r.get("event_object_uid"),
                ],
            }
            for r in rej
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("WROTE", out_path, flush=True)
    print(
        "confirmed", len(conf),
        "rej_reasons", summary.get("rejection_reason_counts"),
        flush=True,
    )
    if conf:
        c = conf[0]
        print(
            "CONFIRMED",
            c.get("bag_class"),
            "uids", c.get("event_actor_person_uid"), c.get("event_object_uid"),
            "rel", payload["confirmed_rel_times"][0],
            "conf", c.get("confidence"),
            flush=True,
        )
    return 0 if conf else 1


if __name__ == "__main__":
    raise SystemExit(main())
