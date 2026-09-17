"""Diagnostic (read-only): run the production pipeline over A.MOV and dump
the full evidence/details of any confirmed event, to identify exactly which
release/carry criteria fired. No FSM/threshold changes made here."""
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.video_normalizer import ensure_cfr_source
from inference.capture.camera_source import VideoFileSource
from inference.detection.novelty_detector import NoveltyConfig, NoveltyDetector
from inference.detection.yolo_detector import YoloDetector
from inference.pipeline import InferencePipeline
from inference.pose.movenet_pose import MovenetPose
from inference.runtime_determinism import configure_determinism
from inference.tracking.bytetrack_tracker import BytetrackTracker
from scripts.run_pipeline import build_shared_file_pipeline, build_tracks_real

SOURCE = ROOT / "evidence_store" / "analysis" / "63" / "source_CFR.mp4"

configure_determinism(0)
source = VideoFileSource(str(SOURCE))
assert source.open(), f"cannot open {SOURCE}"

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
    build_shared_file_pipeline(
        analysis_fps=8.0, camera_id="diag", deterministic=True, learning_tag="diag_amov"
    )
)
pipe.event_detector.reset()

n = 0
last_ts = 0.0
wrist_near_log = []
for pkt in source:
    n += 1
    last_ts = float(pkt.timestamp)
    tracked = detector.track(pkt.frame, persist=True)
    run_pose = pipe.should_analyze(pkt.timestamp)
    persons, objects = build_tracks_real(
        pkt.frame, tracked, movenet, tracker, n - 1, run_pose=run_pose, nov=nov
    )
    pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
source.release()
pipe.finalize(last_ts)

print("=== CONFIRMED EVENTS ===")
for ev in pipe.event_detector.confirmed_events:
    print(json.dumps(ev.to_dict(), indent=1, default=str))

print("=== REJECTED EVENTS (first 5) ===")
for ev in pipe.event_detector.rejected_events[:5]:
    print(json.dumps(ev.to_dict(), indent=1, default=str))
