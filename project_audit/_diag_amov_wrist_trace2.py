"""Diagnostic (read-only, no production code changed): record every tick's
wrist/containment/state for every pair keyed by STABLE (person_uid, bag_uid),
then print full history for whichever pair the pipeline actually confirms."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import littering_event_detector as led

orig_advance = led.LitteringEventDetector._advance_pair
HISTORY = {}


def traced_advance(self, mem, info, timestamp, frame_index):
    key = (mem.person_uid, mem.bag_uid)
    HISTORY.setdefault(key, []).append(
        (
            timestamp,
            frame_index,
            mem.state.value,
            info.carried,
            info.wrist_near,
            info.wrist_d_norm,
            info.containment,
            info.norm_distance,
            mem.ever_aidm_attached,
            mem.aidm_separated,
            mem.max_wrist_d_norm,
        )
    )
    return orig_advance(self, mem, info, timestamp, frame_index)


led.LitteringEventDetector._advance_pair = traced_advance

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
assert source.open()

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
        analysis_fps=8.0, camera_id="diag3", deterministic=True, learning_tag="diag_amov3"
    )
)
pipe.event_detector.reset()

n = 0
last_ts = 0.0
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
    key = (ev.event_actor_person_uid, ev.event_object_uid)
    print(f"CONFIRMED key={key} bag_class={ev.bag_class} conf={ev.confidence}")
    for row in HISTORY.get(key, []):
        print("  ", row)

print("=== ALL PAIR KEYS SEEN (with tick counts) ===")
for key, rows in HISTORY.items():
    max_state = max((r[2] for r in rows), default=None)
    print(key, "ticks=", len(rows), "states=", sorted(set(r[2] for r in rows)))
