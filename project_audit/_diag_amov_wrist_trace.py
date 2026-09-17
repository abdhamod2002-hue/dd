"""Diagnostic (read-only, no production code changed): trace per-tick wrist
distance / containment / state for the pair that produced A.MOV's false
positive, to see whether the AIDM 'grip' was a sustained hold or a single
coincidental tick during normal walking gait."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import littering_event_detector as led

orig_advance = led.LitteringEventDetector._advance_pair


_INTERESTING = {
    led.EventState.BAG_CARRIED,
    led.EventState.BAG_RELEASED,
    led.EventState.BAG_ON_GROUND,
    led.EventState.PERSON_DEPARTED,
    led.EventState.VIOLATION_CONFIRMED,
}


def traced_advance(self, mem, info, timestamp, frame_index):
    result = orig_advance(self, mem, info, timestamp, frame_index)
    if mem.state in _INTERESTING:
        print(
            f"t={timestamp:6.2f} f={frame_index:4d} state={mem.state.value:16s} "
            f"carried={info.carried!s:5s} wrist_near={info.wrist_near!s:5s} "
            f"wrist_d={info.wrist_d_norm} containment={info.containment:.2f} "
            f"norm_dist={info.norm_distance:.3f} "
            f"ever_attach={mem.ever_aidm_attached} aidm_sep={mem.aidm_separated} "
            f"max_wrist_d={mem.max_wrist_d_norm:.3f} bag_id={mem.bag_id} "
            f"person_uid={mem.person_uid} sep_frames={mem.separated_frames} "
            f"max_post_sep={mem.max_post_release_norm_distance:.3f}"
        )
    return result


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
        analysis_fps=8.0, camera_id="diag2", deterministic=True, learning_tag="diag_amov2"
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
print("DONE")
