"""One-shot diagnostic: why IMG_5117 never releases after P0-01 fixes."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path("/app") if Path("/app/littering_event_detector.py").exists() else Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from inference.capture.camera_source import VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pipeline import InferencePipeline, PipelineConfig
from scripts.run_pipeline import build_tracks_real

VIDEO = Path(r"D:\22\IMG_5117.MOV")
if not VIDEO.exists():
    VIDEO = Path("/data/22/IMG_5117.MOV")
if not VIDEO.exists():
    # fallback: copy path used by upload
    cands = list(Path(REPO, "backend/uploaded_videos").glob("*IMG_5117.MOV"))
    VIDEO = cands[-1] if cands else VIDEO

print("VIDEO", VIDEO, "exists", VIDEO.exists(), flush=True)

src = VideoFileSource(str(VIDEO))
det = YoloDetector()
tracker = BytetrackTracker()
pose = MovenetPose()
pipe = InferencePipeline(PipelineConfig(analysis_fps=8.0, auto_tune=True))

rows = []
frame_idx = 0
while True:
    ok, frame = src.read()
    if not ok or frame is None:
        break
    ts = src.timestamp_sec() if hasattr(src, "timestamp_sec") else frame_idx / 30.0
    if not pipe.should_analyze(ts if ts else frame_idx / 30.0):
        frame_idx += 1
        continue
    persons, objects = build_tracks_real(det, tracker, pose, frame, frame_idx, ts)
    before = {k: (m.state.value, m.carried_frames, m.release_frame, m.ever_person_moved_while_carried,
                  m.desync_release_streak, m.ever_off_ground_while_carried)
              for k, m in pipe.event_detector._pairs.items()} if hasattr(pipe, "event_detector") else {}
    # Adaptive wrapper
    ed = getattr(pipe, "event_detector", None) or getattr(pipe, "_event_detector", None)
    core = ed
    if hasattr(ed, "_tiers"):
        core = ed._tiers[0]
    elif hasattr(ed, "primary"):
        core = ed.primary
    elif hasattr(ed, "_detectors"):
        core = ed._detectors[0]

    pipe.process_frame(frame, persons, objects, timestamp=ts, frame_index=frame_idx)

    # find core detector pairs after update
    core = pipe.event_detector
    if hasattr(core, "primary_detector"):
        core = core.primary_detector
    if hasattr(core, "_primary"):
        core = core._primary
    if hasattr(core, "detectors"):
        core = core.detectors[0]
    if hasattr(core, "_tiers") and core._tiers:
        core = core._tiers[0][1] if isinstance(core._tiers[0], tuple) else core._tiers[0]

    pairs = getattr(core, "_pairs", {})
    for k, m in pairs.items():
        rows.append({
            "f": frame_idx,
            "key": str(k),
            "state": m.state.value,
            "carried_f": m.carried_frames,
            "rel": m.release_frame,
            "ever_moved": m.ever_person_moved_while_carried,
            "ever_off": m.ever_off_ground_while_carried,
            "desync": m.desync_release_streak,
            "stat_f": m.stationary_frames,
        })
    frame_idx += 1

summary = pipe.event_detector.summary() if hasattr(pipe.event_detector, "summary") else {}
out = REPO / "project_audit" / "_diag_5117.json"
# unwrap adaptive
try:
    summary = pipe.event_detector.summary()
except Exception as e:
    summary = {"err": str(e)}

payload = {
    "video": str(VIDEO),
    "frames_analyzed": frame_idx,
    "summary": summary,
    "confirmed": [e.to_dict() for e in getattr(pipe.event_detector, "confirmed_events", [])],
    "rejected": [e.to_dict() for e in getattr(pipe.event_detector, "rejected_events", [])],
    "pair_rows_tail": rows[-40:],
    "detector_type": type(pipe.event_detector).__name__,
}
# try adaptive internals
ed = pipe.event_detector
payload["ed_attrs"] = [a for a in dir(ed) if not a.startswith("__")][:40]
out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
print("WROTE", out)
print("summary", json.dumps(summary, default=str)[:800])
print("rejected", len(payload["rejected"]), "confirmed", len(payload["confirmed"]))
if payload["rejected"]:
    r = payload["rejected"][0]
    print("rej0", r.get("reason"), r.get("state"), r.get("frames"), r.get("evidence"))
