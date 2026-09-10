#!/usr/bin/env python3
"""Phase-1 before/after capture for IMG_5302.MOV — real production path."""
import os, sys, time, json, gc
sys.path.insert(0, "/app")

import cv2
from inference.capture.camera_source import VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
from inference.pipeline import InferencePipeline, PipelineConfig
from inference.visualization import build_frame_analysis, render_analysis_frame
from scripts.run_pipeline import build_tracks_real

VIDEO = "/tmp/IMG_5302.MOV"
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/phase1_before"
os.makedirs(OUT_DIR, exist_ok=True)

# Load models
detector = YoloDetector()
detector.load()
tracker = BytetrackTracker()
tracker.load()
nov_cfg = NoveltyConfig.from_yaml()
nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
movenet = MovenetPose()
movenet.load()

cfg = PipelineConfig(buffer_seconds=8.0, analysis_fps=8.0, camera_id="phase1", post_backend_url=None, auto_tune=True)
pipe = InferencePipeline(cfg)
pipe.event_detector.reset()
try:
    pipe.event_detector.learning_video = "IMG_5302.MOV"
except: pass

source = VideoFileSource(VIDEO)
assert source.open(), f"cannot open {VIDEO}"
print(f"Opened {VIDEO}: {source.total_frames} frames, {source.fps} fps, {source.duration_seconds:.1f}s")

# Frames to capture (spread across video)
capture_frames = [60, 150, 300]
if source.total_frames < 350:
    capture_frames = [max(1, source.total_frames//4), max(1, source.total_frames//2), max(1, int(source.total_frames*0.75))]

t_start = time.time()
frame_idx = 0
first_ts = None
records = []
inference_time = 0
render_time = 0

import pathlib
for pkt in source:
    if first_ts is None:
        first_ts = float(pkt.timestamp)
    infer_start = time.time()
    tracked = detector.track(pkt.frame, persist=True)
    run_pose = pipe.should_analyze(pkt.timestamp)
    persons, objects = build_tracks_real(pkt.frame, tracked, movenet, tracker, frame_idx, run_pose=run_pose, nov=nov)
    new_events = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
    inference_time += time.time() - infer_start

    analysis = build_frame_analysis(persons, objects, pipe.event_detector, tracker, float(pkt.timestamp), frame_idx, source_fps=float(source.fps or 0), analysis_fps=float(cfg.analysis_fps), video_name="IMG_5302.MOV", event=None)
    if run_pose:
        records.append(analysis.to_dict())

    render_start = time.time()
    try:
        annotated = render_analysis_frame(pkt.frame, analysis, event=None)
    except Exception as e:
        print(f"render failed at {frame_idx}: {e}")
        annotated = pkt.frame
    render_time += time.time() - render_start

    if frame_idx in capture_frames:
        out_path = os.path.join(OUT_DIR, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(out_path, annotated)
        # Box accounting
        person_boxes = len(analysis.persons)
        waste_boxes = len([o for o in analysis.objects if str(o.source).lower()=="yolo" and not str(o.class_name).lower().startswith("color_candidate") and str(o.class_name).lower()!="detected_object"])
        proposal_boxes = len(analysis.objects) - waste_boxes
        total = person_boxes + len(analysis.objects)
        info = {
            "frame": frame_idx,
            "timestamp": round(float(pkt.timestamp - first_ts), 2),
            "person_boxes": person_boxes,
            "waste_boxes": waste_boxes,
            "proposal_boxes": proposal_boxes,
            "total": total,
            "active_tracks_persons": len(persons),
            "active_tracks_objects": len(objects),
            "rendered_persons": person_boxes,
            "rendered_objects": len(analysis.objects),
        }
        with open(os.path.join(OUT_DIR, f"boxcount_{frame_idx:04d}.json"), "w") as f:
            json.dump(info, f, indent=2)
        print(f"CAPTURE frame {frame_idx}: persons={person_boxes} waste={waste_boxes} proposals={proposal_boxes} total={total} | tracks p={len(persons)} o={len(objects)}")

    frame_idx += 1
    if frame_idx % 30 == 0:
        gc.collect()

source.release()
final_events = pipe.finalize(source.duration_seconds)
elapsed = time.time() - t_start
print(f"Done {frame_idx} frames in {elapsed:.1f}s fps={frame_idx/elapsed:.1f} inference_fps={frame_idx/max(1e-6,inference_time):.1f} render_fps={frame_idx/max(1e-6,render_time):.1f}")
print(f"Events confirmed: {len(pipe.events)} rejected: {len(pipe.event_detector.rejected_events)}")
# Save summary
summary = {
    "total_frames": frame_idx,
    "elapsed_sec": round(elapsed,2),
    "processing_fps": round(frame_idx/elapsed,2),
    "inference_fps": round(frame_idx/max(1e-6,inference_time),2),
    "render_fps": round(frame_idx/max(1e-6,render_time),2),
    "confirmed_events": len(pipe.events),
    "events": [{"event_id": e.event_id, "person": e.person_track_id, "object": e.object_track_id, "conf": e.confidence} for e in pipe.events],
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {OUT_DIR}/summary.json")
