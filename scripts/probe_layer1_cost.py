#!/usr/bin/env python3
"""Probe real per-frame cost of the Layer 1 production path."""
from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, ".")
from inference.capture.camera_source import VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
from scripts.run_pipeline import build_tracks_real

p = sys.argv[1] if len(sys.argv) > 1 else r"D:\22\IMG_5303.MOV"
n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 40

src = VideoFileSource(p)
assert src.open()
total = src.total_frames
fps = src.fps
print(f"video={os.path.basename(p)} frames={total} fps={fps} dur={src.duration_seconds}", flush=True)

det = YoloDetector(); det.load(); det.reset_tracking()
trk = BytetrackTracker(); trk.load()
mov = MovenetPose(); mov.load()
ncfg = NoveltyConfig.from_yaml()
nov = NoveltyDetector(ncfg) if ncfg.enabled else None
print(f"models loaded nov={nov is not None}", flush=True)

pers = set(); objs = set(); sources = {}
t0 = time.time(); times = []
for i, pkt in enumerate(src):
    if i >= n_frames:
        break
    a = time.time()
    tracked = det.track(pkt.frame, persist=True)
    rp = True  # run pose each probed frame to measure full cost
    persons, objects = build_tracks_real(pkt.frame, tracked, mov, trk, i, run_pose=rp, nov=nov)
    dt = time.time() - a
    times.append(dt)
    for pp in persons:
        pers.add(int(pp.track_id))
    for o in objects:
        objs.add(int(o.track_id))
        s = getattr(o, "source", "yolo")
        sources[s] = sources.get(s, 0) + 1
src.release()
wall = time.time() - t0
print(json.dumps({
    "frames": len(times),
    "wall_sec": round(wall, 2),
    "mean_ms": round(sum(times)/len(times)*1000, 1),
    "max_ms": round(max(times)*1000, 1),
    "median_ms": round(sorted(times)[len(times)//2]*1000, 1),
    "distinct_persons": sorted(pers),
    "distinct_objects": sorted(objs),
    "object_sources_seen": sources,
}, indent=2), flush=True)
