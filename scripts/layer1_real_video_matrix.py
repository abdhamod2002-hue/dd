"""Real-video Layer-1 validation matrix (Steps 24-26).

Runs the PRODUCTION path (YoloDetector.track -> person + semantic bag model +
HSV color candidates + ByteTrack) on the required D:\\22 videos, sampled at
the analysis cadence, and reports per-video counts by detector source, ID
switches, and re-associations. No event logic is touched.
"""
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, r"D:\HO")
os.chdir(r"D:\HO")

from inference.detection.yolo_detector import YoloDetector
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.capture.camera_source import VideoFileSource

VIDEOS = ["IMG_5305", "IMG_5306", "IMG_5299", "IMG_5302"]
BASE = r"D:\22"
SAMPLE = 15  # analysis cadence: every 15th source frame

det = YoloDetector()
det.load()
print("bag_weights:", det.bag_weights, "classes:", det._bag_classes, flush=True)
trk = BytetrackTracker()

for vid in VIDEOS:
    path = os.path.join(BASE, vid + ".MOV")
    src = VideoFileSource(path)
    if not src.open():
        print(vid, "CANNOT OPEN", flush=True)
        continue
    person_ids, obj_ids = set(), set()
    switch_count = 0
    prev_obj_raw = {}
    src_counts = Counter()
    cls_counts = Counter()
    obj_tracks = defaultdict(lambda: {"first": None, "last": None, "n": 0})
    fi = -1
    while True:
        pkt = src.read()
        if pkt is None:
            break
        frame = pkt.frame
        fi = pkt.frame_index
        if fi % SAMPLE:
            continue
        tracked = det.track(frame, persist=True)
        trk.update(tracked, frame_index=fi)
        persons, objects = trk.to_tracks(tracked)
        for t in persons:
            person_ids.add(t.track_id)
        for t in objects:
            obj_ids.add(t.track_id)
            src_counts[t.source] += 1
            cls_counts[t.class_name] += 1
            rec = obj_tracks[t.track_id]
            if rec["first"] is None:
                rec["first"] = fi
            rec["last"] = fi
            rec["n"] += 1
            raw = t.track_id % 100000  # strip namespace offset
            prev = prev_obj_raw.get(raw)
            if prev is not None and t.track_id != prev:
                switch_count += 1
            prev_obj_raw[raw] = t.track_id
    src.release()
    det.reset_tracking()
    trk = BytetrackTracker()
    durs = sorted((r["last"] - r["first"]) for r in obj_tracks.values()) if obj_tracks else []
    print(json.dumps({
        "video": vid,
        "sampled_person_track_ids": len(person_ids),
        "object_track_ids": len(obj_ids),
        "detections_by_source": dict(src_counts),
        "detections_by_class": dict(cls_counts),
        "raw_id_switches": switch_count,
        "object_track_duration_frames_min_median_max": [durs[0], durs[len(durs)//2], durs[-1]] if durs else [],
    }), flush=True)
print("VALIDATION_DONE")
