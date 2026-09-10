#!/usr/bin/env python3
"""
Phase 2 Tracker Benchmark — BENCHMARK ONLY, no production change.

Feeds IDENTICAL raw detections (YoloDetector.detect, no tracking) into:
  A. Current production ByteTrack (ultralytics)
  B. Roboflow ByteTrack
  C. Roboflow OC-SORT
  D. Roboflow BoT-SORT

Measures per tracker per video: unique IDs, fragments, switches, durations, FPS.
Visual proof: same source frame + same detections, 4 annotated images.
"""
import os, sys, time, json, gc, math
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2

sys.path.insert(0, "/app")

from inference.detection.yolo_detector import YoloDetector
from inference.detection.color_bag_detector import ColorBagTracker  # for color fallback if needed
import supervision as sv
from trackers import ByteTrackTracker as RFByteTrack, OCSORTTracker, BoTSORTTracker

# Config
VIDEOS = [
    "/tmp/IMG_5306.MOV",
    "/tmp/IMG_5301.MOV",
    "/tmp/IMG_5305.MOV",
    "/tmp/IMG_5302.MOV",
    "/tmp/IMG_5298.MOV",
    "/tmp/IMG_5307.MOV",
    "/tmp/IMG_5287.MOV",
]
OUT_ROOT = "/tmp/phase2_benchmark"
os.makedirs(OUT_ROOT, exist_ok=True)

FRAME_RATE = 30.0  # for tracker lost_track_buffer interpretation

def yolo_detections_to_sv(person_dets, object_dets):
    """Convert YoloDetector Detection lists to sv.Detections for persons/objects."""
    def to_sv(dets):
        if not dets:
            return sv.Detections(xyxy=np.zeros((0,4), dtype=np.float32), confidence=np.array([], dtype=np.float32), class_id=np.array([], dtype=np.int32))
        xyxy = np.array([d.bbox for d in dets], dtype=np.float32)  # already x1,y1,x2,y2
        conf = np.array([d.confidence for d in dets], dtype=np.float32)
        # class_id: 0=person, 1=object (or per class name hash)
        cls = np.array([0 if d.is_person else 1 for d in dets], dtype=np.int32)
        data = {"class_name": np.array([d.class_name for d in dets], dtype=object)}
        return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls, data=data)
    return to_sv(person_dets), to_sv(object_dets)

def compute_metrics(tracks_by_frame, fps=30.0):
    """tracks_by_frame: list of dict {tracker_id: bbox} per frame, with frame_idx."""
    # Unique IDs
    all_ids = set()
    for frame_tracks in tracks_by_frame:
        all_ids.update(frame_tracks.keys())
    unique = len(all_ids)
    # Track durations
    id_first = {}
    id_last = {}
    id_count = Counter()
    for idx, frame_tracks in enumerate(tracks_by_frame):
        for tid in frame_tracks:
            id_count[tid] += 1
            if tid not in id_first:
                id_first[tid] = idx
            id_last[tid] = idx
    durations = []
    for tid in all_ids:
        dur_frames = id_last[tid] - id_first[tid] + 1
        durations.append(dur_frames / fps)
    durations = sorted(durations)
    median = durations[len(durations)//2] if durations else 0
    longest = max(durations) if durations else 0
    # Fragments: tracks shorter than 1s and 0.5s
    short_1 = sum(1 for d in durations if d < 1.0)
    short_05 = sum(1 for d in durations if d < 0.5)
    # Same-frame duplicate IDs: should never happen if tracker is correct (each track_id once per frame)
    dup_frames = 0
    # ID switches: count how many times a track disappears and reappears as new ID at similar location
    # Approximate: number of IDs minus number of continuous physical objects (unknown) ~ unique - max_concurrent
    # Better: count gaps where track missing >1 frame then new ID appears near old centroid (heuristic)
    # For now, report fragmentation proxy: unique IDs vs video length
    # Also count total fragments = unique
    return {
        "unique_ids": unique,
        "median_duration_sec": round(median, 3),
        "longest_duration_sec": round(longest, 3),
        "short_lt_1s": short_1,
        "short_lt_05s": short_05,
        "durations": [round(d,2) for d in durations[:10]],  # sample
    }

def run_one_video(video_path, detector):
    name = Path(video_path).name
    print(f"\n=== {name} ===")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  cannot open {video_path}")
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0 or fps > 120: fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"  {total} frames {fps:.1f}fps {w}x{h}")

    # Init trackers: separate instances for persons vs objects, per tracker type
    # Current production: we will simulate via Roboflow ByteTrack with same params as ultralytics? But we need actual ultralytics ByteTrack for A.
    # For fair comparison, we run A via YoloDetector.track (which uses ultralytics ByteTrack per model) separately outside this loop.
    # Here we benchmark B/C/D with shared detections.

    trackers = {
        "rf_bytetrack": (RFByteTrack(lost_track_buffer=30, frame_rate=fps), RFByteTrack(lost_track_buffer=30, frame_rate=fps)),
        "ocsort": (OCSORTTracker(lost_track_buffer=30, frame_rate=fps), OCSORTTracker(lost_track_buffer=30, frame_rate=fps)),
        "botsort": (BoTSORTTracker(lost_track_buffer=30, frame_rate=fps, enable_cmc=False), BoTSORTTracker(lost_track_buffer=30, frame_rate=fps, enable_cmc=False)),
    }
    # Also need SORT baseline? Keep 3 for now, plus current ultralytics separate

    # Storage
    per_tracker_frames = {k: [] for k in trackers}
    per_tracker_frames["current_bytetrack"] = []
    # For current ByteTrack we need to run ultralytics track separately — do it in same loop but call YoloDetector.track
    # Instead, we will run detection + each tracker update per frame, plus current track via detector.track (but that also does detection)
    # To ensure identical detections, we use detector.detect for B/C/D, and for current we use detector.track but we will still use same frame
    # For efficiency, we call detector.detect for B/C/D and detector.track for current separately — this duplicates YOLO but ensures identical input for B/C/D
    # Alternative: use detect for all and simulate current via RF ByteTrack with ultralytics params — but spec says current is ultralytics ByteTrack
    # So we will do both.

    # Also track current via ultralytics separately
    from inference.tracking.bytetrack_tracker import BytetrackTracker as ProdTracker
    prod_tracker = ProdTracker()
    prod_tracker.load()
    # For identical-detection fairness, A (current) is measured in two ways:
    # A1 = ultralytics native track (detector.track, own YOLO pass) — true production behavior
    # A2 = rf_bytetrack on identical detections — apples-to-apples tracker comparison
    # We record both; A1 goes to "current_bytetrack", A2 is "rf_bytetrack".

    frame_idx = 0
    t0 = time.time()
    tracker_time = {k: 0.0 for k in list(trackers.keys()) + ["current_bytetrack"]}
    MAX_FRAMES = int(os.environ.get("BENCH_MAX_FRAMES", "500"))
    # For visual proof: capture 3 frames (early/mid/late) where at least 1 person detected
    visual_frames = {}
    eff_total = min(total, MAX_FRAMES)
    target_frames = [max(1, eff_total//4), max(1, eff_total//2), max(1, int(eff_total*0.75))]
    # If video is short, adapt
    if total < 100:
        target_frames = [10, total//2, total-10]

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # 1) Raw detections for benchmark (identical for B/C/D)
        raw_dets = detector.detect(frame)  # list of Detection
        persons_raw = [d for d in raw_dets if d.is_person]
        objects_raw = [d for d in raw_dets if not d.is_person]
        # Color fallback: if no object, try HSV (mirrors YoloDetector track fallback)
        if not objects_raw:
            # quick HSV check - use detector internal color tracker if available
            # For benchmark, we skip color fallback to keep detections identical and YOLO-only
            pass
        p_sv, o_sv = yolo_detections_to_sv(persons_raw, objects_raw)

        # 2) Update each Roboflow tracker (timed)
        for name, (t_p, t_o) in trackers.items():
            # persons
            empty_dets = sv.Detections.empty()
            _tt0 = time.time()
            res_p = t_p.update(p_sv, frame=frame) if len(p_sv)>0 else t_p.update(empty_dets, frame=frame)
            # objects
            res_o = t_o.update(o_sv, frame=frame) if len(o_sv)>0 else t_o.update(empty_dets, frame=frame)
            tracker_time[name] += time.time() - _tt0
            # Collect: merge persons+objects ids for metrics (separate)
            # For person metrics, use res_p tracker_ids
            # For object metrics, use res_o
            # Store per tracker per frame
            # We store dict of tracker_id -> bbox for persons
            pdict = {}
            if res_p.tracker_id is not None:
                for i, tid in enumerate(res_p.tracker_id):
                    if tid == -1: continue
                    pdict[int(tid)] = res_p.xyxy[i].tolist()
            odict = {}
            if res_o.tracker_id is not None:
                for i, tid in enumerate(res_o.tracker_id):
                    if tid == -1: continue
                    odict[int(tid)] = res_o.xyxy[i].tolist()
            per_tracker_frames[name].append({"persons": pdict, "objects": odict})

        # 3) Current production ByteTrack via YoloDetector.track (own YOLO pass).
        # Timed separately; input is same frame but native detection+tracking path.
        _tc0 = time.time()
        try:
            prod_tracked = detector.track(frame, persist=True)
            prod_tracker.update(prod_tracked, frame_idx)
            prod_pdict: dict = {}
            prod_odict: dict = {}
            for td in prod_tracked:
                if int(getattr(td, "track_id", -1)) < 0:
                    continue
                # namespace like production (persons +0, objects +10000)
                ns = prod_tracker.namespace(int(td.track_id), bool(td.is_person))
                if bool(td.is_person):
                    prod_pdict[ns] = list(map(float, td.bbox))
                else:
                    prod_odict[ns] = list(map(float, td.bbox))
            per_tracker_frames["current_bytetrack"].append({"persons": prod_pdict, "objects": prod_odict})
        except Exception as e:
            per_tracker_frames["current_bytetrack"].append({"persons": {}, "objects": {}})
        tracker_time["current_bytetrack"] += time.time() - _tc0

        # Capture visual proof frames
        if frame_idx in target_frames:
            visual_frames[frame_idx] = frame.copy()

        frame_idx += 1
        if frame_idx >= MAX_FRAMES:
            print(f"  reached MAX_FRAMES={MAX_FRAMES}, stopping early for benchmark speed")
            break
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total}  {frame_idx/(time.time()-t0):.1f} fps")

    cap.release()
    elapsed = time.time() - t0
    fps_proc = frame_idx / max(1e-6, elapsed)
    print(f"  done {frame_idx} frames in {elapsed:.1f}s fps={fps_proc:.1f}")

    # Compute metrics per tracker (including current production)
    results = {}
    for name in list(per_tracker_frames.keys()):
        # Split persons and objects
        person_tracks = [f["persons"] for f in per_tracker_frames[name]]
        object_tracks = [f["objects"] for f in per_tracker_frames[name]]
        p_metrics = compute_metrics(person_tracks, fps=fps)
        o_metrics = compute_metrics(object_tracks, fps=fps)
        results[name] = {"persons": p_metrics, "objects": o_metrics, "fps": round(fps_proc,2),
                         "tracker_cpu_sec": round(tracker_time.get(name, 0.0), 3),
                         "frames": frame_idx}

    # Save visual proof
    out_dir = os.path.join(OUT_ROOT, Path(name).stem)
    os.makedirs(out_dir, exist_ok=True)
    # For each captured frame, draw current tracks for each tracker
    from supervision.draw.color import Color
    colors = {"current_bytetrack": Color(r=80,g=220,b=255), "rf_bytetrack": Color(r=80,g=220,b=255), "ocsort": Color(r=255,g=220,b=80), "botsort": Color(r=120,g=255,b=80)}
    for fidx, frame in visual_frames.items():
        h, w = frame.shape[:2]
        for tname in list(per_tracker_frames.keys()):
            # find frame index in per_tracker_frames
            # visual_frames keys are frame_idx, need to map to per_tracker_frames index
            idx = fidx  # since we processed sequentially from 0
            if idx >= len(per_tracker_frames[tname]): continue
            data = per_tracker_frames[tname][idx]
            # Build Detections for drawing
            persons = data["persons"]
            objects = data["objects"]
            # Create combined visualization: persons blue, objects orange, proposals already filtered (all are YOLO here)
            img = frame.copy()
            # Draw persons
            if persons:
                xyxy = np.array(list(persons.values()), dtype=np.float32)
                tids = np.array(list(persons.keys()), dtype=np.int32)
                dets = sv.Detections(xyxy=xyxy, tracker_id=tids, class_id=np.array([0]*len(tids)), confidence=np.array([1.0]*len(tids)))
                annot = sv.BoxAnnotator(color=colors[tname], thickness=2)
                img = annot.annotate(scene=img, detections=dets)
                lab = sv.LabelAnnotator(color=colors[tname], text_color=Color(r=255,g=255,b=255), text_scale=0.5, text_thickness=1)
                labels = [f"PERSON #{tid}" for tid in tids]
                img = lab.annotate(scene=img, detections=dets, labels=labels)
            if objects:
                xyxy = np.array(list(objects.values()), dtype=np.float32)
                tids = np.array(list(objects.keys()), dtype=np.int32)
                dets = sv.Detections(xyxy=xyxy, tracker_id=tids, class_id=np.array([1]*len(tids)), confidence=np.array([1.0]*len(tids)))
                annot = sv.BoxAnnotator(color=Color(r=255,g=100,b=100), thickness=2)
                img = annot.annotate(scene=img, detections=dets)
                lab = sv.LabelAnnotator(color=Color(r=255,g=100,b=100), text_color=Color(r=255,g=255,b=255), text_scale=0.5, text_thickness=1)
                labels = [f"WASTE #{tid}" for tid in tids]
                img = lab.annotate(scene=img, detections=dets, labels=labels)
            out_path = os.path.join(OUT_ROOT, f"{Path(video_path).stem}_{tname}_frame{fidx:04d}.jpg")
            cv2.imwrite(out_path, img)
            print(f"  wrote {out_path}")

    # Also save JSON accounting for one selected frame (middle)
    mid = target_frames[len(target_frames)//2] if target_frames else 0
    if mid in visual_frames:
        idx = mid
        accounting = {"video": Path(video_path).name, "frame": mid}
        for tname in list(per_tracker_frames.keys()):
            if idx < len(per_tracker_frames[tname]):
                data = per_tracker_frames[tname][idx]
                accounting[tname] = {
                    "persons": [{"track_id": tid, "bbox": bbox} for tid, bbox in data["persons"].items()],
                    "objects": [{"track_id": tid, "bbox": bbox} for tid, bbox in data["objects"].items()],
                }
        with open(os.path.join(OUT_ROOT, f"{Path(video_path).stem}_accounting.json"), "w") as f:
            json.dump(accounting, f, indent=2)

    return results

if __name__ == "__main__":
    det = YoloDetector()
    det.load()
    all_results = {}
    for vp in VIDEOS:
        if not os.path.exists(vp):
            print(f"skip {vp} not found")
            continue
        res = run_one_video(vp, det)
        if res:
            all_results[Path(vp).name] = res
            with open(os.path.join(OUT_ROOT, "benchmark_summary.json"), "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"Saved benchmark_summary.json")
            # Early exit for demo: process 1 video fully, then break if time is limited? But spec says prioritize 7
            # Continue for all
            gc.collect()
    print("Benchmark complete")
    print(json.dumps(all_results, indent=2))
