#!/usr/bin/env python3
"""
Novelty (scene-change) detection — D:\\W evaluation harness.

Runs THREE things per D:\\W video, all sharing the SAME YOLO+ByteTrack person
boxes so the person-masking is faithful:

  1. NoveltyDetector  (this module)        -> "detected_object" change regions
  2. HSV ColorBagTracker (production path) -> yellow_waste_bag (reference ceiling)
  3. YOLO person tracking                  -> person boxes + ids for masking/assoc

Per video it records:
  * HSV yellow raw detections on every-15th frame  (reproduces the ~773 baseline)
  * novelty raw change-region count on every-15th frame (apples-to-apples)
  * novelty PERSISTENT tracks (objects that stayed >= stationary_frames)
  * whether the novelty detector's persistent track overlaps the dropped yellow
    bag (the agreed "reference ceiling" question: does general change-detection
    still find the yellow bag?)
  * distinct HSV persistent tracks

It then writes, per persistent novelty track, an evidence package for visual
review:
  * snapshot.jpg  (frame + magenta box, NO waste class name, just conf)
  * crop.jpg      (zoomed crop of the region)
  * face_evidence.jpg (via FaceEvidenceCapture, only if a person was associated)

Finally it writes NOVELTY_DETECTION_REPORT.md.

STRICT CONSTRAINTS (from the task):
  * Do NOT modify the HSV production path. We only CALL it for comparison.
  * Clarify in all outputs that this DETECTS CHANGE, it does NOT classify waste.
  * This harness is evaluation-only. It is NOT wired into run_pipeline.py.

Run:
  python scripts/test_novelty_dw.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # type: ignore
import numpy as np  # type: ignore

from inference.detection.novelty_detector import NoveltyConfig, NoveltyDetector
from inference.detection.color_bag_detector import ColorBagTracker, ColorBagConfig
from inference.detection.yolo_detector import YoloDetector

D_W = r"D:\W"
VIDEOS = ["IMG_5115.MOV", "IMG_5117.MOV", "IMG_5118.MOV", "IMG_5119.MOV", "IMG_5120.MOV"]
OUT_ROOT = os.path.join("novelty_output")
SAMPLE_EVERY = 15


def _iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / min(a1, a2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=D_W)
    ap.add_argument("--out", default=OUT_ROOT)
    ap.add_argument("--limit", type=int, default=0, help="process only N frames per video (debug)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    results = {}

    for vid in VIDEOS:
        path = os.path.join(args.videos_dir, vid)
        if not os.path.exists(path):
            print(f"[skip] {vid} not found"); continue
        print(f"\n=== {vid} ===")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"[skip] cannot open {vid}"); continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        analysis_every = max(1, int(round((fps or 30.0) / 8.0)))
        print(f"  fps={fps:.1f} frames={total} analysis_every={analysis_every}")

        # fresh detectors per video so tracker state / background don't bleed
        det = YoloDetector(litter_weights="__none__", color_fallback=False,
                           fallback_coco_classes=False, person_conf=0.4)
        det.load()
        nov = NoveltyDetector(NoveltyConfig(enabled=True, bg_frames=25,
                                            stationary_frames=8,
                                            person_mask_pad=0.2))
        color = ColorBagTracker(ColorBagConfig())  # production default: yellow only

        hsv_raw_sampled = 0
        nov_raw_sampled = 0
        hsv_track_ids = set()
        nov_track_ids = set()
        nov_frames_overlap_yellow = 0
        yellow_overlap_found = False
        frame_index = 0
        frame_records = []  # for face evidence: {frame_number, persons:[{track_id,bbox}]}
        # persistent novelty track -> latest evidence frame
        track_evidence = {}  # track_id -> {img, frame_index, bbox, person_track_id, hits}

        src_i = 0
        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # ultralytics model.track() can mutate the input array in place;
            # work on a copy so the HSV/novelty detectors see the pristine frame.
            frm = frame.copy()
            is_analysis = (src_i % analysis_every == 0)
            is_sample = (src_i + 1) % SAMPLE_EVERY == 0

            # Person detection: needed for (a) masking novelty/HSV change on
            # analysis ticks, and (b) faithful person-masking of the novelty RAW
            # sample. Run YOLO on BOTH analysis and every-15th-sample frames so
            # the raw sampling is person-masked exactly like the live detector.
            persons = []
            person_boxes = []
            person_map = {}
            if is_analysis or is_sample:
                # det.track() mutates its input in place; give it its own copy
                # so the pristine `frm` is used for HSV/novelty detection.
                tracked = det.track(frm.copy(), persist=True)
                persons = [t for t in tracked if t.is_person]
                person_boxes = [tuple(t.bbox) for t in persons]
                person_map = {int(t.track_id): tuple(t.bbox) for t in persons}

            if is_analysis:
                frame_index += 1
                nov_out = nov.update(frm, frame_index, person_boxes, person_map)
                color_dets = color.update(frm, frame_index, person_boxes)

                hsv_yellow = [d for d in color_dets if d.class_name == "yellow_waste_bag"]
                hsv_track_ids.update(int(d.track_id) for d in color_dets)
                nov_bboxes = [tuple(o.bbox) for o in nov_out]
                nov_track_ids.update(int(o.track_id) for o in nov_out)

                # overlap test: does novelty find the dropped yellow bag?
                for nb in nov_bboxes:
                    cx, cy = (nb[0] + nb[2]) / 2.0, (nb[1] + nb[3]) / 2.0
                    hit = False
                    for yd in hsv_yellow:
                        yb = yd.bbox
                        if (yb[0] <= cx <= yb[2] and yb[1] <= cy <= yb[3]) or _iou(nb, yb) > 0.1:
                            hit = True
                            break
                    if hit:
                        yellow_overlap_found = True
                        nov_frames_overlap_yellow += 1

                # record for face evidence (only persons)
                frame_records.append({
                    "frame_number": src_i,
                    "persons": [{"track_id": int(t.track_id), "bbox": [float(v) for v in t.bbox]}
                                for t in persons],
                })

                # keep latest frame per persistent novelty track for artifacts
                for o in nov_out:
                    tid = int(o.track_id)
                    track_evidence[tid] = {
                        "img": frm.copy(),
                        "frame_index": src_i,
                        "bbox": tuple(float(v) for v in o.bbox),
                        "person_track_id": None,  # filled below
                        "conf": float(o.confidence),
                    }

            if is_sample:
                # RAW HSV contours (stateless ColorBagTracker._detect_raw, same
                # 15-frame cadence as eval_hsv_baseline.py -> reproduces the 773
                # reference). Independent of the analysis throttle, so the count
                # is directly comparable to the production baseline.
                hsv_raw = color._detect_raw(frm)
                hsv_raw_sampled += sum(1 for c in hsv_raw if c[3] == "yellow_waste_bag")
                # RAW novelty change regions (stateless sample). Requires the
                # background anchor (_bg); returns 0 during warmup (first ~25
                # analysis ticks), which is correct -- novelty cannot detect
                # before it has a scene anchor.
                nov_raw_sampled += nov.raw_region_count(frm, person_boxes)

            if args.limit and src_i >= args.limit:
                break
            src_i += 1
        cap.release()
        dt = time.time() - t0
        print(f"  processed {src_i} frames in {dt:.1f}s | analysis ticks={frame_index}")
        print(f"  HSV yellow raw (sampled/15): {hsv_raw_sampled}")
        print(f"  Novelty raw regions (sampled/15): {nov_raw_sampled}")
        print(f"  HSV persistent tracks: {len(hsv_track_ids)} | Novelty persistent tracks: {len(nov_track_ids)}")
        print(f"  Novelty overlaps yellow bag: {yellow_overlap_found} "
              f"({nov_frames_overlap_yellow} analysis frames)")

        # ---- write evidence artifacts per persistent novelty track ----
        vid_out = os.path.join(args.out, vid.replace(".MOV", ""))
        os.makedirs(vid_out, exist_ok=True)
        face_capture = None
        try:
            from inference.evidence.face_evidence import FaceEvidenceCapture
            face_capture = FaceEvidenceCapture(backend="retinaface")
        except Exception as e:
            print(f"  [warn] face capture unavailable: {e}")

        track_artifacts = []
        for tid, ev in track_evidence.items():
            tdir = os.path.join(vid_out, f"track_{tid}")
            os.makedirs(tdir, exist_ok=True)
            img = ev["img"].copy()
            x1, y1, x2, y2 = [int(v) for v in ev["bbox"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            # HONESTY: no waste class name — only "detected_object" + confidence
            cv2.putText(img, f"detected_object {ev['conf']:.2f}",
                        (max(0, x1), max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            snap_path = os.path.join(tdir, "snapshot.jpg")
            cv2.imwrite(snap_path, img)

            h, w = img.shape[:2]
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            pad = 0.25
            cx1 = max(0, int(x1 - bw * pad)); cy1 = max(0, int(y1 - bh * pad))
            cx2 = min(w, int(x2 + bw * pad)); cy2 = min(h, int(y2 + bh * pad))
            crop = img[cy1:cy2, cx1:cx2]
            crop_path = os.path.join(tdir, "crop.jpg")
            if crop.size > 0:
                cv2.imwrite(crop_path, crop)

            face_path = None
            # associate the nearest person at this frame for face evidence
            pid = _nearest_person(ev["bbox"], frame_records, ev["frame_index"])
            ev["person_track_id"] = pid
            if pid is not None and face_capture is not None:
                try:
                    res = face_capture.capture_best(
                        path, frame_records, pid,
                        frame_window=(ev["frame_index"], ev["frame_index"]),
                        out_dir=tdir, out_name="face_evidence.jpg")
                    if res.get("captured"):
                        face_path = res.get("face_evidence_path")
                except Exception as e:
                    print(f"    [warn] face capture failed for track {tid}: {e}")

            track_artifacts.append({
                "track_id": tid,
                "bbox": [int(v) for v in ev["bbox"]],
                "confidence": round(ev["conf"], 3),
                "person_track_id": pid,
                "snapshot": snap_path,
                "crop": crop_path,
                "face_evidence": face_path,
            })

        with open(os.path.join(vid_out, "tracks.json"), "w", encoding="utf-8") as f:
            json.dump(track_artifacts, f, indent=2)

        results[vid] = {
            "fps": round(fps, 2),
            "total_frames": total,
            "analysis_ticks": frame_index,
            "hsv_yellow_raw_sampled": hsv_raw_sampled,
            "novelty_raw_regions_sampled": nov_raw_sampled,
            "hsv_persistent_tracks": len(hsv_track_ids),
            "novelty_persistent_tracks": len(nov_track_ids),
            "novelty_overlaps_yellow": yellow_overlap_found,
            "novelty_frames_overlapping_yellow": nov_frames_overlap_yellow,
            "artifacts_dir": vid_out,
            "tracks": track_artifacts,
        }
        print(f"  wrote {len(track_artifacts)} track artifact packages -> {vid_out}")

    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] results -> {os.path.join(args.out, 'results.json')}")
    return results


def _nearest_person(bbox, frame_records, frame_index, max_dist=400.0):
    """Find the person track id present at frame_index nearest to the region."""
    rec = None
    for r in frame_records:
        if r["frame_number"] == frame_index:
            rec = r; break
    if rec is None or not rec["persons"]:
        return None
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    best = None; best_d = max_dist
    for p in rec["persons"]:
        pb = p["bbox"]
        px, py = (pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0
        d = float(np.hypot(cx - px, cy - py))
        if d < best_d:
            best_d = d; best = int(p["track_id"])
    return best


if __name__ == "__main__":
    main()
