#!/usr/bin/env python3
"""Layer-1 visual proof: real detection/tracking overlay on sampled frames.

Runs the REAL production detection path (same code as the audit) on sampled
analysis ticks of the chosen video and writes annotated PNGs showing:
  - person boxes (label PERSON #id, yolo),
  - HSV/color object boxes (label <class> #id, color),
  - Novelty boxes (label detected_object #id, novelty — NOT "waste").
This is the Layer-1 visual evidence that every stable entity gets a real box +
stable id, and that a scene-change (novelty) box is never labelled WASTE.

Usage:
    python scripts/layer1_visualize.py --video IMG_5290 --frames 0,200,400,600
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2  # noqa: E402

from inference.capture.camera_source import VideoFileSource  # noqa: E402
from inference.detection.yolo_detector import YoloDetector  # noqa: E402
from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig  # noqa: E402
from inference.pose.movenet_pose import MovenetPose  # noqa: E402
from inference.tracking.bytetrack_tracker import BytetrackTracker  # noqa: E402
from scripts.run_pipeline import build_tracks_real  # noqa: E402

LOG = r"D:\HO\layer1_runs\visual\run.log"
os.makedirs(os.path.dirname(LOG), exist_ok=True)


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")
        f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", default=r"D:\HO\layer1_runs\visual")
    args = ap.parse_args()
    try:
        _run(args)
    except Exception as e:
        import traceback
        log("ERROR: " + traceback.format_exc())


def _run(args):
    path = os.path.join(r"D:\22", args.video + (".MOV" if not args.video.upper().endswith("MOV") else ""))
    frames_want = set(int(x) for x in args.frames.split(","))
    src = VideoFileSource(path)
    if not src.open():
        log(f"cannot open {path}")
        return
    os.makedirs(args.out, exist_ok=True)
    det = YoloDetector(); det.load(); det.reset_tracking()
    trk = BytetrackTracker(); trk.load()
    mov = MovenetPose(); mov.load()
    ncfg = NoveltyConfig.from_yaml()
    nov = NoveltyDetector(ncfg) if ncfg.enabled else None
    log(f"drawing on {sorted(frames_want)} of {args.video}; nov={'on' if nov else 'off'}")
    wrote = 0
    for i, pkt in enumerate(src):
        if i not in frames_want:
            continue
        tracked = det.track(pkt.frame, persist=True)
        persons, objects = build_tracks_real(pkt.frame, tracked, mov, trk, i,
                                             run_pose=(i % 6 == 0), nov=nov)
        img = pkt.frame.copy()
        for p in persons:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, f"PERSON {p.track_id} [yolo]", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for o in objects:
            src_s = str(getattr(o, "source", "yolo") or "yolo")
            col = {"color": (255, 200, 0), "novelty": (0, 200, 255), "yolo": (0, 255, 255)}[src_s]
            x1, y1, x2, y2 = (int(v) for v in o.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
            label = f"{o.class_name} {o.track_id} [{src_s}]"
            cv2.putText(img, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        out = os.path.join(args.out, f"{args.video}_f{i:04d}.jpg")
        cv2.imwrite(out, img)
        wrote += 1
        log(f"f{i}: persons={[(p.track_id,) for p in persons]} "
            f"objects={[(o.class_name, o.track_id, getattr(o,'source','yolo')) for o in objects]}")
    src.release()
    log("done; wrote " + str(wrote))


if __name__ == "__main__":
    main()