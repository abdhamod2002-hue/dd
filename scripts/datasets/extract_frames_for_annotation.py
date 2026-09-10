"""Extract representative frames from raw videos for HUMAN annotation.

The project currently has NO labeled dataset for the littering-object classes
(best.pt has no `bag`/`trash_bag` class; the production system relies on the
HSV color fallback). Before a general object detector can be trained, a human
must label real frames. This script samples frames so they can be labeled with
LabelMe / CVAT / Roboflow and converted to YOLO format.

It does NOT auto-generate labels. Auto-labeling (scripts/datasets/
auto_label_grounding_dino.py) is a SCAFFOLD that intentionally raises until a
real Grounding DINO backend is wired in — do not treat its output as labels.

Usage:
    python scripts/datasets/extract_frames_for_annotation.py \
        --src D:/W --out datasets/raw_frames --interval 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402


def extract(src: Path, out: Path, interval: float, max_frames: int) -> int:
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"ERROR: cannot open {src}", file=sys.stderr)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(interval * fps)))
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % step == 0:
            path = out / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        idx += 1
    cap.release()
    return saved


def main() -> int:
    ap = argparse.ArgumentParser(description="Sample frames from videos for manual annotation")
    ap.add_argument("--src", required=True, help="directory of source videos")
    ap.add_argument("--out", default="datasets/raw_frames", help="output root")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between sampled frames")
    ap.add_argument("--max-per-video", type=int, default=200, help="cap frames per video")
    args = ap.parse_args()

    src = Path(args.src)
    total = 0
    for vid in sorted(src.iterdir()):
        if vid.suffix.lower() not in {".mov", ".mp4", ".avi", ".mkv", ".webm"}:
            continue
        dest = Path(args.out) / vid.stem
        n = extract(vid, dest, args.interval, args.max_per_video)
        print(f"[OK] {vid.name}: {n} frames -> {dest}")
        total += n
    print(f"\nTOTAL {total} frames extracted. Label them with LabelMe/CVAT, then convert to "
          f"YOLO txt and run prepare_datasets.py + train_yolo.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
