"""
Auto-label generator using Grounding DINO (open-grounded-dino-base).

Generates preliminary bounding boxes on FAILING video frames by prompting
Grounding DINO with text cues like "crumpled tissue paper" and "small white litter".
These boxes are NOT final labels — they are starting points for manual review
and YOLO training. The canonical class is assigned by unified_class_map.py.

Usage:
    python scripts/datasets/auto_label_grounding_dino.py \\
        --video path/to/failing_video.mp4 \\
        --output labels/ \\
        --cues "crumpled tissue paper" "small white litter" "plastic bottle"

Requirements:
    pip install groundingdino
    (or use the transformers + groundiNodino pipeline)

Outputs one .txt per frame in YOLO format: class_id cx cy w h.
The class_id comes from unified_class_map.canonical_id(cue_text).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.datasets.unified_class_map import canonical_id, CANONICAL_CLASSES


def _frame_paths(video_path: str) -> List[Tuple[str, "object"]]:
    """Yield (timestamp, frame) pairs from a video file using cv2."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield (idx, frame)
        idx += 1
    cap.release()


def _grounding_dino_predict(frame, cues: List[str], model=None, processor=None):
    """Run Grounding DINO on a single frame with text cues.

    Returns list of (cue_text, confidence, bbox) tuples.
    Lazily loads the model on first call.
    """
    # Lazy import — heavyweight
    from groundingdino.util.inference import load_model, predict
    # or from transformers import pipeline
    raise NotImplementedError(
        "Grounding DINO must be installed: pip install groundingdino\n"
        "Or use: from groundingdino import GroundingDINO\n"
        "This script is a scaffold — the actual inference call depends\n"
        "on which Grounding DINO package is available in the environment.\n"
        "On the laptop, install groundingdino and uncomment the real call below.\n\n"
        "See docs/research_alignment_audit.md for the full pipeline."
    )

    # --- REAL implementation (uncomment after pip install groundingdino) ---
    # from groundingdino.util.inference import load_model, predict, load_image
    # model = load_model("IDEA-Research/grounding-dino-base", device="cpu")
    # results = predict(model, frame, cues)
    # return [(c, conf, box) for c, conf, box in results]
    return []


def auto_label_video(
    video_path: str,
    output_dir: str,
    cues: List[str],
) -> int:
    """Generate YOLO-format label files for each frame of a FAILING video.

    Uses Grounding DINO to produce preliminary boxes; the canonical
    class_id is assigned from the cue text via unified_class_map.

    Returns the number of label files written.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    count = 0

    for idx, frame in _frame_paths(video_path):
        results = _grounding_dino_predict(frame, cues)
        if not results:
            continue
        h, w = frame.shape[:2]
        txt_path = output / f"frame_{idx:06d}.txt"
        with open(txt_path, "w") as f:
            for cue_text, conf, (x1, y1, x2, y2) in results:
                cls = canonical_id(cue_text)
                if cls < 0:
                    continue
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        count += 1

    return count


def main():
    ap = argparse.ArgumentParser(description="Auto-label FAILING frames with Grounding DINO")
    ap.add_argument("--video", required=True, help="path to the FAILING video to label")
    ap.add_argument("--output", default="labels", help="output directory for .txt label files")
    ap.add_argument("--cues", nargs="+", default=["crumpled tissue paper", "small white litter", "plastic bottle"],
                       help="text cues for Grounding DINO")
    args = ap.parse_args()

    # SCAFFOLD GUARD: the actual Grounding DINO inference call is intentionally
    # not implemented here (it requires `pip install groundingdino` plus a model
    # download). Running this without that implementation must NOT silently write
    # empty labels and claim success — that would poison any training set. We
    # therefore surface the limitation clearly and exit non-zero.
    try:
        n = auto_label_video(args.video, args.output, args.cues)
    except NotImplementedError as exc:
        print("ERROR: auto-label inference is a SCAFFOLD and is not wired to a real "
              "Grounding DINO backend yet.", file=sys.stderr)
        print(str(exc).strip(), file=sys.stderr)
        print("To enable: install groundingdino, then implement _grounding_dino_predict() "
              "and re-run. Do NOT use empty outputs as training labels.", file=sys.stderr)
        sys.exit(2)

    if n == 0:
        print("WARNING: wrote 0 label files — review cues / detector before using as labels.", file=sys.stderr)
    else:
        print(f"[OK] Wrote {n} label files to {args.output}/")
    print(f"  Cue → canonical class mapping:")
    for cue in args.cues:
        cid = canonical_id(cue)
        name = CANONICAL_CLASSES[cid] if 0 <= cid < len(CANONICAL_CLASSES) else "UNMAPPED"
        print(f"    '{cue}' → {cid}: {name}")


if __name__ == "__main__":
    main()
