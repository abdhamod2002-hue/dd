"""
Auto-label generator using Grounding DINO (open-vocabulary object detection).

Phase A2 (NEW_PRODUCTION_ROADMAP.md): bootstrap preliminary YOLO labels for
the canonical classes that no currently-prepared dataset covers at all
(can=0 instances, crumpled_tissue=0, other_litter=0 per the Phase A1 merge
report) plus `person`, by sampling frames from the real site videos in
D:\\22 and prompting Grounding DINO with open-vocabulary text cues.

These boxes are NOT final labels — they are starting points for manual
review before anything is used to train a model. Every run writes a
`_REVIEW_REQUIRED.txt` marker into the output directory as a reminder.

Requirements: `pip install transformers` (already installed for this repo;
uses `IDEA-Research/grounding-dino-tiny`, ~172M params, runs on CPU — this
repo is a confirmed CPU-only deployment, see MASTER_REPAIR_PLAN.md).

Usage (single video):
    python scripts/datasets/auto_label_grounding_dino.py \\
        --video D:/22/IMG_5287.MOV --output datasets/d22_autolabel --fps 1

Usage (all videos in a directory, e.g. all of D:\\22):
    python scripts/datasets/auto_label_grounding_dino.py \\
        --video-dir D:/22 --output datasets/d22_autolabel --fps 1

Outputs, per labeled frame, BOTH the YOLO label .txt AND the source frame
.jpg (the original scaffold only wrote labels — with no paired image, the
output isn't usable as a training set at all).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.datasets.unified_class_map import CANONICAL_CLASSES  # noqa: E402

# Explicit cue -> canonical class id map. NOT unified_class_map.canonical_id():
# that function has no "person" entry in ANY of its five per-dataset maps at
# all (it was built only for the litter side, assuming person comes from the
# separate base YOLO person model), and remap_local_datasets.py already found
# its substring matching order-dependent/fragile for at least one real class
# name. An explicit map here is unambiguous and doesn't depend on either issue.
DEFAULT_CUES: Dict[str, int] = {
    "person": 0,
    "aluminum can": 3,
    "soda can": 3,
    "beverage can": 3,
    "crumpled tissue paper": 6,
    "tissue box": 6,
    "napkin": 6,
    "cigarette butt": 7,
    "food wrapper": 7,
    "small piece of litter": 7,
}

_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_model = None
_processor = None


def _load_model():
    """Lazily load Grounding DINO. Heavyweight (~172M params) — loaded once
    per process, not per frame."""
    global _model, _processor
    if _model is None:
        import torch  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        print(f"Loading {_MODEL_ID} (first call downloads ~350MB, cached after)...", file=sys.stderr)
        _processor = AutoProcessor.from_pretrained(_MODEL_ID)
        _model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID)
        _model.eval()
    return _model, _processor


def _grounding_dino_predict(
    frame_bgr,
    cues: List[str],
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    """Run Grounding DINO on one BGR (OpenCV-order) frame with text cues.

    Returns a list of (cue_text, confidence, (x1, y1, x2, y2)) in pixel
    coordinates, where cue_text is guaranteed to be one of the exact strings
    passed in `cues` (via `text_labels=[cues]`, not free-text span parsing).
    """
    import cv2
    import torch
    from PIL import Image

    model, processor = _load_model()
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    text = ". ".join(cues) + "."
    inputs = processor(images=image, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
        text_labels=[cues],
    )[0]

    out: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
    labels = results.get("text_labels") or results.get("labels")
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        x1, y1, x2, y2 = (float(v) for v in box.tolist())
        out.append((str(label), float(score), (x1, y1, x2, y2)))
    return out


def _iter_sampled_frames(video_path: str, sample_fps: float):
    """Yield (frame_index, timestamp_sec, frame) at a fixed sample_fps
    stride, regardless of the source video's native fps."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, round(native_fps / sample_fps))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride == 0:
            yield idx, idx / native_fps, frame
        idx += 1
    cap.release()


def auto_label_video(
    video_path: str,
    output_dir: str,
    cues: Dict[str, int],
    sample_fps: float = 1.0,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> Dict[str, int]:
    """Sample frames from one video at `sample_fps`, run Grounding DINO with
    `cues`, and write paired (image, YOLO label) files for every sampled
    frame — including frames with zero detections (empty label file), since
    those are real negative examples a reviewer needs to see too.

    Returns per-canonical-class instance counts for this video.
    """
    import cv2

    out = Path(output_dir)
    images_dir = out / "images"
    labels_dir = out / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(video_path).stem
    cue_list = list(cues.keys())
    counts: Dict[str, int] = {name: 0 for name in cues}
    frames_written = 0

    for idx, ts, frame in _iter_sampled_frames(video_path, sample_fps):
        h, w = frame.shape[:2]
        detections = _grounding_dino_predict(frame, cue_list, box_threshold, text_threshold)

        lines: List[str] = []
        for cue_text, conf, (x1, y1, x2, y2) in detections:
            cls = cues.get(cue_text)
            if cls is None:
                continue  # shouldn't happen (text_labels constrains the vocabulary), but stay defensive
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            counts[cue_text] += 1

        frame_name = f"{stem}_f{idx:06d}"
        cv2.imwrite(str(images_dir / f"{frame_name}.jpg"), frame)
        (labels_dir / f"{frame_name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        frames_written += 1

    print(f"  {stem}: {frames_written} frames sampled @ {sample_fps}fps", file=sys.stderr)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap YOLO labels with Grounding DINO (Phase A2)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="path to a single video to label")
    src.add_argument("--video-dir", help="directory of videos to label (all .mov/.mp4/.avi/.mkv)")
    ap.add_argument("--output", default="datasets/d22_autolabel", help="output directory")
    ap.add_argument("--fps", type=float, default=1.0, help="sampling rate (frames per second)")
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--limit-videos", type=int, default=None, help="process only the first N videos (smoke-testing)")
    ap.add_argument(
        "--cues", nargs="+", default=None,
        help="override the default cue list (must be keys already known to this "
             "script's DEFAULT_CUES map, or edit DEFAULT_CUES directly for new classes)",
    )
    args = ap.parse_args()

    cues = DEFAULT_CUES
    if args.cues:
        unknown = [c for c in args.cues if c not in DEFAULT_CUES]
        if unknown:
            print(f"ERROR: unknown cue(s) {unknown} — add them to DEFAULT_CUES in this script first "
                  "with an explicit canonical class id.", file=sys.stderr)
            sys.exit(2)
        cues = {c: DEFAULT_CUES[c] for c in args.cues}

    if args.video:
        videos = [args.video]
    else:
        exts = (".mov", ".mp4", ".avi", ".mkv")
        videos = sorted(
            str(p) for p in Path(args.video_dir).iterdir()
            if p.suffix.lower() in exts and not p.name.startswith("._")
        )
        if args.limit_videos:
            videos = videos[: args.limit_videos]

    if not videos:
        print("ERROR: no videos found.", file=sys.stderr)
        sys.exit(2)

    print(f"Cue -> canonical class mapping ({len(cues)} cues):")
    for cue, cid in cues.items():
        name = CANONICAL_CLASSES[cid] if 0 <= cid < len(CANONICAL_CLASSES) else "UNMAPPED"
        print(f"  '{cue}' -> {cid}: {name}")
    print(f"\nProcessing {len(videos)} video(s) at {args.fps} fps into {args.output}/\n")

    total_counts: Dict[str, int] = {name: 0 for name in cues}
    for video_path in videos:
        try:
            counts = auto_label_video(
                video_path, args.output, cues, args.fps, args.box_threshold, args.text_threshold
            )
        except Exception as exc:
            print(f"  ERROR processing {video_path}: {exc}", file=sys.stderr)
            continue
        for k, v in counts.items():
            total_counts[k] += v

    out = Path(args.output)
    review_marker = out / "_REVIEW_REQUIRED.txt"
    review_marker.write_text(
        "These labels were generated by Grounding DINO open-vocabulary detection "
        "and are UNREVIEWED. Do not use for training until a human has checked "
        "for false positives/negatives per NEW_PRODUCTION_ROADMAP.md Phase A2.\n",
        encoding="utf-8",
    )

    print("\nPer-cue instance counts across all processed videos:")
    for cue, n in total_counts.items():
        print(f"  {cue} ({CANONICAL_CLASSES[cues[cue]]}): {n}")
    print(f"\nWrote images/labels to {out}/  (paired 1:1, empty .txt = no detection that frame)")
    print(f"Review marker: {review_marker}")


if __name__ == "__main__":
    main()
