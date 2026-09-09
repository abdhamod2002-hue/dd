"""
Litter Detection Trainer — trains YOLO on the unified 8-class schema.

Combines:
- TACO (real-world litter in context)
- Garbage Object Detection
- WADE-AI / Roboflow litter
- Custom CCTV frames (crumpled_tissue class from your camera)

The trainer is a thin wrapper around ultralytics YOLO training.
It expects labels in YOLO format (from unified_class_map + prepare_datasets + auto_label_grounding_dino).

TRANSFER LEARNING (default):
    The DEFAULT starting weights are the official COCO-pretrained
    ``yolov8s.pt`` (or ``yolov8n.pt`` for speed). We do NOT start from the
    existing custom ``best.pt``: per the generalization audit it is BROKEN on
    real media (top score 0.024 at conf=0.001, empty at conf>=0.05 on
    person_bottle2.jpg / IMG_5113 / IMG_5114; see diag_bestpt2.txt). Starting a
    new model from it would transfer that failure. yolov8s.pt is globally
    tested and already contains a real, working ``bottle`` class, making it a
    sound warm start for litter detection. TACO provides the litter-domain
    labels (mapped to our schema) for fine-tuning.

    If you instead want to reuse a working litter model as a warm start, pass
    ``--weights <path>`` to any verified checkpoint — but NEVER best.pt until it
    is independently proven to detect on real footage.

Usage:
    python scripts/train_yolo.py \\
        --data datasets/taco_yolo/ \\
        --weights yolov8s.pt \\
        --epochs 50 \\
        --batch 8 \\
        --img 640 \\
        --device cpu \\
        --name taco_transfer_v1

This does NOT overwrite best.pt. The new weights land in
runs/detect/<name>/weights/best.pt and should be copied to a clearly named
file (e.g. inference/detection/weights/taco_transfer_v1.pt).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def train_yolo(
    data_yaml: str,
    weights: str = "yolov8s.pt",
    epochs: int = 50,
    batch: int = 8,
    img: int = 640,
    device: str = "cpu",
    name: str = "taco_transfer_v1",
    project: str = "D:/HO/runs",
) -> bool:
    """Train YOLO, transfer-learning from an official COCO checkpoint.

    Args:
        data_yaml: path to a data.yaml in ultralytics format (train/val paths + nc + names)
        weights: starting weights. DEFAULT ``yolov8s.pt`` (COCO-pretrained,
            globally tested, contains a real ``bottle`` class). Passing the old
            custom ``best.pt`` is intentionally NOT the default because it is
            broken on real media (see diag_bestpt2.txt) and would transfer that
            failure. Use yolov8n.pt for speed.
        epochs: training epochs (20-50 is enough for fine-tuning on TACO)
        batch: batch size (2-8 for CPU, 16+ for GPU)
        img: input image size (640 default; 416/512 for faster CPU runs)
        device: 'cpu' or '0' (GPU index)
        name: run name for the trained weights (lands in runs/detect/<name>/...)

    Returns:
        True if training completed and weights were saved.

    The actual training is delegated to ultralytics:
        yolo train model=<weights> data=data_yaml epochs=... img=... name=<name>
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    print(f"[INFO] Starting YOLO training: {epochs} epochs, batch={batch}, img={img}, device={device}")
    print(f"  Weights: {weights}  (NOTE: NOT best.pt — best.pt is broken on real media)")
    print(f"  Data: {data_yaml}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=img,
        device=device,
        name=name,
        project=project,
    )
    # Ultralytics saves under <project>/<name>/weights/... when `project` is
    # explicitly passed, and under <project>/detect/<name>/... only when project
    # defaults to "runs/detect". Check both so the completion check is correct.
    candidates = [
        f"{project}/{name}/weights/best.pt",
        f"{project}/detect/{name}/weights/best.pt",
    ]
    weights_path = next((c for c in candidates if os.path.exists(c)), None)
    if weights_path:
        print(f"[OK] Training complete. Weights saved to: {weights_path}")
        print(f"  To use in the pipeline, point YoloDetector(litter_weights='{weights_path}')")
        return True
    print(f"[ERROR] Training did not produce weights at any of: {candidates}")
    return False


def make_data_yaml(data_dir: str, output: str = "datasets/data.yaml") -> str:
    """Generate the ultralytics data.yaml with the unified 8-class names."""
    from scripts.datasets.unified_class_map import CANONICAL_CLASSES

    data_dir = Path(data_dir)
    yaml_path = Path(output)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)

    # ultralytics data.yaml format
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CANONICAL_CLASSES))
    content = f"""# Unified 8-class litter detection dataset
path: {data_dir.resolve()}
train: {data_dir}/train/images
val: {data_dir}/val/images
test: {data_dir}/test/images

nc: {len(CANONICAL_CLASSES)}
names:
{names_block}
"""
    yaml_path.write_text(content)
    return str(yaml_path)


def main():
    ap = argparse.ArgumentParser(description="Train YOLO11s on unified litter dataset")
    ap.add_argument("--data", default="datasets", help="dataset root directory")
    ap.add_argument("--weights", default="yolov8s.pt",
                    help="starting weights (default: yolov8s.pt; do NOT use broken best.pt)")
    ap.add_argument("--epochs", type=int, default=50, help="training epochs")
    ap.add_argument("--batch", type=int, default=8, help="batch size")
    ap.add_argument("--img", type=int, default=640, help="input image size")
    ap.add_argument("--device", default="cpu", help="cpu or GPU index")
    ap.add_argument("--name", default="litter_yolo11s_v1", help="run name")
    ap.add_argument("--make-yaml", action="store_true", help="generate data.yaml before training")
    args = ap.parse_args()

    data_yaml = args.data + "/data.yaml"
    if args.make_yaml:
        p = make_data_yaml(args.data, data_yaml)
        print(f"[OK] data.yaml written to {p}")

    ok = train_yolo(
        data_yaml=data_yaml,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        img=args.img,
        device=args.device,
        name=args.name,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
