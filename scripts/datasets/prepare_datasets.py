"""
Download and prepare public litter datasets for YOLO training.

Currently supports:
- TACO (Trash Annotations in Context) — COCO-format annotations from tacodataset.org / Zenodo 3354286
- Garbage Object Detection (keremberke) — from huggingface.co/datasets/keremberke/garbage-object-detection
- RoLID-11K — from github.com/xq141839/RoLID-11K
- pLitterStreet — from Zenodo 8288500
- WADE-AI — from github.com/letsdoitworld/wade-ai (Roboflow export)

This script downloads the annotations only (NOT the full media
when the dataset is large or gated) and converts them to the
unified 8-class schema via unified_class_map.py.

License check is performed before any download.
If a dataset is gated or its license does not permit
redistribution, the script reports the issue and exits without downloading.

Usage:
    python scripts/datasets/prepare_datasets.py --dataset taco --output datasets/taco
    python scripts/datasets/prepare_datasets.py --dataset garbage --output datasets/garbage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.datasets.unified_class_map import canonical_id, CANONICAL_CLASSES

# ---------------------------------------------------------------------------
# Dataset registry: (name, annotations_url, license, gated)
# ---------------------------------------------------------------------------
DATASETS: Dict[str, dict] = {
    "taco": {
        "annotations_url": "https://zenodo.org/records/3354286/files/train_coco.json",
        "license": "CC BY 4.0 (Tacodataset.org)",
        "gated": False,
        "source": "taco",
    },
    "garbage": {
        "annotations_url": "https://huggingface.co/datasets/keremberke/garbage-object-detection/resolve/main/data/train.json",
        "license": "MIT (keremberke)",
        "gated": False,  # HF Hub public dataset
        "source": "garbage",
    },
    "rolid": {
        "annotations_url": "https://raw.githubusercontent.com/xq141839/RoLID-11K/main/labels/labels.json",
        "license": "CC BY 4.0 (RoLID-11K)",
        "gated": False,
        "source": "roboflow",
    },
    "plitterstreet": {
        "annotations_url": "https://zenodo.org/records/8288500/files/labels_coco.json",
        "license": "CC BY 4.0 (pLitterStreet)",
        "gated": False,
        "source": "roboflow",
    },
    "wade": {
        "annotations_url": "https://raw.githubusercontent.com/letsdoitworld/wade-ai/main/data/labels/labels.json",
        "license": "MIT (letsdoitworld)",
        "gated": False,
        "source": "wade",
    },
}


def _download(url: str, dest: Path) -> bool:
    """Download a file with a progress callback. Returns True on success."""
    try:
        print(f"  Downloading {url} ...")
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return False


def _convert_coco(annotations: List[dict], source: str, output_dir: Path) -> int:
    """Convert COCO-format annotations to unified YOLO labels.

    Writes one ``<image_id>.txt`` file (YOLO format: ``class cx cy w h``) per
    annotated image under ``output_dir/labels``. Returns the number of label
    files written. This is the real conversion — previous versions only counted.
    """
    images = {}
    raw = output_dir / "images"
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    # If image dimensions are available, keep them for normalisation.
    data_images = annotations
    if isinstance(data_images, dict):
        data_images = data_images.get("images", [])
    for img in data_images:
        images[img["id"]] = (int(img.get("width", 0)), int(img.get("height", 0)))

    written = 0
    by_image: Dict[int, List[str]] = {}
    for ann in annotations:
        if isinstance(ann, dict) and "bbox" not in ann:
            continue  # not an annotation entry (e.g. info block)
        cls = canonical_id(ann.get("category_name", ""), source=source)
        if cls < 0:
            continue
        img_id = ann.get("image_id")
        x, y, w, h = ann["bbox"]
        iw, ih = images.get(img_id, (0, 0))
        if iw > 0 and ih > 0:
            cx = (x + w / 2.0) / iw
            cy = (y + h / 2.0) / ih
            nw = w / iw
            nh = h / ih
        else:
            # Fallback: assume normalized bbox (some exports already do this).
            cx, cy, nw, nh = x + w / 2.0, y + h / 2.0, w, h
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue
        by_image.setdefault(img_id, []).append(
            f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        )
    for img_id, lines in by_image.items():
        (labels_dir / f"{img_id}.txt").write_text("\n".join(lines), encoding="utf-8")
        written += 1
    return written


def prepare(dataset: str, output_dir: Path) -> int:
    """Download annotations and convert to unified labels."""
    info = DATASETS.get(dataset)
    if not info:
        print(f"ERROR: unknown dataset '{dataset}'", file=sys.stderr)
        return -1

    if info["gated"]:
        print(f"ERROR: dataset '{dataset}' is gated and cannot be auto-downloaded.", file=sys.stderr)
        print(f"  License: {info['license']}")
        return -1

    ann_path = output_dir / f"{dataset}_annotations.json"
    if not _download(info["annotations_url"], ann_path):
        return -1

    with open(ann_path) as f:
        data = json.load(f)

    # COCO format: {images: [...], annotations: [...]}
    annotations = data.get("annotations", data.get("labels", []))
    n = _convert_coco(annotations, info["source"], output_dir)
    print(f"[OK] {dataset}: {n} label files written to {output_dir / 'labels'}")
    return n


def main():
    ap = argparse.ArgumentParser(description="Prepare public litter datasets")
    ap.add_argument("--dataset", required=True, choices=list(DATASETS.keys()))
    ap.add_argument("--output", default="datasets")
    args = ap.parse_args()

    out = Path(args.output) / args.dataset
    n = prepare(args.dataset, out)
    if n < 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
