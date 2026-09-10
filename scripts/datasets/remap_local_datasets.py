"""
Remap already-prepared, YOLO-formatted local datasets into the canonical
8-class schema (unified_class_map.py) and merge them into one training-ready
tree at datasets/unified_v1/.

NEW_PRODUCTION_ROADMAP.md Phase A1. This closes the gap the audit found:
three disjoint, narrow class vocabularies exist in this repo (garbage_bag_v2.pt:
1 class, best.pt: 5 unrelated classes, taco_yolo: 4 classes) and a canonical
8-class schema was designed (unified_class_map.py, ANNOTATION_PLAN.md) but
never actually executed against real data. This script is that execution,
for the two local datasets that are already YOLO-formatted and don't need
re-annotation:

  - datasets/roboflow_colored_bags/Trash.v1i.yolov8/  (1 class: "Garbage Bag")
  - datasets/taco_yolo/                                (4 classes: bag, bottle, cup, paper)

READ-ONLY on the source datasets: every image/label is COPIED, never moved
or edited in place. No model weights are touched. No training is run.

IMPORTANT correctness note: this deliberately does NOT use unified_class_map's
`canonical_id()` fuzzy substring matcher for these two datasets, even though
`canonical_id("Garbage Bag", source="auto")` happens to return the right
answer (4, via an exact "Garbage bag" entry in TACO_MAP that's tried before
GARBAGE_MAP). Calling it with the seemingly-obvious `source="garbage"`
instead silently returns 7 (other_litter) — verified live:
    >>> canonical_id("Garbage Bag", source="garbage")
    7   # WRONG — would mislabel all 847 roboflow_colored_bags images
    >>> canonical_id("Garbage Bag", source="auto")
    4   # right, but only by accident of which map is tried first
Getting the right answer for "auto" depends on TACO_MAP happening to list
this exact phrase ahead of GARBAGE_MAP's "garbage"/"bag" substring entries —
fragile, and a future edit to either map's ordering or contents could flip
it silently. This script uses its own explicit, exact-name map per dataset
instead, so the mapping doesn't depend on cross-map iteration order at all.

Usage:
    python scripts/datasets/remap_local_datasets.py [--output datasets/unified_v1]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.datasets.unified_class_map import CANONICAL_CLASSES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
IMG_EXTS = (".jpg", ".jpeg", ".png")

# Explicit, exact-name maps — see module docstring for why NOT canonical_id().
ROBOFLOW_COLORED_BAGS_MAP: Dict[str, int] = {"garbage bag": 4}  # -> bag
TACO_YOLO_MAP: Dict[str, int] = {"bag": 4, "bottle": 1, "cup": 2, "paper": 5}


def _remap_split(
    src_images: Path,
    src_labels: Path,
    dst_images: Path,
    dst_labels: Path,
    old_class_names: List[str],
    class_map: Dict[str, int],
    tag: str,
) -> Tuple[int, int, Counter]:
    """Copy one split's images + re-indexed labels. Returns
    (images_copied, instances_written, per-canonical-class instance counter).
    Images with zero mappable instances are still copied (background/negative
    frames have real training value) but get an empty label file, matching
    standard YOLO convention for "no object of interest here"."""
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    old_id_to_canonical: Dict[int, int] = {}
    for old_id, old_name in enumerate(old_class_names):
        canon = class_map.get(old_name.strip().lower())
        if canon is not None:
            old_id_to_canonical[old_id] = canon

    images_copied = 0
    instances_written = 0
    counts: Counter = Counter()

    if not src_images.is_dir():
        return 0, 0, counts

    for img_path in sorted(src_images.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = src_labels / f"{img_path.stem}.txt"
        new_name = f"{tag}_{img_path.name}"
        shutil.copy2(img_path, dst_images / new_name)
        images_copied += 1

        out_lines: List[str] = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                canon = old_id_to_canonical.get(old_id)
                if canon is None:
                    continue  # class not in our canonical map -> drop this box only
                out_lines.append(" ".join([str(canon), *parts[1:]]))
                counts[canon] += 1
                instances_written += 1
        (dst_labels / f"{tag}_{img_path.stem}.txt").write_text(
            "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
        )

    return images_copied, instances_written, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="datasets/unified_v1", help="merged output directory")
    args = ap.parse_args()

    out_root = REPO_ROOT / args.output
    total_counts: Counter = Counter()
    total_images = 0
    total_instances = 0

    jobs = [
        # (tag, splits: {dst_split: (src_images, src_labels)}, old_class_names, class_map)
        (
            "rfbag",
            {
                "train": (
                    REPO_ROOT / "datasets/roboflow_colored_bags/Trash.v1i.yolov8/train/images",
                    REPO_ROOT / "datasets/roboflow_colored_bags/Trash.v1i.yolov8/train/labels",
                ),
                "val": (
                    REPO_ROOT / "datasets/roboflow_colored_bags/Trash.v1i.yolov8/valid/images",
                    REPO_ROOT / "datasets/roboflow_colored_bags/Trash.v1i.yolov8/valid/labels",
                ),
                "test": (
                    REPO_ROOT / "datasets/roboflow_colored_bags/Trash.v1i.yolov8/test/images",
                    REPO_ROOT / "datasets/roboflow_colored_bags/Trash.v1i.yolov8/test/labels",
                ),
            },
            ["garbage bag"],
            ROBOFLOW_COLORED_BAGS_MAP,
        ),
        (
            "taco",
            {
                "train": (REPO_ROOT / "datasets/taco_yolo/images/train", REPO_ROOT / "datasets/taco_yolo/labels/train"),
                "val": (REPO_ROOT / "datasets/taco_yolo/images/val", REPO_ROOT / "datasets/taco_yolo/labels/val"),
            },
            ["bag", "bottle", "cup", "paper"],
            TACO_YOLO_MAP,
        ),
    ]

    print(f"Merging into: {out_root}\n")
    per_dataset_report: List[str] = []
    for tag, splits, old_names, class_map in jobs:
        ds_images = 0
        ds_instances = 0
        ds_counts: Counter = Counter()
        for split, (src_images, src_labels) in splits.items():
            imgs, inst, counts = _remap_split(
                src_images, src_labels,
                out_root / "images" / split, out_root / "labels" / split,
                old_names, class_map, tag,
            )
            ds_images += imgs
            ds_instances += inst
            ds_counts.update(counts)
            total_counts.update(counts)
        total_images += ds_images
        total_instances += ds_instances
        per_dataset_report.append(
            f"  {tag}: {ds_images} images, {ds_instances} instances -> "
            + ", ".join(f"{CANONICAL_CLASSES[c]}={n}" for c, n in sorted(ds_counts.items())) if ds_counts
            else f"  {tag}: {ds_images} images, 0 instances"
        )

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        "# Generated by scripts/datasets/remap_local_datasets.py (Phase A1)\n"
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"nc: {len(CANONICAL_CLASSES)}\n"
        "names:\n" + "\n".join(f"  {i}: {name}" for i, name in enumerate(CANONICAL_CLASSES)) + "\n",
        encoding="utf-8",
    )

    print("Per-dataset breakdown:")
    print("\n".join(per_dataset_report))
    print(f"\nTOTAL: {total_images} images, {total_instances} instances across {len(jobs)} source datasets")
    print("\nMerged class distribution (canonical 8-class schema):")
    for i, name in enumerate(CANONICAL_CLASSES):
        print(f"  {i}: {name:<16} {total_counts.get(i, 0)}")
    print(f"\nWrote {data_yaml}")


if __name__ == "__main__":
    main()
