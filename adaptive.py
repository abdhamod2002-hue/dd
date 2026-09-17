"""
Active-learning data collection for the littering pipeline.

When the temporal FSM establishes a CARRY or confirms a VIOLATION, the
person/object crops of that exact frame are dumped under
``datasets/active_learning/<tag>/`` with normalized YOLO-format labels in
``labels.jsonl``. These are the hard positives a future trash-bag detector
needs most — real deployment footage, zero manual labeling.

Collection only; training is a separate explicit step. Threshold learning
lives in adaptive_tuner.LearningStore (learning/learning.json).

No video-specific logic lives here (spec: no filename/frame hacks).
"""

from __future__ import annotations

import json
import os
import threading
from typing import Optional, Tuple

_LOCK = threading.Lock()

ACTIVE_LEARNING_ROOT = os.path.join("datasets", "active_learning")


def active_learning_dir(tag: str) -> str:
    return os.path.join(ACTIVE_LEARNING_ROOT, str(tag).replace(":", "_").replace("/", "_"))


def dump_pair_crops(
    tag: str,
    frame,
    person_bbox: Optional[Tuple[float, float, float, float]],
    bag_bbox: Optional[Tuple[float, float, float, float]],
    label: str,
    frame_index: int,
    person_track_id: int,
    bag_track_id: int,
    max_images: int = 40,
) -> Optional[str]:
    """Save person/object crops + labels for future training.

    Called on carry establishment and on confirmation. Cheap (two imwrites);
    capped per video tag. Returns the written labels path or None.
    """
    if frame is None or person_bbox is None:
        return None
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    out_dir = active_learning_dir(tag)
    try:
        os.makedirs(out_dir, exist_ok=True)
        if len([n for n in os.listdir(out_dir) if n.endswith(".jpg")]) >= max_images:
            return None
        h, w = frame.shape[:2]

        def _crop(bbox, pad_ratio=0.08):
            x1, y1, x2, y2 = bbox
            px = (x2 - x1) * pad_ratio
            py = (y2 - y1) * pad_ratio
            xa, ya = max(0, int(x1 - px)), max(0, int(y1 - py))
            xb, yb = min(w, int(x2 + px)), min(h, int(y2 + py))
            if xb - xa < 8 or yb - ya < 8:
                return None
            return frame[ya:yb, xa:xb], (xa, ya, xb, yb)

        base = f"f{frame_index:07d}_p{person_track_id}_b{bag_track_id}_{label}"
        written = []
        person_crop = _crop(person_bbox)
        if person_crop is not None:
            crop, box = person_crop
            path = os.path.join(out_dir, base + "_person.jpg")
            cv2.imwrite(path, crop)
            written.append((path, "person", box))
        if bag_bbox is not None:
            bag_crop = _crop(bag_bbox, pad_ratio=0.15)
            if bag_crop is not None:
                crop, box = bag_crop
                path = os.path.join(out_dir, base + "_waste.jpg")
                cv2.imwrite(path, crop)
                written.append((path, "waste", box))
        if not written:
            return None
        labels_path = os.path.join(out_dir, "labels.jsonl")
        with _LOCK:
            with open(labels_path, "a", encoding="utf-8") as f:
                for path, cls, (xa, ya, xb, yb) in written:
                    rec = {
                        "image": os.path.basename(path),
                        "class": cls,
                        "source_tag": str(tag),
                        "frame": int(frame_index),
                        "person_track_id": int(person_track_id),
                        "bag_track_id": int(bag_track_id),
                        "bbox_xyxy": [xa, ya, xb, yb],
                        "bbox_xywh_norm": [
                            round(((xa + xb) / 2.0) / w, 6),
                            round(((ya + yb) / 2.0) / h, 6),
                            round((xb - xa) / w, 6),
                            round((yb - ya) / h, 6),
                        ],
                        "label": label,
                    }
                    f.write(json.dumps(rec) + "\n")
        return labels_path
    except Exception:
        return None
