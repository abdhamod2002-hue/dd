"""Write operator verdicts into dated ``data/incoming/...`` folders."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from learning_offline import CLASS_NAMES, DATA_INCOMING


Verdict = str  # "confirm" | "reject"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def event_dir(event_id: Union[int, str], day: Optional[str] = None) -> Path:
    return DATA_INCOMING / (day or _today()) / f"event_{event_id}"


def write_verdict(
    event_id: Union[int, str],
    verdict: Verdict,
    *,
    notes: str = "",
    crop_paths: Optional[Dict[str, Path]] = None,
    waste_bbox_norm: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    day: Optional[str] = None,
) -> Path:
    """Persist a human review outcome for offline retrain.

    ``verdict``:
      * ``confirm`` — true litter; YOLO positive label written if bbox/crop given
      * ``reject``  — hard negative; empty label file (background)

    Returns the event directory path.
    """
    v = str(verdict).strip().lower()
    if v not in ("confirm", "reject"):
        raise ValueError("verdict must be 'confirm' or 'reject'")

    out = event_dir(event_id, day=day)
    out.mkdir(parents=True, exist_ok=True)

    crop_paths = crop_paths or {}
    for key, src in crop_paths.items():
        src = Path(src)
        if src.is_file():
            dest = out / str(key)
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)

    # Prefer an explicit waste crop as the training image.
    train_img = None
    for name in ("crop_waste.jpg", "waste.jpg", "object.jpg"):
        cand = out / name
        if cand.is_file():
            train_img = cand
            break
    if train_img is None:
        for p in crop_paths.values():
            p = Path(p)
            if p.is_file() and "waste" in p.name.lower():
                dest = out / "crop_waste.jpg"
                if p.resolve() != dest.resolve():
                    shutil.copy2(p, dest)
                train_img = dest
                break

    labels_path = out / "labels.txt"
    if v == "confirm":
        # Full-crop positive when no image-space bbox is supplied.
        box = waste_bbox_norm or [0.5, 0.5, 0.95, 0.95]
        if len(box) != 4:
            raise ValueError("waste_bbox_norm must be [cx, cy, w, h] normalized")
        cx, cy, bw, bh = [float(x) for x in box]
        labels_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")
    else:
        # Hard negative: image present, no positive boxes.
        labels_path.write_text("", encoding="utf-8")

    payload = {
        "event_id": event_id,
        "verdict": v,
        "notes": notes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "class_names": CLASS_NAMES,
        "train_image": train_img.name if train_img else None,
        "labels_file": labels_path.name,
        "hard_negative": v == "reject",
        "metadata": metadata or {},
    }
    (out / "verdict.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return out


def list_incoming_verdicts(root: Optional[Path] = None) -> List[Path]:
    root = root or DATA_INCOMING
    if not root.is_dir():
        return []
    return sorted(root.glob("*/event_*/verdict.json"))
