"""Build a YOLO dataset mixing new verdicts with experience-replay samples."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from learning_offline import CLASS_NAMES, DATA_INCOMING, DATA_REPLAY, OFFLINE_RUNS


def _collect_samples(root: Path) -> List[Tuple[Path, Path, dict]]:
    """Return (image, labels.txt, verdict) triples under a root."""
    out: List[Tuple[Path, Path, dict]] = []
    if not root.is_dir():
        return out
    for verdict_path in root.glob("**/verdict.json"):
        data = json.loads(verdict_path.read_text(encoding="utf-8"))
        folder = verdict_path.parent
        img_name = data.get("train_image")
        img = folder / img_name if img_name else None
        if img is None or not img.is_file():
            for cand in ("crop_waste.jpg", "waste.jpg", "object.jpg"):
                if (folder / cand).is_file():
                    img = folder / cand
                    break
        labels = folder / "labels.txt"
        if img is None or not img.is_file() or not labels.is_file():
            continue
        out.append((img, labels, data))
    return out


def build_mixed_dataset(
    *,
    run_id: str,
    new_ratio: float = 0.5,
    seed: int = 0,
    incoming_root: Optional[Path] = None,
    replay_root: Optional[Path] = None,
    max_samples: int = 400,
) -> Path:
    """Create ``learning_offline/runs/<run_id>/dataset`` with 50/50 mix.

    Experience replay (Section 7d): each training batch composition targets
    ``new_ratio`` new operator-labelled samples and ``1-new_ratio`` replay
    samples from ``data/replay`` (historical confirmed crops). Ultralytics +
    continual-learning practice: mix old+new to mitigate catastrophic forgetting.
    """
    rng = random.Random(int(seed))
    incoming = _collect_samples(incoming_root or DATA_INCOMING)
    replay = _collect_samples(replay_root or DATA_REPLAY)

    if not incoming and not replay:
        raise RuntimeError(
            "No labelled samples found under data/incoming or data/replay. "
            "Submit operator verdicts first (POST /api/events/{id}/verdict)."
        )

    n_target = min(max_samples, max(len(incoming) + len(replay), 1))
    n_new = int(round(n_target * float(new_ratio)))
    n_old = n_target - n_new

    picked: List[Tuple[Path, Path, dict]] = []
    if incoming:
        picked.extend(rng.sample(incoming, k=min(n_new, len(incoming))))
        # If not enough new, top up from remaining incoming.
        remain = [s for s in incoming if s not in picked]
        while len([p for p in picked if p in incoming]) < n_new and remain:
            picked.append(remain.pop())
    if replay:
        need_old = max(0, n_old - max(0, len(picked) - min(n_new, len(incoming) or 0)))
        # Simpler: take up to n_old from replay
        old_pick = rng.sample(replay, k=min(n_old, len(replay)))
        picked.extend(old_pick)
    elif incoming:
        # No replay yet — use leftover incoming so training can still run.
        remain = [s for s in incoming if s not in picked]
        picked.extend(remain[: max(0, n_target - len(picked))])

    # Deduplicate while preserving order
    seen = set()
    unique: List[Tuple[Path, Path, dict]] = []
    for item in picked:
        key = str(item[0].resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)

    if len(unique) < 2:
        raise RuntimeError("Need at least 2 samples to build train/val split")

    rng.shuffle(unique)
    split = max(1, int(len(unique) * 0.8))
    train_items = unique[:split]
    val_items = unique[split:] or unique[-1:]

    ds_root = OFFLINE_RUNS / run_id / "dataset"
    for split_name, items in (("train", train_items), ("val", val_items)):
        img_dir = ds_root / "images" / split_name
        lbl_dir = ds_root / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i, (img, labels, _meta) in enumerate(items):
            stem = f"{split_name}_{i:04d}"
            shutil.copy2(img, img_dir / f"{stem}{img.suffix.lower() or '.jpg'}")
            shutil.copy2(labels, lbl_dir / f"{stem}.txt")

    names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    yaml_path = ds_root / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {ds_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(names)}",
                "names: [" + ", ".join(repr(n) for n in names) + "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_incoming_available": len(incoming),
        "n_replay_available": len(replay),
        "new_ratio": new_ratio,
        "seed": seed,
    }
    (OFFLINE_RUNS / run_id / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return yaml_path
