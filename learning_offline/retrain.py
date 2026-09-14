#!/usr/bin/env python3
"""Offline self-learning retrain (Section 7).

Steps:
  1. Build 50/50 new+replay YOLO dataset from data/incoming + data/replay
  2. Fine-tune from current best.pt (or yolov8s.pt) with backbone freeze + low LR
  3. Write candidate weights under inference/detection/weights/candidates/

Does NOT promote to production — run ``python -m learning_offline.gate_cli`` /
``promote`` after frozen eval.

Usage:
  python -m learning_offline.retrain --epochs 20 --device cpu
  python -m learning_offline.retrain --dry-run   # build dataset only
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_offline import CANDIDATE_DIR, OFFLINE_RUNS, PRODUCTION_WEIGHTS  # noqa: E402
from learning_offline.dataset_builder import build_mixed_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Motared offline self-learning retrain")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--lr0", type=float, default=0.001, help="Low LR fine-tune (Section 7d)")
    ap.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Freeze first N layers (10 ≈ YOLOv8 backbone; Ultralytics docs)",
    )
    ap.add_argument("--new-ratio", type=float, default=0.5, help="Fraction of NEW samples (rest=replay)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", type=str, default="", help="Warm-start weights (default: best.pt or yolov8s.pt)")
    ap.add_argument("--dry-run", action="store_true", help="Build dataset only; skip ultralytics train")
    ap.add_argument("--run-id", type=str, default="")
    args = ap.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("retrain_%Y%m%dT%H%M%SZ")
    yaml_path = build_mixed_dataset(
        run_id=run_id,
        new_ratio=float(args.new_ratio),
        seed=int(args.seed),
    )
    print(f"[retrain] dataset → {yaml_path}")

    if args.dry_run:
        print("[retrain] dry-run: skipping YOLO train")
        return 0

    weights = Path(args.weights) if args.weights else PRODUCTION_WEIGHTS
    if not weights.is_file():
        weights = Path("yolov8s.pt")
        print(f"[retrain] production weights missing; warm-start {weights}")

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as exc:
        print(f"[retrain] ultralytics not available: {exc}", file=sys.stderr)
        return 2

    # Research note (Ultralytics train docs + continual-learning issues):
    # freeze=10 freezes backbone; low lr0 + 50/50 replay mitigates catastrophic forgetting.
    model = YOLO(str(weights))
    project = str(OFFLINE_RUNS / run_id)
    results = model.train(
        data=str(yaml_path),
        epochs=int(args.epochs),
        batch=int(args.batch),
        imgsz=int(args.img),
        device=args.device,
        project=project,
        name="yolo",
        exist_ok=True,
        freeze=int(args.freeze),
        lr0=float(args.lr0),
        optimizer="AdamW",
        seed=int(args.seed),
        deterministic=True,
        plots=False,
        verbose=True,
    )

    # Locate best.pt from the run
    best = Path(project) / "yolo" / "weights" / "best.pt"
    if not best.is_file():
        # ultralytics may return save_dir
        save_dir = Path(getattr(results, "save_dir", project))
        alt = save_dir / "weights" / "best.pt"
        if alt.is_file():
            best = alt
    if not best.is_file():
        print(f"[retrain] ERROR: trained weights not found under {project}", file=sys.stderr)
        return 1

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    candidate = CANDIDATE_DIR / f"{run_id}.pt"
    candidate.write_bytes(best.read_bytes())
    meta = {
        "run_id": run_id,
        "candidate": str(candidate),
        "source_weights": str(weights),
        "data_yaml": str(yaml_path),
        "freeze": args.freeze,
        "lr0": args.lr0,
        "new_ratio": args.new_ratio,
        "epochs": args.epochs,
    }
    (OFFLINE_RUNS / run_id / "retrain_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[retrain] candidate written → {candidate}")
    print("[retrain] NEXT: run frozen eval on candidate, then learning_offline.gate_cli")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
