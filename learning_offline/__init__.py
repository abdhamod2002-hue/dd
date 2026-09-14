"""Section 7 — offline self-learning (experience replay + promotion gate).

This package is intentionally SEPARATE from ``adaptive_tuner.LearningStore``
(online FSM threshold tweaks). Section 7 only updates YOLO weights offline,
versioned, and only after a frozen-holdout promotion gate passes.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_INCOMING = ROOT / "data" / "incoming"
DATA_REPLAY = ROOT / "data" / "replay"
OFFLINE_RUNS = ROOT / "learning_offline" / "runs"
CANDIDATE_DIR = ROOT / "inference" / "detection" / "weights" / "candidates"
WEIGHTS_DIR = ROOT / "inference" / "detection" / "weights"
PRODUCTION_WEIGHTS = WEIGHTS_DIR / "best.pt"
FROZEN_SET = ROOT / "evaluation" / "frozen_test_set.v1.json"

# Single litter class for crop-level fine-tuning (person stays on COCO/person weights).
CLASS_NAMES = {0: "trash_bag"}
