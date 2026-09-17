#!/usr/bin/env python3
"""Prove learning.json read-leak (P0-A / RC-1). No inference logic changes.

1. Snapshot learning.json hash + effective tier-0 overrides.
2. Build AdaptiveEventDetector(deterministic=True) three times — same overrides.
3. Temporarily mutate adjusted_thresholds; rebuild — overrides change (read leak).
4. Confirm freeze_learning_writes prevents finalize from rewriting the file.
5. Restore original learning.json.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LEARNING = ROOT / "learning" / "learning.json"
OUT = ROOT / "project_audit" / "_comprehensive_plan_learning_leak.json"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    from adaptive_tuner import AdaptiveEventDetector, LearningStore
    from littering_event_detector import load_event_config

    backup = LEARNING.read_bytes()
    h0 = _hash(LEARNING)
    data0 = json.loads(backup.decode("utf-8"))
    adj0 = dict(data0.get("adjusted_thresholds") or {})

    results: dict = {
        "learning_path": str(LEARNING),
        "sha256_before": h0,
        "videos_analyzed": data0.get("videos_analyzed"),
        "adjusted_thresholds_before": adj0,
        "runs": [],
    }

    base = load_event_config()
    store = LearningStore(str(LEARNING))

    for i in range(3):
        det = AdaptiveEventDetector(
            base,
            store,
            camera_id="leak-probe",
            deterministic=True,
            freeze_learning_writes=True,
        )
        cfg = det.config
        results["runs"].append(
            {
                "i": i,
                "min_carried_frames": int(cfg.min_carried_frames),
                "release_distance_ratio": float(cfg.release_distance_ratio),
                "departure_motion_ratio": float(cfg.departure_motion_ratio),
                "freeze_learning_writes": bool(det.freeze_learning_writes),
                "sha256_after_construct": _hash(LEARNING),
            }
        )
        # Deterministic finalize must not rewrite learning.json
        det.learning_video = "LEAK_PROBE_SHOULD_NOT_WRITE.MOV"
        det.finalize()
        results["runs"][-1]["sha256_after_finalize"] = _hash(LEARNING)

    # Mutate step_index on disk (simulates another process advancing learning).
    # learned_overrides() RECOMPUTES from step_index — editing adjusted_thresholds
    # alone is a no-op (see adaptive_tuner.LearningStore.learned_overrides).
    mutated = json.loads(LEARNING.read_text(encoding="utf-8"))
    steps_before = dict(mutated.get("step_index") or {})
    mutated.setdefault("step_index", {})
    # Drop carry/release learning to YAML defaults for this construction only.
    mutated["step_index"]["NOT_ENOUGH_CARRIED_FRAMES"] = 0
    mutated["step_index"]["NO_RELEASE_TRANSITION"] = 0
    LEARNING.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    h_mut = _hash(LEARNING)

    store2 = LearningStore(str(LEARNING))  # re-read from disk
    det2 = AdaptiveEventDetector(
        load_event_config(),
        store2,
        camera_id="leak-probe",
        deterministic=True,
        freeze_learning_writes=True,
    )
    cfg2 = det2.config
    yaml_cfg = load_event_config()
    results["after_manual_edit"] = {
        "sha256": h_mut,
        "step_index_before": steps_before,
        "step_index_edited": dict(mutated["step_index"]),
        "min_carried_frames": int(cfg2.min_carried_frames),
        "release_distance_ratio": float(cfg2.release_distance_ratio),
        "yaml_min_carried_frames": int(yaml_cfg.min_carried_frames),
        "yaml_release_distance_ratio": float(yaml_cfg.release_distance_ratio),
        "read_leak_proven": (
            int(cfg2.min_carried_frames) != int(results["runs"][0]["min_carried_frames"])
            or abs(
                float(cfg2.release_distance_ratio)
                - float(results["runs"][0]["release_distance_ratio"])
            )
            > 1e-9
        ),
        "note": (
            "Editing step_index changes tier-0 thresholds on the next "
            "AdaptiveEventDetector construction even with deterministic=True / "
            "freeze_learning_writes=True. adjusted_thresholds is a derived cache."
        ),
    }

    # Restore
    LEARNING.write_bytes(backup)
    results["sha256_after_restore"] = _hash(LEARNING)
    results["restored_ok"] = results["sha256_after_restore"] == h0
    results["writes_frozen_across_3x"] = all(
        r["sha256_after_construct"] == h0 and r["sha256_after_finalize"] == h0
        for r in results["runs"]
    )
    results["thresholds_stable_across_3x"] = (
        len({(r["min_carried_frames"], r["release_distance_ratio"]) for r in results["runs"]})
        == 1
    )

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
