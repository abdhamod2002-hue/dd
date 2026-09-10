#!/usr/bin/env python3
"""Before/after evaluation of the adaptive self-tuning layer on REAL data.

Replays the per-tick person/object tracks recorded during the phase-1
production runs (phase1_runs/<video>/frames.jsonl — 19 real D:\\22 videos)
through:

  1. BASELINE : LitteringEventDetector with the production config (tier 0)
  2. ADAPTIVE : AdaptiveEventDetector (tier ladder + online learning,
                videos processed sequentially so learning accumulates)

No YOLO/pose re-inference is needed: the cached ticks are byte-identical to
what the production pipeline fed the detector.

Usage:
    .venv/Scripts/python.exe scripts/auto_tune_eval.py [--run-dir phase1_runs]

Outputs:
    adaptive_runs/before_after.json   machine-readable results
    adaptive_runs/before_after.md     human-readable comparison table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from adaptive_tuner import AdaptiveEventDetector, LearningStore  # noqa: E402
from littering_event_detector import (  # noqa: E402
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    LitteringEventDetector,
    load_event_config,
)


def load_ticks(frames_jsonl: Path):
    ticks = []
    with open(frames_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ticks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ticks


def rebuild_tick(rec: dict):
    """Mirror InferencePipeline._detector_person/_detector_bag faithfully:
    bag timestamps stay 0.0 (production never sets them — see audit note)."""
    persons = []
    for p in rec.get("persons") or []:
        kp_raw = p.get("keypoints") or {}
        kp = None
        if kp_raw:
            kp = DetectorKeypoints(
                left_wrist=tuple(kp_raw["left_wrist"]) if kp_raw.get("left_wrist") else None,
                right_wrist=tuple(kp_raw["right_wrist"]) if kp_raw.get("right_wrist") else None,
                left_shoulder=tuple(kp_raw["left_shoulder"]) if kp_raw.get("left_shoulder") else None,
                right_shoulder=tuple(kp_raw["right_shoulder"]) if kp_raw.get("right_shoulder") else None,
                torso_center=tuple(kp_raw["torso_center"]) if kp_raw.get("torso_center") else None,
            )
        persons.append(DetectorPerson(
            track_id=int(p["track_id"]),
            bbox=tuple(float(v) for v in p["bbox"]),
            confidence=float(p.get("confidence") or 0.0),
            keypoints=kp,
        ))
    bags = []
    for o in rec.get("objects") or []:
        source = str(o.get("source") or "yolo")
        bags.append(DetectorBag(
            track_id=int(o["track_id"]),
            bbox=tuple(float(v) for v in o["bbox"]),
            confidence=float(o.get("confidence") or 0.0),
            class_name=str(o.get("class_name") or "trash_bag"),
            source=source,
            yolo_confirmed=(source == "yolo"),
        ))
    return persons, bags


def run_baseline(ticks) -> dict:
    det = LitteringEventDetector(load_event_config())
    confirmed = 0
    for rec in ticks:
        persons, bags = rebuild_tick(rec)
        events = det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_number"]))
        confirmed += sum(1 for e in events if e.confirmed)
    for e in det.finalize():
        confirmed += 1 if e.confirmed else 0
    return det.summary() | {"confirmed_during_stream": confirmed}


def run_adaptive(ticks, store: LearningStore, video_name: str) -> dict:
    det = AdaptiveEventDetector(load_event_config(), store=store)
    det.learning_video = video_name
    confirmed = 0
    for rec in ticks:
        persons, bags = rebuild_tick(rec)
        events = det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_number"]))
        confirmed += sum(1 for e in events if e.confirmed)
    events = det.finalize()
    confirmed += sum(1 for e in events if e.confirmed)
    summary = det.summary()
    summary["confirmed_during_stream"] = confirmed
    tiers = {}
    for i, tier_det in enumerate(det._detectors):
        tiers[f"tier{i}"] = tier_det.summary().get("confirmed_violations", 0)
    summary["tiers_confirmed"] = tiers
    return summary


def _top_reasons(summary: dict, n: int = 3) -> str:
    counts = summary.get("rejection_reason_counts") or {}
    items = sorted(counts.items(), key=lambda kv: -kv[1])[:n]
    return ", ".join(f"{k}:{v}" for k, v in items) or "-"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(REPO / "phase1_runs"))
    args = ap.parse_args()
    run_dir = Path(args.run_dir)

    videos = sorted(p.parent.name for p in run_dir.glob("*/frames.jsonl"))
    print(f"replaying {len(videos)} videos from {run_dir}")

    store = LearningStore(path=str(REPO / "adaptive_runs" / "learning.json"))
    # Evaluation must start from a CLEAN learning state so the before/after
    # table reflects exactly this run's sequential learning (the store
    # accumulates across runs otherwise).
    store.reset()
    results = []
    for name in videos:
        ticks = load_ticks(run_dir / name / "frames.jsonl")
        if not ticks:
            print(f"  {name}: NO TICKS, skipped")
            continue
        t0 = time.time()
        base = run_baseline(ticks)
        adap = run_adaptive(ticks, store, name)
        row = {
            "video": name,
            "ticks": len(ticks),
            "baseline_confirmed": base.get("confirmed_during_stream", 0),
            "adaptive_confirmed": adap.get("confirmed_during_stream", 0),
            "tiers_confirmed": adap.get("tiers_confirmed", {}),
            "baseline_top_reasons": _top_reasons(base),
            "adaptive_top_reasons": _top_reasons(adap),
            "sec": round(time.time() - t0, 2),
        }
        results.append(row)
        print(f"  {name}: baseline={row['baseline_confirmed']} "
              f"adaptive={row['adaptive_confirmed']} tiers={row['tiers_confirmed']} "
              f"({row['sec']}s)")

    out_dir = REPO / "adaptive_runs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "before_after.json").write_text(
        json.dumps({"results": results, "learning": store._doc},
                   indent=2, ensure_ascii=False),
        encoding="utf-8")

    b_total = sum(r["baseline_confirmed"] for r in results)
    a_total = sum(r["adaptive_confirmed"] for r in results)
    b_vids = sum(1 for r in results if r["baseline_confirmed"] > 0)
    a_vids = sum(1 for r in results if r["adaptive_confirmed"] > 0)
    lines = [
        "# Adaptive self-tuning — before/after on real D:\\22 videos",
        "",
        "| Metric | Before (production config) | After (adaptive tiers + learning) |",
        "|---|---|---|",
        f"| Confirmed events | {b_total} | {a_total} |",
        f"| Videos with >=1 confirmed event | {b_vids}/{len(results)} | {a_vids}/{len(results)} |",
        "",
        "| Video | ticks | before | after | tiers (0/1/2) | baseline top reasons |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        t = r["tiers_confirmed"]
        lines.append(
            f"| {r['video']} | {r['ticks']} | {r['baseline_confirmed']} | "
            f"{r['adaptive_confirmed']} | {t.get('tier0', 0)}/{t.get('tier1', 0)}/{t.get('tier2', 0)} | "
            f"{r['baseline_top_reasons']} |"
        )
    (out_dir / "before_after.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'before_after.md'}")
    print(f"TOTAL: before={b_total} ({b_vids} videos)  after={a_total} ({a_vids} videos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
