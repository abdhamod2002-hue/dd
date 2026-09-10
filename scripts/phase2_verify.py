"""Phase 2 — final regression gate.

Replays all 5 videos using the config EXACTLY AS LOADED FROM DISK (no
overrides) and compares against the known production numbers recorded before
Phase 2 began. Any difference here means the Phase-2 code changed production
behaviour while the flag is off, which must not happen.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from littering_event_detector import (  # noqa: E402
    LitteringEventDetector, load_event_config,
)
from scripts.phase1_sweep import CACHE_DIR, load_ticks  # noqa: E402
from scripts.phase1_probe_runs import build_frame  # noqa: E402

# Recorded from the real production run before Phase 2 (PHASE1_REPORT.md).
BASELINE = {
    "IMG_5115": ("REJECTED", "NO_RELEASE_TRANSITION", 0.2572),
    "IMG_5117": ("CONFIRMED", "VIOLATION_CONFIRMED", 0.8581),
    "IMG_5118": ("CONFIRMED", "VIOLATION_CONFIRMED", 0.8285),
    "IMG_5119": ("CONFIRMED", "VIOLATION_CONFIRMED", 0.8280),
    "IMG_5120": ("CONFIRMED", "VIOLATION_CONFIRMED", 0.8280),
}


def main() -> int:
    cfg = load_event_config()
    print(f"config as loaded from disk:")
    print(f"  pose_association_enabled = {cfg.pose_association_enabled}")
    print(f"  pose_score_blend         = {cfg.pose_score_blend}")
    print(f"  stationary_max_pixel_step= {cfg.stationary_max_pixel_step}")
    print()

    ok = True
    print(f"{'video':<10} {'expected':<34} {'actual':<34} {'':<6}")
    print("-" * 88)
    for stem in sorted(p.stem for p in CACHE_DIR.glob("*.jsonl")):
        det = LitteringEventDetector(cfg)
        det.reset()
        for rec in load_ticks(stem):
            persons, bags = build_frame(rec)
            det.update(persons, bags, float(rec["timestamp"]),
                       int(rec["frame_index"]))
        det.finalize()

        events = list(det.confirmed_events) + list(det.rejected_events)
        if not events:
            actual = ("NONE", "-", 0.0)
        else:
            ev = events[0]
            actual = ("CONFIRMED" if ev.confirmed else "REJECTED",
                      str(ev.reason), round(float(ev.confidence), 4))

        exp = BASELINE[stem]
        match = (exp[0] == actual[0] and exp[1] == actual[1]
                 and abs(exp[2] - actual[2]) < 5e-4)
        ok = ok and match
        print(f"{stem:<10} {exp[0]+'/'+exp[1]+'/'+format(exp[2],'.4f'):<34} "
              f"{actual[0]+'/'+actual[1]+'/'+format(actual[2],'.4f'):<34} "
              f"{'MATCH' if match else 'DIFF':<6}")

    print("-" * 88)
    print("RESULT:", "PRODUCTION UNCHANGED — safe" if ok
                     else "REGRESSION — do not ship")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
