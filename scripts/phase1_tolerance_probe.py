"""Phase-1 diagnosis: is the research's longer persistence reachable AT ALL?

The consecutive-run counter resets to zero on a single non-stationary tick.
That makes long persistence unreachable even though the bags really are sitting
still for seconds at a time -- one jitter tick wipes the count.

This probe reconstructs the per-tick `smooth_stationary` boolean sequence (the
signal that actually increments the counter) and asks: how long a run could we
sustain if the counter tolerated k consecutive dropouts instead of resetting?

MEASUREMENT ONLY. No production code is modified and nothing is applied.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from littering_event_detector import (  # noqa: E402
    EventDetectorConfig, LitteringEventDetector, load_event_config,
)
from scripts.phase1_sweep import ANALYSIS_FPS, CACHE_DIR, load_ticks  # noqa: E402
from scripts.phase1_probe_runs import build_frame  # noqa: E402


def stationary_sequence(stem: str):
    """Recover the per-tick smooth_stationary booleans after release."""
    cfg = load_event_config()
    data = cfg.to_dict()
    data["min_stationary_frames"] = 10 ** 6
    data["min_abandonment_frames"] = 10 ** 6
    cfg = EventDetectorConfig.from_dict(data)
    det = LitteringEventDetector(cfg)
    det.reset()

    seq = []
    prev = {}
    ever_released = set()
    for rec in load_ticks(stem):
        persons, bags = build_frame(rec)
        det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_index"]))
        for key, mem in det._pairs.items():
            cur = mem.stationary_frames
            before = prev.get(key, 0)
            # counter incremented -> stationary this tick; reset to 0 -> not
            seq.append((key, int(rec["frame_index"]), cur > before))
            prev[key] = cur
            if mem.release_frame is not None:
                ever_released.add(key)
    # keep only pairs that ever reached the released state (the only ones whose
    # stationary counter feeds the BAG_ON_GROUND transition). Earlier version
    # filtered per-tick on release_frame, which silently dropped the ticks
    # recorded before release and under-reported the run length.
    return [(k, f, s) for (k, f, s) in seq if k in ever_released]


def max_run_with_tolerance(flags, k):
    """Longest run allowing up to k consecutive False ticks inside it."""
    best = 0
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            miss = 0
            count = 0
            while j < n:
                if flags[j]:
                    count += 1
                else:
                    miss += 1
                    if miss > k:
                        break
                j += 1
            best = max(best, count)
            i = j if miss > k else j
            # advance past the gaps we consumed so we don't re-scan them
            while i > 0 and not flags[i - 1]:
                i -= 1
            i = max(i, j - miss) if miss else j
            if i <= j - miss:
                i = j - miss + 1
        else:
            i += 1
    return best


def max_run_true_only(flags):
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    print("TOLERANCE ANALYSIS -- longest sustainable persistence per video")
    print("k = number of consecutive non-stationary ticks the counter tolerates")
    print("(k=0 is the CURRENT production behaviour: any single miss resets to 0)\n")

    rows = {}
    for stem in stems:
        seq = stationary_sequence(stem)
        if not seq:
            rows[stem] = None
            continue
        by_pair = {}
        for key, fi, flag in seq:
            by_pair.setdefault(key, []).append(flag)
        best = {}
        for key, flags in by_pair.items():
            best[key] = {k: max_run_with_tolerance(flags, k) for k in (0, 1, 2, 3, 5)}
            best[key]["true_only"] = max_run_true_only(flags)
            best[key]["n_ticks"] = len(flags)
        # the pair that carries the event is the one with the longest k=0 run
        top = max(best.items(), key=lambda kv: kv[1][0])
        rows[stem] = (top[0], top[1])

    hdr = (f"{'video':<12} {'pair':<14} {'ticks':>6} "
           + " ".join(f"{'k='+str(k):>7}" for k in (0, 1, 2, 3, 5)))
    print(hdr)
    print("-" * len(hdr))
    for stem in stems:
        r = rows[stem]
        if r is None:
            print(f"{stem:<12} {'(never released)':<14}")
            continue
        key, b = r
        cells = " ".join(f"{b[k]:>7}" for k in (0, 1, 2, 3, 5))
        print(f"{stem:<12} {'P'+str(key[0])+'->B'+str(key[1]):<14} {b['n_ticks']:>6} {cells}")

    print("\nSame, in SECONDS (8 ticks = 1.00 s)")
    print(f"{'video':<12} " + " ".join(f"{'k='+str(k):>7}" for k in (0, 1, 2, 3, 5)))
    print("-" * (12 + 8 * 5))
    for stem in stems:
        r = rows[stem]
        if r is None:
            print(f"{stem:<12} " + " ".join(f"{'-':>7}" for _ in (0, 1, 2, 3, 5)))
            continue
        _, b = r
        print(f"{stem:<12} " + " ".join(f"{b[k]/ANALYSIS_FPS:>7.2f}" for k in (0, 1, 2, 3, 5)))

    print("\nBinding constraint across the 4 currently-working videos:")
    vals = [rows[s][1][0] for s in stems if rows[s]]
    if vals:
        print(f"  min over videos of k=0 run = {min(vals)} ticks = {min(vals)/ANALYSIS_FPS:.2f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
