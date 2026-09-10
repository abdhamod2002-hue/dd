"""Phase-1 item #4: size the absolute-pixel stationarity gate with real data.

Before picking a pixel cap we must know how much the bag centroid actually
drifts per stationarity window on these clips. A cap chosen blind (e.g. the
15 px the brief suggests) could silently delete every stationary tick.

For every (person,bag) pair we dump, per analysis tick:
  * step_px  -- centroid displacement across the 3-tick stationarity window
  * bag_size -- max(bbox width, height), i.e. the current ratio denominator
  * step_norm-- step_px / bag_size  (what the ratio gate already checks)
  * ratio_pass / pixel_pass(cap) for each candidate cap

This is measurement only. No production code path is altered.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from littering_event_detector import (  # noqa: E402
    DetectorBag, DetectorKeypoints, DetectorPerson,
    EventDetectorConfig, LitteringEventDetector, load_event_config,
)
from scripts.phase1_sweep import ANALYSIS_FPS, CACHE_DIR, load_ticks  # noqa: E402
from scripts.phase1_probe_runs import build_frame  # noqa: E402

CAPS = [10, 15, 20, 25, 30, 40, 50, 75, 100]


def probe(stem: str):
    """Record the per-tick stationarity inputs by shadowing the detector's history."""
    cfg = load_event_config()
    data = cfg.to_dict()
    data["min_stationary_frames"] = 10 ** 6
    data["min_abandonment_frames"] = 10 ** 6
    cfg = EventDetectorConfig.from_dict(data)
    det = LitteringEventDetector(cfg)
    det.reset()

    samples = []   # (pair, t_rel, step_px, bag_size, step_norm, ratio_pass)
    for rec in load_ticks(stem):
        persons, bags = build_frame(rec)
        det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_index"]))
        for bag in bags:
            hist = det._bag_history.get(bag.track_id)
            if not hist or len(hist) < 2:
                continue
            window = max(2, int(cfg.stationary_window_frames))
            prev = hist[-min(window, len(hist))]
            curr = hist[-1]
            step_px = ((curr[1][0] - prev[1][0]) ** 2 + (curr[1][1] - prev[1][1]) ** 2) ** 0.5
            bag_size = max(1.0, bag.bbox[3] - bag.bbox[1], bag.bbox[2] - bag.bbox[0])
            step_norm = step_px / bag_size
            samples.append({
                "bag_id": int(bag.track_id),
                "t": round(rec["t_rel"], 3),
                "step_px": round(step_px, 2),
                "bag_size": round(bag_size, 1),
                "step_norm": round(step_norm, 4),
                "ratio_pass": bool(step_norm < cfg.stationary_distance_ratio),
            })
    return samples


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    cfg = load_event_config()
    ratio = cfg.stationary_distance_ratio

    print("BAG CENTROID DRIFT PER STATIONARITY WINDOW (3 ticks @ %.1f fps)" % ANALYSIS_FPS)
    print(f"current ratio gate: step_px / bag_size < {ratio}\n")

    all_rows = {}
    for stem in stems:
        s = probe(stem)
        all_rows[stem] = s
        if not s:
            print(f"{stem}: no bag history\n")
            continue
        px = sorted(x["step_px"] for x in s)
        n = len(px)
        print(f"{stem}: {n} samples | step_px  p50={px[n//2]:.1f} "
              f"p75={px[int(n*0.75)]:.1f} p90={px[int(n*0.90)]:.1f} max={px[-1]:.1f} "
              f"| bag_size p50={sorted(x['bag_size'] for x in s)[n//2]:.0f}px")
    print()

    print("HOW MANY TICKS SURVIVE EACH CANDIDATE PIXEL CAP")
    print("(baseline: how many ticks already pass the ratio gate)\n")
    hdr = f"{'video':<12} {'ratio_pass/total':>16} " + " ".join(f"{c:>7}" for c in CAPS)
    print(hdr + "   <- ticks surviving cap (px)")
    print("-" * len(hdr) + "-" * 26)
    for stem in stems:
        s = all_rows[stem]
        if not s:
            print(f"{stem:<12} {'n/a':>16} " + " ".join(f"{'-':>7}" for _ in CAPS))
            continue
        base = sum(1 for x in s if x["ratio_pass"])
        cells = []
        for c in CAPS:
            k = sum(1 for x in s if x["ratio_pass"] and x["step_px"] <= c)
            cells.append(f"{k:>7}")
        print(f"{stem:<12} {f'{base}/{len(s)}':>16} " + " ".join(cells))
    print()

    print("SAME, AS % OF RATIO-PASSING TICKS RETAINED")
    print(f"{'video':<12} " + " ".join(f"{c:>7}" for c in CAPS))
    print("-" * (12 + 8 * len(CAPS)))
    for stem in stems:
        s = all_rows[stem]
        if not s:
            print(f"{stem:<12} " + " ".join(f"{'-':>7}" for _ in CAPS))
            continue
        base = sum(1 for x in s if x["ratio_pass"]) or 1
        cells = []
        for c in CAPS:
            k = sum(1 for x in s if x["ratio_pass"] and x["step_px"] <= c)
            cells.append(f"{100.0*k/base:>6.0f}%")
        print(f"{stem:<12} " + " ".join(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
