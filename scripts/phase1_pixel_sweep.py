"""Phase-1 item #4, end-to-end: does the absolute-pixel gate change any verdict?

Adds stationary_max_pixel_step alongside (NOT instead of) the existing ratio
gate, at the baseline min_stationary_frames=8, and reports the per-video
confirm/reject outcome for each candidate cap.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from scripts.phase1_sweep import (  # noqa: E402
    ANALYSIS_FPS, CACHE_DIR, OUT_DIR, replay, summarise,
)

CAPS = [None, 10, 12, 15, 20, 25, 30, 40]


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    results = {}
    for cap in CAPS:
        results[str(cap)] = {}
        for stem in stems:
            det, cfg = replay(stem, {"stationary_max_pixel_step": cap})
            results[str(cap)][stem] = summarise(det, stem)

    print("PHASE 1 / ITEM 4 -- ABSOLUTE PIXEL GATE (min_stationary_frames held at 8)")
    print("cap=None is the current production behaviour (gate disabled)\n")
    hdr = f"{'cap(px)':>8} | " + " | ".join(f"{s:<30}" for s in stems)
    print(hdr)
    print("-" * len(hdr))
    for cap in CAPS:
        cells = []
        for stem in stems:
            s = results[str(cap)][stem]
            if s["confirmed"]:
                cell = f"CONFIRMED x{s['confirmed']}"
            elif s["rejected"]:
                reasons = sorted({r["reason"] for r in s["rows"]})
                cell = "REJ:" + ",".join(r[:26] for r in reasons)
            else:
                cell = "no-decision"
            cells.append(f"{cell:<30}")
        print(f"{str(cap):>8} | " + " | ".join(cells))

    print("\nCONFIDENCE OF THE CONFIRMED EVENT PER VIDEO (blank = not confirmed)")
    print(f"{'cap(px)':>8} | " + " | ".join(f"{s:>10}" for s in stems))
    print("-" * (8 + 3 + 13 * len(stems)))
    for cap in CAPS:
        cells = []
        for stem in stems:
            s = results[str(cap)][stem]
            conf = [r["confidence"] for r in s["rows"] if r["confirmed"]]
            cells.append(f"{conf[0]:>10.4f}" if conf else f"{'-':>10}")
        print(f"{str(cap):>8} | " + " | ".join(cells))

    (OUT_DIR / "sweep_pixel.json").write_text(json.dumps(results, indent=2, default=str),
                                              encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'sweep_pixel.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
