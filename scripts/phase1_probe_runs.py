"""Phase-1 diagnostic: measure the REAL headroom of the stationary counter.

The sweep showed MIN_STATIONARY_FRAMES cannot be raised even by 1 tick without
breaking videos. That is only explicable if the counter is a CONSECUTIVE run
(it resets to 0 on any non-stationary tick). This probe measures, per video and
per (person,bag) pair, the longest consecutive stationary run actually observed
after release -- i.e. the true ceiling any threshold must respect.

Method: replay cached ticks with min_stationary_frames set impossibly high
(10**6) so the pair can never transition to BAG_ON_GROUND. The counter then
keeps accumulating and we can read its maximum, which IS the longest
consecutive run. No production code is modified.
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


def build_frame(rec):
    persons = []
    for p in rec["persons"]:
        kpd = p.get("keypoints")
        kp = None
        if kpd:
            kp = DetectorKeypoints(
                left_wrist=tuple(kpd["left_wrist"]) if kpd.get("left_wrist") else None,
                right_wrist=tuple(kpd["right_wrist"]) if kpd.get("right_wrist") else None,
                torso_center=tuple(kpd["torso_center"]) if kpd.get("torso_center") else None,
                left_shoulder=tuple(kpd["left_shoulder"]) if kpd.get("left_shoulder") else None,
                right_shoulder=tuple(kpd["right_shoulder"]) if kpd.get("right_shoulder") else None,
            )
        persons.append(DetectorPerson(track_id=int(p["track_id"]), bbox=tuple(p["bbox"]),
                                      confidence=float(p["confidence"]), keypoints=kp))
    bags = []
    for o in rec["objects"]:
        src = o.get("source", "yolo")
        bags.append(DetectorBag(track_id=int(o["track_id"]), bbox=tuple(o["bbox"]),
                                confidence=float(o["confidence"]),
                                class_name=str(o["class_name"]), source=src,
                                yolo_confirmed=(src == "yolo")))
    return persons, bags


def probe(stem: str):
    cfg = load_event_config()
    data = cfg.to_dict()
    data["min_stationary_frames"] = 10 ** 6   # never transition -> free-running counter
    data["min_abandonment_frames"] = 10 ** 6
    cfg = EventDetectorConfig.from_dict(data)

    det = LitteringEventDetector(cfg)
    det.reset()

    peak = {}        # pair -> max consecutive stationary ticks seen
    ticks_after_release = {}
    for rec in load_ticks(stem):
        persons, bags = build_frame(rec)
        det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_index"]))
        for key, mem in det._pairs.items():
            cur = mem.stationary_frames
            if cur > peak.get(key, 0):
                peak[key] = cur
            if mem.release_frame is not None:
                ticks_after_release[key] = ticks_after_release.get(key, 0) + 1
    return peak, ticks_after_release, det


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    print("LONGEST CONSECUTIVE STATIONARY RUN OBSERVED (analysis_fps=%.1f)" % ANALYSIS_FPS)
    print("(a threshold above a video's run can never be satisfied by that video)\n")
    print(f"{'video':<12} {'pair':<16} {'peak run':>9} {'seconds':>8} {'ticks after release':>20}")
    print("-" * 72)
    summary = {}
    for stem in stems:
        peak, tar, det = probe(stem)
        best_pair, best_run = None, 0
        for key, run in sorted(peak.items(), key=lambda kv: -kv[1]):
            pair = f"P{key[0]}->B{key[1]}"
            print(f"{stem:<12} {pair:<16} {run:>9} {run/ANALYSIS_FPS:>8.2f} {tar.get(key,0):>20}")
            if run > best_run:
                best_pair, best_run = pair, run
        summary[stem] = {"best_pair": best_pair, "peak_run_ticks": best_run,
                         "peak_run_seconds": round(best_run / ANALYSIS_FPS, 3)}
        print()
    print("=" * 72)
    print(f"{'video':<12} {'best pair':<16} {'usable ceiling (ticks)':>22} {'seconds':>9}")
    print("-" * 72)
    for stem, s in summary.items():
        print(f"{stem:<12} {str(s['best_pair']):<16} {s['peak_run_ticks']:>22} {s['peak_run_seconds']:>9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
