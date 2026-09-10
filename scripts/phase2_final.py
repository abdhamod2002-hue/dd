"""Phase 2 — final measurement: pick the blend, document the fallback, cost it.

Answers three questions the report must answer with numbers:

1. EVENT LEVEL: for each candidate blend, the verdict / confidence /
   association score / ambiguity counters on all 5 videos.
2. FALLBACK TIERS: how often each graceful-degradation path is actually taken
   (all three cues / wrist missing / motion missing / no pose at all).
3. COST: wall-clock of the full replay with the feature OFF vs ON. The pose
   itself is already produced by MoveNet for other purposes, so this measures
   only the ADDED cost of the Phase-2 scoring path.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(r"D:\HO")
os.chdir(REPO)
sys.path.insert(0, str(REPO))

from littering_event_detector import (  # noqa: E402
    EventDetectorConfig, LitteringEventDetector, load_event_config,
)
from scripts.phase1_sweep import CACHE_DIR, load_ticks  # noqa: E402
from scripts.phase1_probe_runs import build_frame  # noqa: E402

OUT_DIR = REPO / ".audit" / "phase2"


def run(stem: str, overrides: dict, tiers=None):
    cfg = load_event_config()
    data = cfg.to_dict()
    data.update(overrides or {})
    cfg = EventDetectorConfig.from_dict(data)

    det = LitteringEventDetector(cfg)
    det.reset()

    original = det._select_primary_associations

    def spy(infos):
        if tiers is not None:
            for i in infos:
                if i.pose_score is None:
                    tiers["no_pose"] += 1
                    continue
                p = i.pose_components or {}
                missing = tuple(sorted(k for k in ("wrist", "zone", "motion")
                                       if p.get(k) is None))
                tiers[missing] += 1
        return original(infos)

    det._select_primary_associations = spy

    for rec in load_ticks(stem):
        persons, bags = build_frame(rec)
        det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_index"]))
    det.finalize()

    rows = []
    for ev in det.confirmed_events + det.rejected_events:
        d = ev.details or {}
        rows.append({
            "confirmed": bool(ev.confirmed),
            "reason": str(ev.reason),
            "confidence": round(float(ev.confidence), 4),
            "person": int(ev.person_track_id),
            "bag": int(ev.bag_track_id),
            "assoc_score": round(float(ev.evidence.get("association_score", 0.0)), 4),
            "ambiguous_frames": d.get("ambiguous_frames"),
            "other_person_closer_frames": d.get("other_person_closer_frames"),
            "fallback_frames": d.get("fallback_frames"),
            "carried_frames": d.get("carried_frames"),
            "stationary_frames": d.get("stationary_frames"),
        })
    return {"rows": rows}


def timed_replay(stem: str, overrides: dict, repeats: int = 5) -> float:
    """Median wall-clock seconds for one full replay of one video."""
    ticks = list(load_ticks(stem))
    frames = [build_frame(rec) for rec in ticks]
    ts = [float(rec["timestamp"]) for rec in ticks]
    fi = [int(rec["frame_index"]) for rec in ticks]

    cfg = load_event_config()
    data = cfg.to_dict()
    data.update(overrides or {})
    cfg = EventDetectorConfig.from_dict(data)

    times = []
    for _ in range(repeats):
        det = LitteringEventDetector(cfg)
        det.reset()
        t0 = time.perf_counter()
        for (persons, bags), a, b in zip(frames, ts, fi):
            det.update(persons, bags, a, b)
        det.finalize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blends = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]

    print("PHASE 2 — FINAL MEASUREMENT")
    print("blend=0.0 == current production (pose disabled entirely)\n")

    results = {}
    print("=" * 100)
    print("1. EVENT-LEVEL RESULT PER BLEND")
    print("=" * 100)
    for stem in stems:
        print(f"\n  {stem}")
        print(f"    {'blend':>5} {'verdict':>22} {'conf':>8} {'assoc':>8} "
              f"{'ambg':>5} {'closer':>6} {'carried':>7} {'stat':>5}")
        for b in blends:
            ov = {"pose_association_enabled": True if b > 0 else False,
                  "pose_score_blend": b}
            res = run(stem, ov)
            results[f"{b}|{stem}"] = res
            conf = [r for r in res["rows"] if r["confirmed"]]
            if conf:
                c = conf[0]
                print(f"    {b:>5.2f} {'CONFIRMED':>22} {c['confidence']:>8.4f} "
                      f"{c['assoc_score']:>8.4f} {str(c['ambiguous_frames']):>5} "
                      f"{str(c['other_person_closer_frames']):>6} "
                      f"{str(c['carried_frames']):>7} {str(c['stationary_frames']):>5} "
                      f"P{c['person']}/bag{c['bag']}")
            else:
                r0 = res["rows"][0]
                print(f"    {b:>5.2f} {'REJ:'+r0['reason'][:18]:>22} "
                      f"{r0['confidence']:>8.4f} {r0['assoc_score']:>8.4f}")

    # ---------------- 2. fallback tiers ----------------
    print("\n" + "=" * 100)
    print("2. FALLBACK TIERS  (blend=0.5, fraction of evaluated candidate pairs)")
    print("=" * 100)
    print("     tuple = which components were MISSING")
    tiers_all = Counter()
    for stem in stems:
        tiers = Counter()
        run(stem, {"pose_association_enabled": True, "pose_score_blend": 0.5},
            tiers=tiers)
        tiers_all.update(tiers)
        tot = max(1, sum(tiers.values()))
        print(f"\n  {stem}  (n={tot})")
        for k, v in sorted(tiers.items(), key=lambda kv: -kv[1]):
            label = ("all three cues present" if k == ()
                     else ("NO pose at all -> 100% legacy" if k == "no_pose"
                           else "missing: " + ", ".join(k)))
            print(f"     {100*v/tot:>6.2f}%  {label}")
    tot = max(1, sum(tiers_all.values()))
    print(f"\n  ALL VIDEOS (n={tot})")
    for k, v in sorted(tiers_all.items(), key=lambda kv: -kv[1]):
        label = ("all three cues present" if k == ()
                 else ("NO pose at all -> 100% legacy" if k == "no_pose"
                       else "missing: " + ", ".join(k)))
        print(f"     {100*v/tot:>6.2f}%  {label}")

    # ---------------- 3. cost ----------------
    print("\n" + "=" * 100)
    print("3. ADDED CPU COST  (median of 5 full replays, CPU only)")
    print("=" * 100)
    print(f"  {'video':<10} {'OFF (ms)':>10} {'ON (ms)':>10} {'delta':>9} {'%':>7}")
    tot_off = tot_on = 0.0
    for stem in stems:
        off = timed_replay(stem, {"pose_association_enabled": False,
                                  "pose_score_blend": 0.0})
        on = timed_replay(stem, {"pose_association_enabled": True,
                                 "pose_score_blend": 0.5})
        tot_off += off
        tot_on += on
        print(f"  {stem:<10} {off*1000:>10.1f} {on*1000:>10.1f} "
              f"{(on-off)*1000:>+9.1f} {100*(on-off)/max(1e-9, off):>+6.1f}%")
    print(f"  {'TOTAL':<10} {tot_off*1000:>10.1f} {tot_on*1000:>10.1f} "
          f"{(tot_on-tot_off)*1000:>+9.1f} {100*(tot_on-tot_off)/max(1e-9, tot_off):>+6.1f}%")

    (OUT_DIR / "final.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'final.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
