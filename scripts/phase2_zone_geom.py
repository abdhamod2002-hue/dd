"""Phase 2 — carrying-zone geometry sensitivity.

The weight sweep found that the ZONE component is load-bearing: remove it and
the pose score flips to the runner-up on 8/8 ticks of the confirmed bags. So
the two geometry knobs that define the zone are NOT free parameters — they
need to be measured, not guessed.

  pose_zone_horizontal_margin  half-width added to the body box, as a fraction
                               of body width. 0.30 default.
  pose_zone_vertical_below     how far the band extends below the bbox bottom,
                               as a fraction of body height. 0.10 default.
  pose_zone_vertical_above     how far above the shoulder line the band starts,
                               as a fraction of body height. 0.05 default.

Metrics (all label-free):
  FIRE   fraction of candidate pairs where zone == 1. A zone that never fires
         is dead weight; one that always fires carries no information.
  Q1     agreement with the legacy ranking ON THE CONFIRMED EVENT'S BAG.
  Q2     mean signed pose gap (legacy#1 - legacy#2) over all multi-person ticks.
"""
from __future__ import annotations

import json
import os
import sys
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

H_MARGINS = [0.0, 0.15, 0.30, 0.50, 0.75]
V_BELOWS = [0.0, 0.10, 0.25, 0.50]
V_ABOVES = [0.05, 0.20]


def collect(stem: str, hm: float, vb: float, va: float) -> dict:
    cfg = load_event_config()
    data = cfg.to_dict()
    data.update({
        "pose_association_enabled": True,
        "pose_score_blend": 0.5,
        "pose_zone_horizontal_margin": hm,
        "pose_zone_vertical_below": vb,
        "pose_zone_vertical_above": va,
    })
    cfg = EventDetectorConfig.from_dict(data)

    det = LitteringEventDetector(cfg)
    det.reset()
    races = []
    n_zone1 = 0
    n_pairs = 0
    original = det._select_primary_associations

    def spy(infos):
        nonlocal n_zone1, n_pairs
        by_bag = {}
        for i in infos:
            by_bag.setdefault(i.bag_id, []).append(i)
            n_pairs += 1
            z = (i.pose_components or {}).get("zone")
            if z is not None and z > 0.5:
                n_zone1 += 1
        for bid, cands in by_bag.items():
            if len(cands) >= 2:
                races.append({
                    "frame": int(cands[0].frame_index),
                    "bag_id": int(bid),
                    "cands": [{
                        "pid": int(c.person_id),
                        "legacy": round(float(c.legacy_score), 6),
                        "pose": None if c.pose_score is None else round(float(c.pose_score), 6),
                    } for c in cands],
                })
        return original(infos)

    det._select_primary_associations = spy
    for rec in load_ticks(stem):
        persons, bags = build_frame(rec)
        det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_index"]))
    det.finalize()

    conf = list(det.confirmed_events)
    return {
        "races": races,
        "ev_bag": int(conf[0].bag_track_id) if conf else None,
        "ev_conf": round(float(conf[0].confidence), 4) if conf else None,
        "verdict": "CONFIRMED" if conf else "REJ",
        "n_zone1": n_zone1, "n_pairs": n_pairs,
    }


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PHASE 2 — CARRYING-ZONE GEOMETRY SENSITIVITY  (blend=0.5, weights 0.4/0.3/0.3)")
    print("FIRE = %% of candidate pairs with zone==1 | Q1 = agreement on confirmed bag | Q2 = signed separation\n")

    grid = []
    for va in V_ABOVES:
        print(f"  vertical_above = {va:.2f}")
        print(f"    {'h_margin':>9} {'v_below':>8} {'FIRE':>7} {'Q1':>8} {'nQ1':>5} "
              f"{'Q2':>8} {'nQ2':>5}  verdicts")
        print("    " + "-" * 66)
        for hm in H_MARGINS:
            for vb in V_BELOWS:
                rows = {s: collect(s, hm, vb, va) for s in stems}
                fire_num = sum(r["n_zone1"] for r in rows.values())
                fire_den = max(1, sum(r["n_pairs"] for r in rows.values()))
                gaps, onbag, agree = [], 0, 0
                for s, r in rows.items():
                    for race in r["races"]:
                        ranked = sorted(race["cands"], key=lambda c: -c["legacy"])
                        if len(ranked) < 2:
                            continue
                        a, b = ranked[0], ranked[1]
                        if a["pose"] is None or b["pose"] is None:
                            continue
                        gap = a["pose"] - b["pose"]
                        gaps.append(gap)
                        if race["bag_id"] == r["ev_bag"]:
                            onbag += 1
                            if gap > 0:
                                agree += 1
                q1 = (agree / onbag) if onbag else float("nan")
                q2 = (sum(gaps) / len(gaps)) if gaps else float("nan")
                bad = [s for s, r in rows.items()
                       if r["verdict"] != "CONFIRMED" and s != "IMG_5115"]
                vstr = "ALL PRESERVED" if not bad else "BREAKS " + ",".join(bad)
                print(f"    {hm:>9.2f} {vb:>8.2f} {100*fire_num/fire_den:>6.1f}% "
                      f"{q1:>7.1%} {onbag:>5} {q2:>+8.4f} {len(gaps):>5}  {vstr}")
                grid.append({"v_above": va, "h_margin": hm, "v_below": vb,
                             "fire": 100 * fire_num / fire_den, "q1": q1,
                             "n_q1": onbag, "q2": q2, "n_q2": len(gaps),
                             "breaks": bad})
        print()

    (OUT_DIR / "zone_geom.json").write_text(
        json.dumps(grid, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'zone_geom.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
