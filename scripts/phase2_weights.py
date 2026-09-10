"""Phase 2 — component weight sensitivity.

The brief specified wrist=0.4 / zone=0.3 / motion=0.3. That is a reasonable
prior, but "it was in the brief" is not evidence. This sweep asks whether the
weights actually matter on these 5 clips, using two label-free quality signals:

  Q1. AGREEMENT ON THE CONFIRMED BAG
      On ticks where >=2 persons compete for the bag that ends up producing
      the confirmed event, does the pose score rank the legacy #1 above the
      legacy #2? This is the video that actually matters — a flip here would
      change the event.

  Q2. SEPARATION
      Mean pose-score gap between legacy #1 and legacy #2 on ALL multi-person
      ticks. Bigger gap = the pose score is doing real discrimination work
      rather than hovering near tie.

If every weight vector gives the same Q1 and comparable Q2, the weights are
not a sensitive knob on this data and the brief's defaults should be kept
(simplicity + the prior from the literature). If some vector is clearly better,
adopt it and say why.
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
from scripts.phase2_final import run as replay  # noqa: E402

OUT_DIR = REPO / ".audit" / "phase2"

# name -> (wrist, zone, motion)
WEIGHT_SETS = {
    "brief 0.4/0.3/0.3": (0.4, 0.3, 0.3),
    "uniform 1/3 each": (1 / 3, 1 / 3, 1 / 3),
    "wrist-heavy .5/.25/.25": (0.5, 0.25, 0.25),
    "zone-heavy .25/.5/.25": (0.25, 0.5, 0.25),
    "motion-heavy .25/.25/.5": (0.25, 0.25, 0.5),
    "no wrist 0/.5/.5": (0.0, 0.5, 0.5),
    "no motion .5/.5/0": (0.5, 0.5, 0.0),
    "no zone .5/0/.5": (0.5, 0.0, 0.5),
    "wrist only 1/0/0": (1.0, 0.0, 0.0),
    "motion only 0/0/1": (0.0, 0.0, 1.0),
}


def collect(stem: str, weights) -> dict:
    """Replay once, return the full candidate field + the confirmed bag id."""
    w_wrist, w_zone, w_motion = weights
    cfg = load_event_config()
    data = cfg.to_dict()
    data.update({
        "pose_association_enabled": True,
        "pose_score_blend": 0.5,
        "pose_weight_wrist": w_wrist,
        "pose_weight_zone": w_zone,
        "pose_weight_motion": w_motion,
    })
    cfg = EventDetectorConfig.from_dict(data)

    det = LitteringEventDetector(cfg)
    det.reset()
    races = []
    original = det._select_primary_associations

    def spy(infos):
        by_bag = {}
        for i in infos:
            by_bag.setdefault(i.bag_id, []).append(i)
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

    conf = [ev for ev in det.confirmed_events]
    ev_bag = int(conf[0].bag_track_id) if conf else None
    ev_conf = round(float(conf[0].confidence), 4) if conf else None
    return {"races": races, "ev_bag": ev_bag, "ev_conf": ev_conf,
            "verdict": "CONFIRMED" if conf else
                       str(det.rejected_events[0].reason if det.rejected_events else "-")}


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("PHASE 2 — COMPONENT WEIGHT SENSITIVITY  (blend fixed at 0.5)")
    print("Q1 = agreement with legacy ranking ON THE CONFIRMED EVENT'S BAG")
    print("Q2 = mean |pose#1 - pose#2| over ALL multi-person ticks\n")

    summary = {}
    for name, w in WEIGHT_SETS.items():
        rows = []
        tot_onbag = tot_agree = 0
        gaps = []
        verdicts = {}
        for stem in stems:
            r = collect(stem, w)
            verdicts[stem] = (r["verdict"], r["ev_conf"])
            for race in r["races"]:
                ranked = sorted(race["cands"], key=lambda c: -c["legacy"])
                a, b = ranked[0], ranked[1]
                if a["pose"] is None or b["pose"] is None:
                    continue
                gap = a["pose"] - b["pose"]
                gaps.append(gap)
                if race["bag_id"] == r["ev_bag"]:
                    tot_onbag += 1
                    if gap > 0:
                        tot_agree += 1
        q1 = (tot_agree / tot_onbag) if tot_onbag else float("nan")
        # signed: positive means pose ranks legacy#1 higher
        q2 = (sum(gaps) / len(gaps)) if gaps else float("nan")
        summary[name] = {"q1": q1, "q2": q2, "n_onbag": tot_onbag,
                         "n_all": len(gaps), "verdicts": verdicts}
        print(f"  {name:<24} Q1={q1:>6.1%} (n={tot_onbag:>2})  "
              f"Q2={q2:>+7.4f} (n={len(gaps):>3})")
        ok = all(v[0] == "CONFIRMED" or v[1] is None for v in verdicts.values())
        bad = [s for s, v in verdicts.items()
               if v[0] != "CONFIRMED" and s != "IMG_5115"]
        print(f"  {'':<24} verdicts: "
              f"{'ALL PRESERVED' if not bad else 'BREAKS ' + str(bad)}")

    print("\n  confidence per video (blend=0.5):")
    hdr = f"    {'weights':<24}" + "".join(f"{s:<11}" for s in stems)
    print(hdr)
    for name in WEIGHT_SETS:
        cells = []
        for stem in stems:
            v, c = summary[name]["verdicts"][stem]
            cells.append(f"{(f'{c:.4f}' if c is not None else v[:10]):<11}")
        print(f"    {name:<24}" + "".join(cells))

    (OUT_DIR / "weights.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'weights.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
