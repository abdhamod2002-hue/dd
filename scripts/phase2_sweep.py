"""Phase 2 — pose-based association: measure before/after on the 5 real videos.

Reuses the Phase-1 cached analysis ticks (identical inputs for every config).

What it measures
----------------
1. Verdict + confidence per video, for each candidate config (regression gate).
2. Which person track wins the association, and the SCORE MARGIN over the
   runner-up on multi-person ticks. A bigger margin = a more decisive, less
   ambiguous association. This is the closest thing to "association accuracy"
   that is measurable WITHOUT identity labels (none exist for these clips).
3. Pose-cue availability: how often each component (wrist / zone / motion) is
   computable, which drives the occlusion-fallback design.

Nothing here alters production behaviour; all knobs are passed as overrides.
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


def run(stem: str, overrides: dict):
    """Replay one video; capture verdicts AND the per-tick association race."""
    cfg = load_event_config()
    data = cfg.to_dict()
    data.update(overrides or {})
    cfg = EventDetectorConfig.from_dict(data)

    det = LitteringEventDetector(cfg)
    det.reset()

    races = []          # per (tick, bag) with >=2 candidate persons
    pose_avail = {"wrist": 0, "zone": 0, "motion": 0, "none": 0, "total": 0}

    # wrap the selection step to record the full candidate field
    original = det._select_primary_associations

    def spy(infos):
        by_bag = {}
        for i in infos:
            by_bag.setdefault(i.bag_id, []).append(i)
        for bid, cands in by_bag.items():
            if len(cands) >= 2:
                ranked = sorted(cands, key=lambda x: x.score, reverse=True)
                best, second = ranked[0], ranked[1]
                races.append({
                    "frame": best.frame_index,
                    "bag_id": int(bid),
                    "winner": int(best.person_id),
                    "runner_up": int(second.person_id),
                    "margin": round(best.score - second.score, 4),
                    "winner_score": round(best.score, 4),
                    "runner_score": round(second.score, 4),
                    "winner_pose": best.pose_score,
                    "runner_pose": second.pose_score,
                })
            for c in cands:
                pose_avail["total"] += 1
                if c.pose_score is None:
                    pose_avail["none"] += 1
                else:
                    for k, v in (c.pose_components or {}).items():
                        if v is not None:
                            pose_avail[k] = pose_avail.get(k, 0) + 1
        return original(infos)

    det._select_primary_associations = spy

    for rec in load_ticks(stem):
        persons, bags = build_frame(rec)
        det.update(persons, bags, float(rec["timestamp"]), int(rec["frame_index"]))
    det.finalize()

    rows = []
    for ev in det.confirmed_events + det.rejected_events:
        rows.append({
            "confirmed": bool(ev.confirmed),
            "reason": str(ev.reason),
            "confidence": round(float(ev.confidence), 4),
            "person": int(ev.person_track_id),
            "bag": int(ev.bag_track_id),
            "assoc_score": round(float(ev.evidence.get("association_score", 0.0)), 4),
            "details": ev.details,
        })
    return {"rows": rows, "races": races, "pose_avail": pose_avail}


def summarise(res: dict) -> str:
    rows = res["rows"]
    if not rows:
        return "no-decision"
    out = []
    for r in rows:
        tag = "CONFIRMED" if r["confirmed"] else "REJ:" + r["reason"][:22]
        out.append(f"{tag}({r['confidence']:.3f},P{r['person']},a={r['assoc_score']:.2f})")
    return " | ".join(out)


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blends = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    results = {}

    print("PHASE 2 — POSE SCORE BLEND SWEEP")
    print("blend=0.0 is the current production behaviour (legacy score only)\n")
    hdr = f"{'blend':>6} | " + " | ".join(f"{s:<9}" for s in stems)
    print(hdr)
    print("-" * len(hdr))
    for b in blends:
        cells = []
        for stem in stems:
            res = run(stem, {"pose_association_enabled": True, "pose_score_blend": b})
            results[f"{b}|{stem}"] = res
            conf = [r for r in res["rows"] if r["confirmed"]]
            cells.append(f"{('OKx'+str(len(conf))) if conf else 'none':<9}")
        print(f"{b:>6.2f} | " + " | ".join(cells))

    print("\nCONFIDENCE + SELECTED PERSON + ASSOCIATION SCORE")
    for stem in stems:
        print(f"\n  {stem}")
        for b in blends:
            res = results[f"{b}|{stem}"]
            conf = [r for r in res["rows"] if r["confirmed"]]
            if conf:
                c = conf[0]
                print(f"    blend={b:.2f}  conf={c['confidence']:.4f}  "
                      f"person=P{c['person']}  assoc_score={c['assoc_score']:.4f}")
            else:
                rej = res["rows"][0]["reason"] if res["rows"] else "-"
                print(f"    blend={b:.2f}  NOT CONFIRMED ({rej})")

    print("\nPOSE-CUE AVAILABILITY (blend=0.5, fraction of candidate pairs)")
    for stem in stems:
        r = run(stem, {"pose_association_enabled": True, "pose_score_blend": 0.5})
        pa = r["pose_avail"]
        tot = max(1, pa["total"])
        print(f"  {stem:<10} total={pa['total']:>5}  wrist={100*pa['wrist']/tot:5.1f}%  "
              f"zone={100*pa['zone']/tot:5.1f}%  motion={100*pa['motion']/tot:5.1f}%  "
              f"no-pose-fallback={100*pa['none']/tot:5.1f}%")

    (OUT_DIR / "sweep_blend.json").write_text(
        json.dumps({k: {"rows": v["rows"], "n_races": len(v["races"])}
                    for k, v in results.items()}, indent=2, default=str),
        encoding="utf-8")
    print(f"\nwrote {OUT_DIR / 'sweep_blend.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
