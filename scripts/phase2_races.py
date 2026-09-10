"""Phase 2 — multi-person association races (v2).

The blend sweep told us the *verdicts* never break. But "the verdict didn't
break" is a weak result: it is also what you get when the new score is inert.
The question that actually decides whether pose association is worth adopting
is:

    On ticks where MORE THAN ONE person is a plausible owner of the bag, does
    the pose score separate them, and does it separate them the RIGHT WAY?

Measurable without identity labels:

  A. DELEGACY RANKING  rank candidates purely by ``legacy_score`` (the current
                       production criterion). This is a FIXED reference that
                       does not move with the blend.
  B. POSE AGREEMENT    does ``pose_score`` rank the legacy #1 above the legacy
                       #2? 100% = pose corroborates. ~50% = pose is noise.
                       0%   = pose systematically contradicts legacy.
  C. MARGIN            winner - runner-up under the blended score. Bigger =
                       more decisive, fewer ``ambiguous`` flags.
  D. WINNER FLIPS      does the blended winner ever differ from the legacy
                       winner, and on which bag/person?

v1 bug fixed: the v1 "agreement" metric compared pose against the *current
blend's* winner, which is circular by construction and returned 100% trivially.
Pose is now scored against the FIXED blend=0.0 ranking.
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
BLENDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def run(stem: str, overrides: dict):
    """Replay one video and capture the FULL per-tick candidate field."""
    cfg = load_event_config()
    data = cfg.to_dict()
    data.update(overrides or {})
    cfg = EventDetectorConfig.from_dict(data)

    det = LitteringEventDetector(cfg)
    det.reset()

    races = []
    ambiguous = 0
    total_primary = 0

    original = det._select_primary_associations

    def spy(infos):
        nonlocal ambiguous, total_primary
        by_bag = {}
        for i in infos:
            by_bag.setdefault(i.bag_id, []).append(i)
        for bid, cands in by_bag.items():
            total_primary += 1
            if len(cands) >= 2:
                races.append({
                    "frame": int(cands[0].frame_index),
                    "bag_id": int(bid),
                    "cands": [{
                        "pid": int(c.person_id),
                        "legacy": round(float(c.legacy_score), 6),
                        "pose": None if c.pose_score is None
                                else round(float(c.pose_score), 6),
                        "score": round(float(c.score), 6),
                        "parts": dict(c.pose_components or {}),
                    } for c in cands],
                })
        out = original(infos)
        ambiguous += sum(1 for i in out if i.ambiguous)
        return out

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
    return {"rows": rows, "races": races,
            "ambiguous": ambiguous, "primary": total_primary}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    s = sorted(xs)
    if not s:
        return float("nan")
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def main() -> int:
    stems = sorted(p.stem for p in CACHE_DIR.glob("*.jsonl"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_out = {}

    print("PHASE 2 — MULTI-PERSON ASSOCIATION RACES (v2)")
    print("Every tick where >=2 person tracks compete for the same bag.")
    print("Pose is scored against the FIXED blend=0.0 (legacy) ranking.\n")

    for stem in stems:
        print("=" * 80)
        print(f"{stem}")
        print("=" * 80)

        per_blend = {}
        for b in BLENDS:
            per_blend[b] = run(stem, {"pose_association_enabled": True,
                                      "pose_score_blend": b})

        base = per_blend[0.0]
        conf = [r for r in base["rows"] if r["confirmed"]]
        if conf:
            ev_bag, ev_person = conf[0]["bag"], conf[0]["person"]
            print(f"  confirmed event: person=P{ev_person}  bag={ev_bag}  "
                  f"conf={conf[0]['confidence']:.4f}")
        else:
            ev_bag = ev_person = None
            print(f"  NOT CONFIRMED ({base['rows'][0]['reason'] if base['rows'] else '-'})")

        races = base["races"]
        if not races:
            print("  no multi-person ticks -> pose association cannot be judged here\n")
            continue

        # index races by (frame, bag) for cross-blend lookup
        idx = {b: {(r["frame"], r["bag_id"]): r
                   for r in per_blend[b]["races"]} for b in BLENDS}
        keys = sorted(idx[0.0].keys())

        # pose_score is BLEND-INDEPENDENT, but it is deliberately not computed
        # at all when pose_score_blend == 0.0 (see _evaluate_pair: the guard is
        # ``pose_association_enabled and pose_score_blend > 0.0``). So the
        # legacy ranking and the pose values must be read from different runs:
        # legacy from blend 0.0, pose from the highest blend.
        POSE_B = BLENDS[-1]
        pose_by_pk = {}
        for r in per_blend[POSE_B]["races"]:
            for c in r["cands"]:
                pose_by_pk[(r["frame"], r["bag_id"], c["pid"])] = c
        parts_by_pk = pose_by_pk  # same records carry the components

        print(f"  multi-person ticks: {len(keys)}   "
              f"bags: {sorted(set(k[1] for k in keys))}")
        on_event_bag = [k for k in keys if k[1] == ev_bag]
        print(f"  of those, on the CONFIRMED event's bag ({ev_bag}): {len(on_event_bag)}")

        # ---- B. pose agreement vs FIXED legacy ranking -------------------
        print(f"\n  B. POSE AGREEMENT with the legacy ranking")
        print(f"     (does pose rank legacy-#1 above legacy-#2?)")
        n_both = 0
        n_agree = 0
        n_tie = 0
        for k in keys:
            r0 = idx[0.0][k]
            legacy_rank = sorted(r0["cands"], key=lambda c: -c["legacy"])
            l1, l2 = legacy_rank[0], legacy_rank[1]
            # pose values come from the POSE_B run (see note above)
            c1 = pose_by_pk.get((k[0], k[1], l1["pid"]))
            c2 = pose_by_pk.get((k[0], k[1], l2["pid"]))
            p1 = None if c1 is None else c1["pose"]
            p2 = None if c2 is None else c2["pose"]
            if p1 is None or p2 is None:
                continue
            n_both += 1
            if p1 > p2:
                n_agree += 1
            elif p1 == p2:
                n_tie += 1
        if n_both:
            print(f"     both-posed ticks: {n_both}   (pose sourced from blend={POSE_B:.2f} run)")
            print(f"     pose AGREES    : {n_agree} ({100*n_agree/n_both:.1f}%)")
            print(f"     pose CONTRADICTS: {n_both-n_agree-n_tie} "
                  f"({100*(n_both-n_agree-n_tie)/n_both:.1f}%)")
            print(f"     exact tie      : {n_tie} ({100*n_tie/n_both:.1f}%)")
        else:
            print("     no tick has pose on both top-2 legacy candidates")

        # ---- C. margin ---------------------------------------------------
        print(f"\n  C. MARGIN  (blended winner - runner-up)")
        print(f"     {'blend':>6} {'mean':>9} {'median':>9} {'min':>9} {'max':>9} "
              f"{'ambiguous':>10}")
        for b in BLENDS:
            ms = []
            for k in keys:
                r = idx[b].get(k)
                if not r or len(r["cands"]) < 2:
                    continue
                ranked = sorted(r["cands"], key=lambda c: -c["score"])
                ms.append(ranked[0]["score"] - ranked[1]["score"])
            print(f"     {b:>6.2f} {mean(ms):>9.4f} {median(ms):>9.4f} "
                  f"{min(ms):>9.4f} {max(ms):>9.4f} "
                  f"{str(per_blend[b]['ambiguous']) + '/' + str(per_blend[b]['primary']):>10}")

        # ---- D. winner flips vs legacy winner ----------------------------
        print(f"\n  D. WINNER FLIPS vs the legacy winner")
        for b in BLENDS[1:]:
            flips = []
            for k in keys:
                r0 = idx[0.0][k]
                rb = idx[b].get(k)
                if not rb:
                    continue
                w0 = sorted(r0["cands"], key=lambda c: -c["score"])[0]["pid"]
                wb = sorted(rb["cands"], key=lambda c: -c["score"])[0]["pid"]
                if w0 != wb:
                    flips.append((k[0], k[1], w0, wb))
            tag = f"blend={b:.2f}"
            print(f"     {tag}  flips={len(flips):>3}/{len(keys)}")
            for f, g, w0, wb in flips:
                star = " <-- CONFIRMED EVENT'S BAG" if g == ev_bag else ""
                print(f"          frame={f:<5} bag={g:<7} P{w0} -> P{wb}{star}")

        # ---- E. pose component separation, legacy #1 vs legacy #2 ---------
        print(f"\n  E. POSE COMPONENTS: legacy-#1 vs legacy-#2")
        for comp in ("wrist", "zone", "motion"):
            wv, rv = [], []
            for k in keys:
                r0 = idx[0.0][k]
                legacy_rank = sorted(r0["cands"], key=lambda c: -c["legacy"])
                if len(legacy_rank) < 2:
                    continue
                c1 = parts_by_pk.get((k[0], k[1], legacy_rank[0]["pid"]))
                c2 = parts_by_pk.get((k[0], k[1], legacy_rank[1]["pid"]))
                a = None if c1 is None else c1["parts"].get(comp)
                c = None if c2 is None else c2["parts"].get(comp)
                if a is not None:
                    wv.append(a)
                if c is not None:
                    rv.append(c)
            if wv and rv:
                print(f"     {comp:<7} #1 mean={mean(wv):.3f} (n={len(wv)})  "
                      f"#2 mean={mean(rv):.3f} (n={len(rv)})  delta={mean(wv)-mean(rv):+.3f}")
            else:
                print(f"     {comp:<7} not computable")

        # ---- F. raw detail on the confirmed event's bag ------------------
        if ev_bag is not None:
            kb = [k for k in keys if k[1] == ev_bag]
            print(f"\n  F. RAW DETAIL on the confirmed event's bag {ev_bag} "
                  f"({len(kb)} multi-person tick(s))")
            if not kb:
                print("     no competing person ever appeared for this bag")
            for k in kb:
                r0 = idx[0.0][k]
                legacy_rank = sorted(r0["cands"], key=lambda c: -c["legacy"])
                print(f"     frame={k[0]}:")
                for rank, c in enumerate(legacy_rank, 1):
                    pc = pose_by_pk.get((k[0], k[1], c["pid"]))
                    ps = "n/a" if pc is None or pc["pose"] is None else f"{pc['pose']:.4f}"
                    parts = "" if pc is None else "  " + str(
                        {kk: (None if vv is None else round(vv, 3))
                         for kk, vv in pc["parts"].items()})
                    w1 = sorted(idx[1.0][k]["cands"], key=lambda x: -x["score"])[0]["pid"]
                    star = "  <= blended winner at blend=1.0" if c["pid"] == w1 else ""
                    print(f"       #{rank} P{c['pid']:<4} legacy={c['legacy']:.4f}  "
                          f"pose={ps}{parts}{star}")

        all_out[stem] = {
            "confirmed_bag": ev_bag, "confirmed_person": ev_person,
            "blends": {str(b): per_blend[b] for b in BLENDS},
        }
        print()

    (OUT_DIR / "races.json").write_text(
        json.dumps(all_out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'races.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
