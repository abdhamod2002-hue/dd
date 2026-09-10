#!/usr/bin/env python3
"""Compact Layer 1 + Addendum report (A-K) from layer1_runs/raw/*.json."""
from __future__ import annotations
import glob, json, os
from collections import Counter

OUT = r"D:\HO\layer1_runs"
RAW = os.path.join(OUT, "raw")


def _load(n):
    p = os.path.join(OUT, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def tb(rows):
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    fm = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"
    s = "|" + "|".join("-" * (x + 2) for x in w) + "|"
    return "\n".join([fm(rows[0]), s] + [fm(r) for r in rows[1:]])


def _ok():
    out = []
    for p in sorted(glob.glob(os.path.join(RAW, "*.json"))):
        if "_frames" not in p:
            out.append(json.load(open(p, encoding="utf-8")))
    return [r for r in out if "error" not in r]


def build(ok):
    # Compute aggregates directly from the current per-video raw records, so we
    # never present stale summary/coverage files from an earlier partial run.
    summary = {
        "total_object_tracks": sum(v["object_track_quality"]["n_tracks"] for v in ok),
        "total_object_id_switches": sum(v["object_track_quality"]["id_switches"] for v in ok),
        "avg_object_track_duration_s": round(
            sum(v["object_track_quality"]["durations_s"]["avg"] for v in ok) / max(1, len(ok)), 2),
    }
    obj_src_agg = Counter()
    for v in ok:
        for s, c in v["sources"]["object_sources_seen"].items():
            obj_src_agg[s] += c
    summary["object_sources_aggregate"] = dict(obj_src_agg)
    cov = []
    for v in sorted(ok, key=lambda x: x["video"]):
        cov.append({
            "video": v["video"], "resolution": v.get("resolution"), "fps": v.get("fps"),
            "duration_s": v.get("duration_s"),
            "distinct_persons": v["sources"]["distinct_person_tracks"],
            "distinct_objects": v["sources"]["distinct_object_tracks"],
            "global_motion_mean": v["diversity"].get("global_motion_mean"),
            "camera_viewpoint": v["coverage"].get("camera_viewpoint", "STATIC"),
        })
    A = []; ad = A.append
    ad("# Layer 1 + Addendum — Detection/Tracking Audit (REAL D:\\22)")
    ad("")
    ad("**Scope:** detection+tracking only. Path: `VideoFileSource -> YoloDetector.track("
       "persist=True) -> build_tracks_real (MoveNet@analysis_fps + Novelty@analysis_fps)`.")
    ad(f"Videos measured so far: {len(ok)} (see 'Coverage' for which completed; the rest of "
       f"D:\\22 may still be processing — see Layer 1 blockers/limitations).")
    ad("")
    ad("## A. Detector source breakdown")
    r = [["Video", "PerRec", "ObjRec", "obj_sources", "#Per", "#Obj", "OidSw", "PidSw"]]
    for v in sorted(ok, key=lambda x: x["video"]):
        s = v["sources"]
        r.append([v["video"], str(s["person_detection_records"]), str(s["object_detection_records"]),
                  ", ".join(f"{k}={c}" for k, c in sorted(s["object_sources_seen"].items())) or "-",
                  str(s["distinct_person_tracks"]), str(s["distinct_object_tracks"]),
                  str(v["object_track_quality"]["id_switches"]), str(v["person_track_quality"]["id_switches"])])
    ad(tb(r)); ad("")
    if summary.get("object_sources_aggregate"):
        ad("Aggregate obj records by source: " +
           ", ".join(f"{k}={c}" for k, c in sorted(summary["object_sources_aggregate"].items())))
    ad("")
    ad("## B. Object track quality")
    r = [["Video", "n", "avgS", "medS", "maxS", "<1s", "raw/obj(med)", "reassoc", "idSw"]]
    for v in sorted(ok, key=lambda x: x["video"]):
        q = v["object_track_quality"]
        r.append([v["video"], str(q["n_tracks"]), str(q["durations_s"]["avg"]),
                  str(q["durations_s"]["median"]), str(q["durations_s"]["max"]),
                  str(q["short_tracks_lt_1s"]), str(q["raw_ids_per_physical_object"]["median"]),
                  str(q["reassociated_tracks"]), str(q["id_switches"])])
    ad(tb(r))
    ad(f"\nAggregate: total_obj_tracks={summary.get('total_object_tracks')}, "
       f"obj_id_switches={summary.get('total_object_id_switches')}, "
       f"avg_obj_dur={summary.get('avg_object_track_duration_s')}s")
    ad("")
    ad("## C. Person track quality")
    r = [["Video", "n", "avgS", "medS", "maxS", "<1s", "raw/per(med)", "reassoc", "idSw"]]
    for v in sorted(ok, key=lambda x: x["video"]):
        q = v["person_track_quality"]
        r.append([v["video"], str(q["n_tracks"]), str(q["durations_s"]["avg"]),
                  str(q["durations_s"]["median"]), str(q["durations_s"]["max"]),
                  str(q["short_tracks_lt_1s"]), str(q["raw_ids_per_person"]["median"]),
                  str(q["reassociated_tracks"]), str(q["id_switches"])])
    ad(tb(r)); ad("")
    ad("## D. Multi-person test")
    any_mp = False
    for v in sorted(ok, key=lambda x: x["video"]):
        pts = [p for p in v["persons"] if p["n_frames"] >= 2]
        if len(pts) >= 2:
            any_mp = True
            ad(f"### {v['video']} — {len(pts)} person tracks")
            r = [["P", "first(f,ts)", "last(f,ts)", "dur_s", "n_f", "cx", "cy", "src"]]
            for p in pts:
                r.append([str(p["id"]), f"{p['first_frame']},{p['first_ts']}",
                          f"{p['last_frame']},{p['last_ts']}", str(p["duration_s"]),
                          str(p["n_frames"]), str(p["mean_cx"]), str(p["mean_cy"]), p["source"]])
            ad(tb(r)); ad("")
    if not any_mp:
        ad("No measured D:\\22 video had >=2 person tracks >=2 frames. Multi-person rule is "
           "covered by per-video person tables (stable ids) + `test_acceptance_multiperson.py`. "
           "A dedicated 3-actor real video is required to run CASE-1 at Layer 2 (see blockers).")
        ad("")
    ad("## E. Multi-object test")
    for v in sorted(ok, key=lambda x: x["video"]):
        ots = [o for o in v["objects"] if o["n_frames"] >= 2]
        if len(ots) >= 2:
            ad(f"### {v['video']} — {len(ots)} object tracks (top40)")
            r = [["O", "first(f,ts)", "last(f,ts)", "dur_s", "n_f", "src", "class"]]
            for o in sorted(ots, key=lambda x: -x["n_frames"])[:40]:
                r.append([str(o["id"]), f"{o['first_frame']},{o['first_ts']}",
                          f"{o['last_frame']},{o['last_ts']}", str(o["duration_s"]),
                          str(o["n_frames"]), o["source"], o["class"]])
            ad(tb(r)); ad("")
    ad("No object is assigned to any event at Layer 1.")
    ad("")
    ad("## F. Detection diversity")
    for v in sorted(ok, key=lambda x: x["video"]):
        d = v["diversity"]
        ad(f"- **{v['video']}**: classes=" +
           (", ".join(f"{k}:{c}" for k, c in d["class_distribution"].items()) or "-") +
           f"; objRec={d['object_detection_records']}; hue={d.get('hue_mean_of_objects')}; "
           f"bright={d['brightness_mean']}+-{d['brightness_stdev']}; gmotion={d.get('global_motion_mean')}px")
    ad("")
    ad("Class-agnostic `detected_object`/`UNKNOWN_OBJECT` kept distinct from HSV/YOLO; NOT WASTE.")
    ad("")
    ad("## G. Real dataset coverage")
    r = [["Video", "res", "fps", "dur_s", "#Per", "#Obj", "gmotion", "camera"]]
    for c in sorted(cov, key=lambda x: x["video"]):
        r.append([c["video"], "x".join(str(v) for v in c.get("resolution") or ()),
                  str(c["fps"]), str(c["duration_s"]), str(c["distinct_persons"]),
                  str(c["distinct_objects"]), str(c["global_motion_mean"]),
                  c.get("camera_viewpoint", "STATIC")])
    ad(tb(r)); ad("")
    ad("## H. Potential false-detection candidates (NOT confirmed)")
    for v in sorted(ok, key=lambda x: x["video"]):
        fc = v["false_candidates"]
        ad(f"**{v['video']}** — {fc['n_candidates']} potential ({fc['status']}).")
        for e in fc["examples"][:10]:
            ad(f"  - id={e['id']} class={e['class']} src={e['source']} dur={e['duration_s']}s "
               f"-> {', '.join(e['reason'])}")
    ad("")
    ad("## I. Detector failure patterns")
    pat = Counter(); details = []
    for v in sorted(ok, key=lambda x: x["video"]):
        for f in v.get("failures", []):
            pat[f["pattern"]] += 1; details.append((v["video"], f))
    ad(tb([["pattern", "count"]] + [[k, str(c)] for k, c in pat.most_common()])); ad("")
    shown = 0
    for vid, f in details:
        if shown >= 40: break
        ad(f"- {vid}: {f['pattern']} ({ {k: v for k, v in f.items() if k != 'pattern'} })"); shown += 1
    if not details:
        ad("No automated failure pattern above threshold.")
    ad(f"\n**Dominant fragility:** HSV colour fallback re-births ids during carry/release "
       f"(obj_id_switches={summary.get('total_object_id_switches')}). Layer-2 stable object_uid + "
       f"EVENT ACTOR/OBJECT OWNERSHIP must guard; a box alone must never create an event.")
    ad("")
    ad("## J. Layer 1 acceptance criteria")
    ad("1 person tracks stable (C) · 2 object tracks survive window (B) · 3 multi-person distinct (D)"
       " · 4 multi-object distinct (E) · 5 ID switches measured (B/C/E/I) · 6 sources separated (A)"
       " · 7 novelty NOT auto-waste (class-gated) · 8 frame-level info -> per-video "
       "`<stem>_frames.jsonl`")
    ad("")
    vcnt = Counter(v["verdict"]["overall"] for v in ok)
    ad("## K. Required output + verdict")
    ad("Per-video verdicts: " + str(dict(vcnt)))
    allp = all(v["verdict"]["overall"] == "PASS" for v in ok)
    final = "PASS" if (allp and len(ok) >= 3) else ("PARTIAL" if len(ok) >= 1 else "FAIL")
    sym = {"PASS": "✅ PASS", "PARTIAL": "🟡 PARTIAL", "FAIL": "❌ FAIL"}[final]
    ad(f"**FINAL Layer 1 VERDICT: {sym}**")
    ad("")
    ad("> STOP — do not proceed to Layer 2 until approved.")
    return "\n".join(A)


if __name__ == "__main__":
    out = build(_ok())
    t = os.path.join(OUT, "LAYER1_REPORT.md")
    open(t, "w", encoding="utf-8").write(out)
    print(f"wrote {t} ({len(out)} chars)")