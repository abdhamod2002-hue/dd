#!/usr/bin/env python3
"""PHASE 1 — real-video stable-identity + temporal-event audit.

Runs the REAL production path (YoloDetector.track -> ByteTrack -> MoveNet ->
novelty -> InferencePipeline -> LitteringEventDetector) over real D:\\22
videos and records, per analysis tick:

  * raw person track id  -> STABLE person uid   (PersonIdentityManager)
  * raw object track id  -> STABLE object uid   (ObjectIdentityManager)
  * per-pair temporal state (near/carried/released/ground/departure/regrab)
  * emitted events (confirmed + rejected) with full identity + evidence

Identity evidence aggregated at the end:
  * person uid -> set of raw ids  (>1 raw id => a real P2->P7 style switch
    was resolved while preserving identity)
  * same-frame uid collisions (must always be ZERO)
  * object uid -> set of raw ids (churn absorption evidence)

NO video-specific logic: the same loop runs for every video.

Usage:
    python scripts/phase2_identity_audit.py --videos D:/22/IMG_5305.MOV ...
    python scripts/phase2_identity_audit.py --videos D:/22/IMG_5305.MOV --max-seconds 90
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, ".")

from inference.capture.camera_source import VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pipeline import InferencePipeline, PipelineConfig


def audit_video(video_path: str, out_dir: str, max_seconds: float = 0.0) -> dict:
    name = os.path.splitext(os.path.basename(video_path))[0]
    src = VideoFileSource(video_path)
    if not src.open():
        return {"video": name, "error": "cannot open source"}

    detector = YoloDetector()
    detector.load()
    movenet = MovenetPose()
    movenet.load()
    tracker = BytetrackTracker()
    tracker.load()

    from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
    nov_cfg = NoveltyConfig.from_yaml()
    nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None

    pipe = InferencePipeline(PipelineConfig(auto_tune=False))

    ticks = []          # one record per ANALYSIS tick (heavy path)
    events = []         # emitted detector events (confirmed + rejected)
    t0 = time.time()
    frame_count = 0
    last_analysis_frame = -1

    for pkt in src:
        if max_seconds and (pkt.timestamp - getattr(audit_video, "_start", 0.0) > max_seconds):
            break
        tracked = detector.track(pkt.frame, persist=True)
        run_pose = pipe.should_analyze(pkt.timestamp)
        if not run_pose:
            frame_count += 1
            continue

        # novelty (additive proposal source) on the analysis tick
        if nov is not None:
            _person_tracked = [t for t in tracked if t.is_person]
            _person_boxes = [tuple(t.bbox) for t in _person_tracked]
            _person_map = {int(t.track_id): tuple(t.bbox) for t in _person_tracked}
            _nov_out = nov.update(pkt.frame, frame_count, _person_boxes, _person_map)
            if _nov_out:
                tracked = list(tracked) + list(_nov_out)

        person_tracked = [t for t in tracked if t.is_person]
        person_bboxes = [t.bbox for t in person_tracked]
        pose_results = movenet.estimate(pkt.frame, person_bboxes) if person_tracked else []
        kp_by_ns = {}
        for td, pr in zip(person_tracked, pose_results):
            kp_by_ns[tracker.namespace(td.track_id, is_person=True)] = pr.keypoints if pr else None
        persons, objects = tracker.to_tracks(tracked, keypoints_by_person_ns=kp_by_ns)

        tick_record = {
            "tick_frame": frame_count,
            "timestamp": round(float(pkt.timestamp), 3),
            "persons": [
                {"raw_id": int(p.track_id),
                 "uid": pipe.event_detector._person_uid_of(int(p.track_id))}
                for p in persons
            ],
            "objects": [
                {"raw_id": int(o.track_id), "source": str(getattr(o, "source", "yolo")),
                 "class": str(o.class_name)}
                for o in objects
            ],
            "pairs": [],
        }
        events.extend(pipe.event_detector.update(
            [pipe._detector_person(p) for p in persons],
            [pipe._detector_bag(o) for o in objects],
            float(pkt.timestamp),
            frame_count,
        ))
        # NOTE: we call the detector directly for clean identity telemetry;
        # evidence/backend layers are Phase 3+ and intentionally untouched.
        tick_record["person_uid_map"] = dict(pipe.event_detector.last_person_uid_map)
        tick_record["object_uid_map"] = {
            str(k): v for k, v in pipe.event_detector.last_object_uid_map.items()
        }
        for key, mem in pipe.event_detector._pairs.items():
            tick_record["pairs"].append({
                "person_uid": key[0],
                "object_uid": key[1],
                "raw_person_id": mem.person_id,
                "raw_bag_id": mem.bag_id,
                "state": mem.state.value,
                "carried_frames": mem.carried_frames,
                "stationary_frames": mem.stationary_frames,
                "departed_frames": mem.departed_frames,
                "abandonment_frames": mem.abandonment_frames,
                "reclaimed": mem.reclaimed,
                "assoc_score": round(mem.mean_association_score(), 4),
                "ground_evidence_frames": mem.ground_evidence_frames,
            })
        # process_frame for evidence/pipeline parity is skipped: Phase 1 is
        # identity + decision only. But rejected/confirmed lists are refreshed.
        pipe.rejected_events = pipe.event_detector.rejected_events
        ticks.append(tick_record)
        last_analysis_frame = frame_count
        frame_count += 1
        if max_seconds and (time.time() - t0) > max_seconds * 6:
            print(f"[{name}] wall-clock guard hit, stopping early", file=sys.stderr)
            break

    events.extend(pipe.event_detector.finalize())

    # ---------------- identity aggregation ----------------
    person_uid_raw: dict = {}
    collisions = 0
    for rec in ticks:
        uids_this_tick = [p["uid"] for p in rec["persons"]]
        if len(uids_this_tick) != len(set(uids_this_tick)):
            collisions += 1
        for p in rec["persons"]:
            person_uid_raw.setdefault(p["uid"], set()).add(p["raw_id"])
    object_uid_raw: dict = {}
    for rec in ticks:
        for raw, uid in rec["object_uid_map"].items():
            if uid is not None:
                object_uid_raw.setdefault(uid, set()).add(int(raw))

    person_switches = {
        str(uid): sorted(raws) for uid, raws in person_uid_raw.items() if len(raws) > 1
    }
    object_churn = {
        str(uid): sorted(raws) for uid, raws in object_uid_raw.items() if len(raws) > 1
    }
    confirmed = [e for e in events if e.confirmed]
    rejected = [e for e in events if not e.confirmed]

    result = {
        "video": name,
        "path": video_path,
        "duration_s": round(time.time() - t0, 1),
        "analysis_ticks": len(ticks),
        "last_analysis_frame": last_analysis_frame,
        "identity": {
            "distinct_person_uids": len(person_uid_raw),
            "person_uid_multi_raw_ids": person_switches,
            "same_frame_uid_collisions": collisions,
            "distinct_object_uids": len(object_uid_raw),
            "object_uid_multi_raw_ids": {k: v[:20] for k, v in object_churn.items()},
            "object_uid_multi_raw_count": len(object_churn),
        },
        "events": {
            "confirmed": [e.to_dict() for e in confirmed],
            "rejected_summary": [
                {"reason": e.reason, "confidence": e.confidence,
                 "person_track_id": e.person_track_id,
                 "event_actor_person_uid": e.event_actor_person_uid,
                 "event_object_uid": e.event_object_uid,
                 "location_status": e.details.get("location_status"),
                 "frames": e.frames}
                for e in rejected
            ],
        },
        "detector_summary": pipe.event_detector.summary(),
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"result": result, "ticks": ticks}, f, indent=1, default=str)
    return result


def main():
    ap = argparse.ArgumentParser(description="Phase-1 real-video identity audit")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join("phase2_runs", "identity_audit"))
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="limit per-video source seconds (0 = full video)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "SUMMARY.json")
    summaries = []
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summaries = json.load(f)
    done = {s.get("video") for s in summaries}

    for video in args.videos:
        name = os.path.splitext(os.path.basename(video))[0]
        if name in done:
            print(f"[skip] {name} already audited")
            continue
        print(f"=== auditing {video} ===", flush=True)
        try:
            res = audit_video(video, args.out, args.max_seconds)
        except Exception as e:
            res = {"video": name, "error": repr(e)}
        summaries = [s for s in summaries if s.get("video") != name]
        summaries.append(res)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=1, default=str)
        ident = res.get("identity", {})
        ev = res.get("events", {})
        print(json.dumps({
            "video": res.get("video"),
            "error": res.get("error"),
            "ticks": res.get("analysis_ticks"),
            "distinct_person_uids": ident.get("distinct_person_uids"),
            "person_uid_multi_raw_ids": ident.get("person_uid_multi_raw_ids"),
            "same_frame_uid_collisions": ident.get("same_frame_uid_collisions"),
            "distinct_object_uids": ident.get("distinct_object_uids"),
            "object_uid_multi_raw_count": ident.get("object_uid_multi_raw_count"),
            "confirmed_events": len(ev.get("confirmed", [])),
            "rejected": len(ev.get("rejected_summary", [])),
            "detector_summary": res.get("detector_summary"),
        }, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
