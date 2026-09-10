"""
Part-1 verification: face capture wired to the REAL confirmed-event moment.

This script replicates ONLY the evidence-production half of the production
video-analysis job (backend/routers/analysis.py::_run_video_analysis_job),
WITHOUT the DB/HTTP layers, and WITHOUT any synthetic event:

  1. run the REAL InferencePipeline (YOLO + ByteTrack + MoveNet + the actual
     LitteringEventDetector) on each D:\\W video,
  2. collect the REAL confirmed LitteringEvent(s) the detector emits
     (littering_event_detector._evaluate_confirmation -> VIOLATION_CONFIRMED),
  3. for each confirmed event call the EXACT production call
     write_event_evidence_package(...) which internally fires face capture,
  4. verify face_evidence.jpg was written into the real evidence package and
     recorded in metadata.json.

Difference vs the earlier test:
  * BEFORE: scripts.test_face_evidence passed a hand-built "representative"
    person-track span to write_event_evidence_package -> face capture ran on
    EVERY video regardless of whether a violation was actually confirmed.
  * AFTER:  write_event_evidence_package is reached ONLY for events the
    LitteringEventDetector genuinely CONFIRMED (pipe.events). If the detector
    does not confirm, face capture does not fire at all (correct behaviour).

Output: D:\\HO\\.audit\\face_on_confirmed\\<VID>\\*  (real evidence package)
        D:\\HO\\.audit\\face_on_confirmed_summary.json
        D:\\HO\\.audit\\face_on_confirmed_report.md
"""
import sys, os, json, time, gc
sys.path.insert(0, r"D:\HO")

import cv2  # type: ignore
from inference.capture.camera_source import VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pipeline import InferencePipeline, PipelineConfig
from inference.visualization.evidence_package import write_event_evidence_package
from scripts.run_pipeline import build_tracks_real

VIDS = [r"D:\W\IMG_5117.MOV", r"D:\W\IMG_5118.MOV",
        r"D:\W\IMG_5119.MOV", r"D:\W\IMG_5120.MOV"]
OUT_BASE = r"D:\HO\.audit\face_on_confirmed"


def compact(dev):
    d = dev.to_dict()
    return {k: d.get(k) for k in (
        "event_id", "person_track_id", "bag_track_id", "state", "confirmed",
        "reason", "confidence", "evidence", "frames", "timestamps",
        "bag_class", "fallback_used", "yolo_reconfirmed", "detector_source",
        "other_person_closer")}


def run_video(vpath):
    vname = os.path.basename(vpath)
    print(f"\n=== {vname} ===", flush=True)
    source = VideoFileSource(vpath)
    if not source.open():
        return {"video": vname, "error": "cannot open", "confirmed_events": 0}
    detector = YoloDetector(); detector.load()
    movenet = MovenetPose(); movenet.load()
    tracker = BytetrackTracker(); tracker.load()

    cfg = PipelineConfig(
        buffer_seconds=8.0, analysis_fps=8.0, camera_id="verify",
        post_backend_url=None, pre_seconds=3.0, post_seconds=3.0,
    )
    pipe = InferencePipeline(cfg)
    pipe.event_detector.reset()

    target = os.path.join(OUT_BASE, vname)
    os.makedirs(target, exist_ok=True)
    analyzed_path = os.path.join(target, "analyzed.mp4")
    writer = None
    writer_fps = float(source.fps or 30.0)

    frame_records = []
    first_ts = None
    frame_idx = 0
    last_ts = 0.0
    confirmed_devs = []

    for pkt in source:
        if first_ts is None:
            first_ts = float(pkt.timestamp)
        ts = float(pkt.timestamp)
        tracked = detector.track(pkt.frame, persist=True)
        run_pose = pipe.should_analyze(ts)
        persons, objects = build_tracks_real(pkt.frame, tracked, movenet, tracker, frame_idx, run_pose=run_pose)
        pipe.process_frame(pkt.frame, ts, persons, objects)

        # record frame for the evidence package (same keys write_event_evidence_package reads)
        frame_records.append({
            "frame_number": frame_idx,
            "timestamp": ts,
            "persons": [{"track_id": int(p.track_id), "bbox": [float(x) for x in p.bbox],
                         "class_name": getattr(p, "class_name", "person")} for p in persons],
            "objects": [{"track_id": int(o.track_id), "bbox": [float(x) for x in o.bbox],
                         "class_name": getattr(o, "class_name", "object")} for o in objects],
        })

        if writer is None:
            h, w = pkt.frame.shape[:2]
            writer = cv2.VideoWriter(analyzed_path, cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (w, h))
        writer.write(pkt.frame)

        last_ts = ts
        frame_idx += 1
        if frame_idx % 50 == 0:
            gc.collect()

    source.release()
    if writer:
        writer.release()

    # Flush any open detector pairs -> may emit more confirmed events.
    for ev in pipe.finalize(last_ts):
        pass

    # Collect the REAL confirmed events only.
    for ev in pipe.events:
        dev = next((d for d in pipe.event_detector.confirmed_events if d.event_id == ev.event_id), None)
        if dev is not None:
            confirmed_devs.append(dev)

    print(f"  confirmed events from detector: {len(confirmed_devs)}", flush=True)

    packages = []
    for dev in confirmed_devs:
        event_dict = compact(dev)
        try:
            pkg = write_event_evidence_package(
                analyzed_video_path=analyzed_path,
                target_dir=target,
                event=event_dict,
                frame_records=frame_records,
                job_id=1,
                original_filename=vname,
                source_fps=writer_fps,
                pre_seconds=3.0,
                post_seconds=3.0,
            )
        except Exception as e:
            print(f"  package error for event {dev.event_id}: {e}", flush=True)
            continue
        meta_path = os.path.join(target, "metadata.json")
        fe = None
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, encoding="utf-8"))
            fe = meta.get("face_evidence", {})
        face_path = (fe or {}).get("face_evidence_path")
        face_ok = bool(face_path and os.path.exists(face_path) and os.path.getsize(face_path) > 0)
        packages.append({
            "event_id": dev.event_id,
            "person_track_id": dev.person_track_id,
            "bag_track_id": dev.bag_track_id,
            "reason": dev.reason,
            "confidence": dev.confidence,
            "files": {k: pkg.get(k) for k in ("snapshot", "person", "waste", "clip", "metadata")},
            "face_evidence_path": face_path if face_ok else None,
            "face_captured": face_ok,
            "face_status": (fe or {}).get("status"),
            "face_size_px": (fe or {}).get("face_size_px"),
            "face_detection_confidence": (fe or {}).get("face_detection_confidence"),
            "frame_number": (fe or {}).get("frame_number"),
        })

    det_summary = pipe.event_detector.summary()
    return {
        "video": vname,
        "confirmed_events": len(confirmed_devs),
        "detector_summary": det_summary,
        "packages": packages,
    }


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    results = []
    for vp in VIDS:
        results.append(run_video(vp))
    json.dump(results, open(os.path.join(OUT_BASE, "face_on_confirmed_summary.json"), "w"),
              indent=2, ensure_ascii=False, default=str)

    # markdown report
    lines = ["# Face capture tied to the REAL confirmed-event moment\n"]
    lines.append("> Verified on the 4 previously-CONFIRMED D:\\W videos by running the REAL "
                 "LitteringEventDetector and calling `write_event_evidence_package` ONLY for "
                 "events it actually confirmed.\n")
    lines.append("| Video | Confirmed events | face_evidence.jpg | size(px) | conf | frame |")
    lines.append("|-------|------------------|-------------------|----------|------|-------|")
    for r in results:
        v = r["video"]
        if r.get("confirmed_events", 0) == 0:
            lines.append(f"| {v} | 0 (detector did NOT confirm) | — (correctly not captured) | — | — | — |")
            continue
        for p in r["packages"]:
            fe = "✅ " + os.path.basename(p["face_evidence_path"]) if p["face_captured"] else "❌ not captured"
            lines.append(f"| {v} | {r['confirmed_events']} | {fe} | {p['face_size_px']} | {p['face_detection_confidence']} | {p['frame_number']} |")
    open(os.path.join(OUT_BASE, "face_on_confirmed_report.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
