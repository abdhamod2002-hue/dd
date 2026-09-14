#!/usr/bin/env python3
"""
Run the live inference pipeline against Camo/USB, RTSP, or a video file.

Usage:
    python scripts/run_pipeline.py --source camo --device 0 --buffer 10
    python scripts/run_pipeline.py --source rtsp --url rtsp://user:pass@ip/stream
    python scripts/run_pipeline.py --source file --video path/clip.mp4
    python scripts/run_pipeline.py --source camo --post-backend http://localhost:8000/api/events

Section 6 live path:
  * Capture thread keeps ONLY the latest frame + a ring buffer of the last
    ``--buffer`` seconds (evidence pre/post clips).
  * Inference loop never shares VideoCapture; it polls ``get_latest_frame()``.
  * Empty-scene / hourly hygiene resets YOLO ByteTrack + FSM identity.
  * Best face crop is scored during BAG_CARRIED (sharpness + size + frontality).

File mode stays single-threaded for determinism (offline eval / CFR).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, Optional

# allow running from repo root
sys.path.insert(0, ".")

from inference.capture.camera_source import CameraSource, VideoFileSource
from inference.capture.live_camera_reader import LiveCameraReader
from inference.capture.live_hygiene import (
    LiveHygieneConfig,
    LiveSessionHygiene,
    reset_live_trackers,
)
from inference.detection.yolo_detector import YoloDetector
from inference.evidence.face_evidence import FaceEvidenceCapture
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pipeline import InferencePipeline, PipelineConfig
from inference.visualization import build_frame_analysis, render_analysis_frame
from littering_event_detector import EventState


def build_tracks_real(frame, tracked, movenet, tracker, frame_index, run_pose: bool = True, nov=None, object_identity=None):
    """
    REAL detection→tracking→pose→Track adapter.

    ``tracked`` is the output of ``YoloDetector.track()`` — a list of
    TrackedDetection objects carrying STABLE ByteTrack ids (persist=True
    keeps the tracker state between calls, so the same physical entity
    keeps the same id across frames).
    """
    tracker.update(tracked, frame_index)

    if nov is not None and getattr(nov.config, "enabled", False) and run_pose:
        _person_tracked = [t for t in tracked if t.is_person]
        _person_boxes = [tuple(t.bbox) for t in _person_tracked]
        _person_map = {int(t.track_id): tuple(t.bbox) for t in _person_tracked}
        _nov_out = nov.update(frame, frame_index, _person_boxes, _person_map)
        if _nov_out:
            from inference.detection.yolo_detector import boxes_duplicate
            _kept_obj_boxes = [tuple(t.bbox) for t in tracked
                               if not t.is_person
                               and str(getattr(t, "source", "yolo")) != "novelty"]
            _nov_filtered = [
                n for n in _nov_out
                if not any(boxes_duplicate(tuple(n.bbox), yb)
                           for yb in _kept_obj_boxes)
            ]
            tracked = list(tracked) + list(_nov_filtered)

    person_tracked = [t for t in tracked if t.is_person]
    person_bboxes = [t.bbox for t in person_tracked]
    pose_results = movenet.estimate(frame, person_bboxes) if (person_tracked and run_pose) else []

    kp_by_ns = {}
    for td, pr in zip(person_tracked, pose_results):
        ns_id = tracker.namespace(td.track_id, is_person=True)
        kp_by_ns[ns_id] = pr.keypoints if pr else None

    persons, objects = tracker.to_tracks(tracked, keypoints_by_person_ns=kp_by_ns)
    return persons, objects


def build_tracks(frame, detections, yolo, movenet, tracker_ns, namespace_offset):
    """DEPRECATED shim — do not use."""
    raise RuntimeError(
        "build_tracks() is deprecated — it assigned fake per-frame ids. "
        "Use build_tracks_real() with YoloDetector.track() output instead."
    )


def _update_best_face_during_carry(
    face_cap: FaceEvidenceCapture,
    best_faces: Dict[int, Dict[str, Any]],
    frame,
    persons,
    event_detector,
) -> None:
    """Score face crops while FSM is in BAG_CARRIED (clearest identity window)."""
    for mem in getattr(event_detector, "_pairs", {}).values():
        if mem.state != EventState.BAG_CARRIED:
            continue
        pid = int(mem.person_id)
        person = next((p for p in persons if int(p.track_id) == pid), None)
        if person is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in person.bbox]
        h, w = frame.shape[:2]
        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
        cx2, cy2 = min(w, int(x2)), min(h, int(y2))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        dets = face_cap._detect_faces_in_crop(crop)
        if not dets:
            continue
        det = max(dets, key=lambda d: d["h"])
        fx, fy, fw, fh = det["x"], det["y"], det["w"], det["h"]
        face = crop[fy:fy + fh, fx:fx + fw]
        if face.size == 0 or fh < face_cap.min_face_px:
            continue
        kps = getattr(person, "keypoints", None)
        score = face_cap.frame_quality_score(face, det, keypoints=kps)
        prev = best_faces.get(pid)
        if prev is None or score > float(prev["score"]):
            best_faces[pid] = {
                "score": float(score),
                "crop": face.copy(),
                "bbox": [cx1 + fx, cy1 + fy, cx1 + fx + fw, cy1 + fy + fh],
            }


def _maybe_save_best_face(best_faces: Dict[int, Dict[str, Any]], person_id: int, out_dir: str) -> Optional[str]:
    entry = best_faces.get(int(person_id))
    if not entry:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"face_best_person_{int(person_id)}.jpg")
    try:
        import cv2  # type: ignore
        ok = cv2.imwrite(path, entry["crop"])
        return path if ok else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Run AI Littering Detection pipeline")
    ap.add_argument("--source", choices=["camo", "file", "rtsp"], default="camo")
    ap.add_argument("--device", type=int, default=-1, help="OpenCV device index (-1 = auto-discover)")
    ap.add_argument("--url", type=str, default="", help="RTSP/HTTP URL for --source rtsp")
    ap.add_argument("--video", type=str, default="", help="path for --source file")
    ap.add_argument("--buffer", type=float, default=10.0, help="circular buffer window (s)")
    ap.add_argument("--analysis-fps", type=float, default=10.0)
    ap.add_argument("--pre", type=float, default=3.0)
    ap.add_argument("--post", type=float, default=3.0)
    ap.add_argument("--camera-id", type=str, default="cam-01")
    ap.add_argument("--post-backend", type=str, default="", help="FastAPI URL to POST events")
    ap.add_argument("--show", action="store_true", help="display live frame (dev)")
    ap.add_argument(
        "--legacy-sync-capture",
        action="store_true",
        help="Force single-thread CameraSource loop (debug only)",
    )
    args = ap.parse_args()

    cfg = PipelineConfig(
        buffer_seconds=args.buffer,
        analysis_fps=args.analysis_fps,
        pre_seconds=args.pre,
        post_seconds=args.post,
        camera_id=args.camera_id,
        post_backend_url=args.post_backend or None,
        auto_tune=True,
    )
    pipe = InferencePipeline(cfg)
    if args.source == "file" and args.video:
        try:
            pipe.event_detector.learning_video = os.path.basename(args.video)
        except Exception:
            pass

    live_reader: Optional[LiveCameraReader] = None
    src = None

    if args.source == "file":
        src = VideoFileSource(args.video)
        if not src.open():
            print(f"ERROR: cannot open source ({args.source})", file=sys.stderr)
            sys.exit(1)
    elif args.source == "rtsp":
        if not args.url:
            print("ERROR: --url is required for --source rtsp", file=sys.stderr)
            sys.exit(1)
        live_reader = LiveCameraReader(args.url, buffer_seconds=args.buffer, target_fps=30)
        live_reader.start()
        print(f"LiveCameraReader started for RTSP {args.url!r}")
    else:
        device_idx = args.device
        if device_idx < 0:
            try:
                from scripts.camera_discovery import discover_cameras
                cams = discover_cameras(max_idx=5, probe_frames=8)
                live = [c for c in cams if c.status == "LIVE"]
                if live:
                    device_idx = live[0].index
                    print(f"Auto-discovered LIVE camera at index {device_idx}")
                else:
                    print("ERROR: no LIVE camera found - connect iPhone via Camo and re-run", file=sys.stderr)
                    sys.exit(1)
            except Exception as e:
                print(f"ERROR: camera discovery failed: {e}", file=sys.stderr)
                sys.exit(1)
        if args.legacy_sync_capture:
            src = CameraSource(device_index=device_idx, target_fps=30)
            if not src.open():
                print(f"ERROR: cannot open source ({args.source})", file=sys.stderr)
                sys.exit(1)
        else:
            live_reader = LiveCameraReader(device_idx, buffer_seconds=args.buffer, target_fps=30)
            live_reader.start()
            print(f"LiveCameraReader started for USB device {device_idx}")

    detector = YoloDetector()
    detector.load()
    if detector.litter_classes:
        print(f"YOLO loaded. Litter classes: {detector.litter_classes}")
    else:
        print("YOLO loaded. Litter model (best.pt) NOT found.")
        print("  FALLBACK MODE: using COCO classes (bottle, cup, ...) - NOT the final litter model.")
        print("  For the final demo, place best.pt at inference/detection/weights/best.pt")
    movenet = MovenetPose()
    movenet.load()
    tracker_ns = BytetrackTracker()
    tracker_ns.load()

    from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
    nov_cfg = NoveltyConfig.from_yaml()
    nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
    if nov is not None:
        print(
            f"Novelty (scene-change) detector ENABLED: bg_frames={nov.config.bg_frames}, "
            f"stationary_frames={nov.config.stationary_frames}, person_mask_pad={nov.config.person_mask_pad}"
        )
    else:
        print("Novelty (scene-change) detector DISABLED (NOVELTY_ENABLED=false).")

    face_cap = FaceEvidenceCapture(backend="region")
    best_faces: Dict[int, Dict[str, Any]] = {}
    hygiene = LiveSessionHygiene(LiveHygieneConfig()) if live_reader is not None else None

    print(f"Running. buffer={args.buffer}s analysis_fps={args.analysis_fps} pre/post={args.pre}/{args.post}s")
    print("Press Ctrl+C to stop.")

    _latest_frame = [None]

    try:
        from backend.routers.stream import register_frame_source
        register_frame_source(lambda: _latest_frame[0])
        print("Frame source registered with backend stream router.")
    except Exception:
        print("Backend stream router not available — dashboard will show 'WAITING FOR CAMERA'.")

    try:
        import cv2  # type: ignore
        has_cv2 = True
    except Exception:
        has_cv2 = False

    frame_count = 0
    t0 = time.time()
    last_infer_time = 0.0
    last_seen_index = -1

    def _frame_iter():
        nonlocal last_seen_index
        if live_reader is not None:
            while True:
                pkt = live_reader.get_latest_frame()
                if pkt is None:
                    time.sleep(0.01)
                    continue
                if pkt.frame_index == last_seen_index:
                    time.sleep(0.005)
                    continue
                last_seen_index = pkt.frame_index
                yield type(pkt)(
                    frame=pkt.frame.copy(),
                    timestamp=pkt.timestamp,
                    frame_index=pkt.frame_index,
                    width=pkt.width,
                    height=pkt.height,
                )
        else:
            yield from src  # type: ignore

    try:
        for pkt in _frame_iter():
            t_start = time.time()

            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker_ns, frame_count, run_pose=run_pose, nov=nov,
            )
            events = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)

            if live_reader is not None and run_pose:
                _update_best_face_during_carry(
                    face_cap, best_faces, pkt.frame, persons, pipe.event_detector,
                )

            if hygiene is not None and hygiene.observe(person_count=len(persons), now=pkt.timestamp):
                print("[hygiene] resetting trackers / FSM (empty scene or hourly)")
                reset_live_trackers(
                    detector=detector,
                    tracker_ns=tracker_ns,
                    event_detector=pipe.event_detector,
                    novelty=nov,
                )
                best_faces.clear()
                hygiene.mark_reset(now=pkt.timestamp)

            current_event = None
            if events:
                ev = events[-1]
                current_event = {
                    "banner": "LITTERING EVENT CANDIDATE DETECTED",
                    "timestamp": ev.event_timestamp,
                    "frame": frame_count,
                    "person_track_id": ev.person_track_id,
                    "object_track_id": ev.object_track_id,
                    "confidence": ev.confidence,
                }
            try:
                analysis = build_frame_analysis(
                    persons,
                    objects,
                    pipe.event_detector,
                    tracker_ns,
                    pkt.timestamp,
                    frame_count,
                    source_fps=float(getattr(src, "fps", 0.0) or 0.0) if src else float(30.0),
                    analysis_fps=float(cfg.analysis_fps),
                    video_name=args.source,
                    event=current_event,
                )
                annotated = render_analysis_frame(pkt.frame, analysis, event=current_event)
            except Exception as e:
                print(f"[visualization warning] {e}", file=sys.stderr)
                annotated = pkt.frame
            _latest_frame[0] = annotated

            last_infer_time = (time.time() - t_start) * 1000.0

            for ev in events:
                face_path = _maybe_save_best_face(
                    best_faces,
                    int(ev.person_track_id),
                    out_dir=os.path.join("evidence_store", "live_faces", args.camera_id),
                )
                extra = f" face={face_path}" if face_path else ""
                print(
                    f"[{ev.event_timestamp:.2f}] LITTERING CONFIRMED — {ev.object_type} "
                    f"person={ev.person_track_id} object={ev.object_track_id} "
                    f"conf={ev.confidence:.2f}{extra}"
                )
            if args.show and has_cv2:
                cv2.imshow("littering", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_count += 1

            if frame_count % 5 == 0:
                h_img, w_img = pkt.frame.shape[:2]
                entities = []
                pair_by_person = {int(k[0]): mem for k, mem in pipe.event_detector._pairs.items()}
                pair_by_bag = {int(k[1]): mem for k, mem in pipe.event_detector._pairs.items()}
                for p in persons:
                    x1, y1, x2, y2 = p.bbox
                    mem = pair_by_person.get(int(p.track_id))
                    entities.append({
                        "trackId": p.track_id,
                        "label": "Person",
                        "bbox": {"x": x1 / w_img, "y": y1 / h_img, "w": (x2 - x1) / w_img, "h": (y2 - y1) / h_img},
                        "confidence": float(getattr(p, "confidence", 0.0) or 0.0),
                        "isPerson": True,
                        "source": str(getattr(p, "source", "yolo") or "yolo"),
                        "state": getattr(mem.state, "value", None) if mem is not None else None,
                        "associatedObjectId": int(mem.bag_id) if mem is not None else None,
                    })
                for o in objects:
                    x1, y1, x2, y2 = o.bbox
                    mem = pair_by_bag.get(int(o.track_id))
                    entities.append({
                        "trackId": o.track_id,
                        "label": o.class_name,
                        "bbox": {"x": x1 / w_img, "y": y1 / h_img, "w": (x2 - x1) / w_img, "h": (y2 - y1) / h_img},
                        "confidence": float(getattr(o, "confidence", 0.0) or 0.0),
                        "isPerson": False,
                        "source": str(getattr(o, "source", "yolo") or "yolo"),
                        "state": getattr(mem.state, "value", None) if mem is not None else None,
                        "associatedPersonId": int(mem.person_id) if mem is not None else None,
                    })

                fps = frame_count / max(1e-6, time.time() - t0)
                pipe.push_status(
                    pkt.timestamp,
                    capture_fps=fps,
                    analysis_fps_actual=min(fps, args.analysis_fps),
                    inference_latency_ms=last_infer_time,
                    source_type=args.source,
                    live_entities=entities,
                )

            if frame_count % 100 == 0:
                st = pipe.stats()
                fps = frame_count / max(1e-6, time.time() - t0)
                conn = ""
                if live_reader is not None:
                    conn = (
                        f" connected={live_reader.is_connected}"
                        f" reconnects={live_reader.reconnect_count}"
                        f" ring={len(live_reader.ring_buffer)}"
                    )
                print(f"[stats] {st} capture_fps={fps:.1f} latency={last_infer_time:.1f}ms{conn}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if live_reader is not None:
            live_reader.stop()
        if src is not None:
            src.release()
        if args.show and has_cv2:
            cv2.destroyAllWindows()
        try:
            pipe.finalize()
        except Exception as e:
            print(f"finalize warning: {e}")
        print(f"Total confirmed events: {len(pipe.events)}")
        print(f"Event detector summary: {pipe.event_detector.summary()}")


if __name__ == "__main__":
    main()
