#!/usr/bin/env python3
"""
Run the live inference pipeline against the iPhone camera (Camo/Iriun).

Usage:
    python scripts/run_pipeline.py --source camo --device 0 --buffer 6
    python scripts/run_pipeline.py --source file --video path/clip.mp4
    python scripts/run_pipeline.py --source camo --post-backend http://localhost:8000/api/events

This script is the live entry point. It wires the real CV components
(YOLO, ByteTrack, MoveNet) which require the heavy deps installed on
the laptop — not in the sandbox. The core logic (buffer, association,
FSM, voting, evidence) is already unit-tested without these.

Note on FPS: capture runs at full camera FPS; the heavy analysis runs
at --analysis-fps (default 10). Evidence clips are assembled from the
full-FPS buffer, so they stay smooth even when analysis is throttled.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# allow running from repo root
sys.path.insert(0, ".")

from inference.capture.camera_source import CameraSource, VideoFileSource
from inference.detection.yolo_detector import YoloDetector
from inference.pose.movenet_pose import MovenetPose
from inference.tracking.bytetrack_tracker import BytetrackTracker
from inference.pipeline import InferencePipeline, PipelineConfig
from inference.visualization import build_frame_analysis, render_analysis_frame


def build_tracks_real(frame, tracked, movenet, tracker, frame_index, run_pose: bool = True, nov=None, object_identity=None):
    """
    REAL detection→tracking→pose→Track adapter.

    ``tracked`` is the output of ``YoloDetector.track()`` — a list of
    TrackedDetection objects carrying STABLE ByteTrack ids (persist=True
    keeps the tracker state between calls, so the same physical entity
    keeps the same id across frames).

    Steps:
      1. record tracked detections into the TrackStore (history),
      2. run MoveNet pose ONLY on person crops when ``run_pose`` is true
         (the caller throttles this to the analysis tick — biggest CPU save),
      3. convert to namespaced Track objects for the association engine.

    This replaces the old build_tracks() which assigned fake per-frame
    ids (i+1) and broke the entire temporal layer.
    """
    # 1) record history
    tracker.update(tracked, frame_index)

    # 1b) CLASS-AGNOSTIC NOVELTY (scene-change) DETECTION — additive, separate
    # source. Runs only when a live, enabled NoveltyDetector is supplied. Its
    # TrackedDetection(source="novelty") outputs are appended to `tracked` so
    # tracker.to_tracks() namespaces them into the SAME `objects` list the
    # association engine + FSM already consume. The HSV production path is
    # completely untouched. A novelty object is NOT a litter-candidate class,
    # so it surfaces as a detected object without spawning false events.
    # See NOVELTY_DETECTION_REPORT.md.
    if nov is not None and getattr(nov.config, "enabled", False) and run_pose:
        # Gate to the ANALYSIS tick (run_pose), NOT every source frame. Running
        # novelty on all ~60 source fps was a 3-4x slowdown vs the HSV-only path
        # AND diverged from the tuning validated in the harness (which calls
        # update() once per analysis tick). The detector's warmup uses an internal
        # buffer counter, so this cadence change is safe and restores the intended
        # stationary_frames / decay_frames semantics (measured in analysis ticks).
        _person_tracked = [t for t in tracked if t.is_person]
        _person_boxes = [tuple(t.bbox) for t in _person_tracked]
        _person_map = {int(t.track_id): tuple(t.bbox) for t in _person_tracked}
        _nov_out = nov.update(frame, frame_index, _person_boxes, _person_map)
        if _nov_out:
            # Phase 3: a novelty proposal describing the SAME physical object
            # as a same-frame detector box is redundant — the semantic
            # detection stays authoritative. Suppressed here at append time
            # (novelty internals untouched); same symmetric predicate as the
            # detector dedup. Person boxes are excluded: novelty already
            # erases person regions from its change mask.
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

    # 2) person crops for pose (lazy: only persons, only analysis ticks)
    person_tracked = [t for t in tracked if t.is_person]
    person_bboxes = [t.bbox for t in person_tracked]
    pose_results = movenet.estimate(frame, person_bboxes) if (person_tracked and run_pose) else []

    # map namespaced person id → keypoints
    kp_by_ns = {}
    for td, pr in zip(person_tracked, pose_results):
        ns_id = tracker.namespace(td.track_id, is_person=True)
        kp_by_ns[ns_id] = pr.keypoints if pr else None

    # 3) convert to Track objects (namespaced ids, stable across frames)
    persons, objects = tracker.to_tracks(tracked, keypoints_by_person_ns=kp_by_ns)
    # 3b) Stable object identity is assigned by the event detector's pair
    # memory (which already persists a bag across tracker id churn via its
    # rebind logic) and stamped onto the object Tracks in process_frame. The
    # ``object_identity`` argument is accepted for API compatibility but the
    # authoritative uid comes from the pair, not the raw detections.
    return persons, objects


def build_tracks(frame, detections, yolo, movenet, tracker_ns, namespace_offset):
    """DEPRECATED shim — kept only for backwards compatibility with old
    call sites. The live pipeline now uses build_tracks_real() with
    detector.track() output. Do NOT use this in production: it assigned
    fake per-frame ids and was the root cause of the audit's P0 finding.
    """
    raise RuntimeError(
        "build_tracks() is deprecated — it assigned fake per-frame ids. "
        "Use build_tracks_real() with YoloDetector.track() output instead."
    )


def main():
    ap = argparse.ArgumentParser(description="Run AI Littering Detection pipeline")
    ap.add_argument("--source", choices=["camo", "file"], default="camo")
    ap.add_argument("--device", type=int, default=-1, help="OpenCV device index (-1 = auto-discover via camera_discovery)")
    ap.add_argument("--video", type=str, default="", help="path for --source file")
    ap.add_argument("--buffer", type=float, default=6.0, help="circular buffer window (s)")
    ap.add_argument("--analysis-fps", type=float, default=10.0)
    ap.add_argument("--pre", type=float, default=3.0)
    ap.add_argument("--post", type=float, default=3.0)
    ap.add_argument("--camera-id", type=str, default="cam-01")
    ap.add_argument("--post-backend", type=str, default="", help="FastAPI URL to POST events")
    ap.add_argument("--show", action="store_true", help="display live frame (dev)")
    args = ap.parse_args()

    cfg = PipelineConfig(
        buffer_seconds=args.buffer,
        analysis_fps=args.analysis_fps,
        pre_seconds=args.pre,
        post_seconds=args.post,
        camera_id=args.camera_id,
        post_backend_url=args.post_backend or None,
        # Adaptive self-tuning (adaptive_tuner.py): concurrent tier ladder +
        # online learning store. In file mode the learning record is
        # attributed to the video file name.
        auto_tune=True,
    )
    pipe = InferencePipeline(cfg)
    if args.source == "file" and args.video:
        # Attribute the online-learning record to the video file name.
        try:
            pipe.event_detector.learning_video = os.path.basename(args.video)
        except Exception:
            pass

    # source
    if args.source == "camo":
        device_idx = args.device
        if device_idx < 0:
            # auto-discover: find a LIVE camera via camera_discovery
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
        src = CameraSource(device_index=device_idx, target_fps=30)
    else:
        src = VideoFileSource(args.video)
    if not src.open():
        print(f"ERROR: cannot open source ({args.source})", file=sys.stderr)
        sys.exit(1)

    # CV components (lazy-loaded; needs laptop deps)
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

    # Class-agnostic NOVELTY (scene-change) detector — additive, separate source.
    # Reads NOVELTY_* keys from config/events.yaml (NOVELTY_ENABLED gates it).
    from inference.detection.novelty_detector import NoveltyDetector, NoveltyConfig
    nov_cfg = NoveltyConfig.from_yaml()
    nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
    if nov is not None:
        print(f"Novelty (scene-change) detector ENABLED: bg_frames={nov.config.bg_frames}, "
              f"stationary_frames={nov.config.stationary_frames}, person_mask_pad={nov.config.person_mask_pad}")
    else:
        print("Novelty (scene-change) detector DISABLED (NOVELTY_ENABLED=false).")

    print(f"Running. buffer={args.buffer}s analysis_fps={args.analysis_fps} pre/post={args.pre}/{args.post}s")
    print("Press Ctrl+C to stop.")

    _latest_frame = [None]  # mutable holder so the lambda can see updates

    # Register the frame source with the backend's MJPEG stream router so the
    # dashboard LiveCamera component receives real frames instead of the
    # "WAITING FOR CAMERA" placeholder. This is optional — if the backend
    # isn't importable, the stream simply stays on the placeholder.
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
    try:
        for pkt in src:
            t_start = time.time()

            # REAL ByteTrack path: detector.track() returns TrackedDetection
            # objects with stable ids (persist=True keeps tracker state).
            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker_ns, frame_count, run_pose=run_pose, nov=nov,
            )
            events = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)

            # Visualize the SAME production inference results for MJPEG stream.
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
                    source_fps=float(getattr(src, "fps", 0.0) or 0.0),
                    analysis_fps=float(cfg.analysis_fps),
                    video_name=args.source,
                    event=current_event,
                )
                annotated = render_analysis_frame(pkt.frame, analysis, event=current_event)
            except Exception as e:
                print(f"[visualization warning] {e}", file=sys.stderr)
                annotated = pkt.frame
            _latest_frame[0] = annotated  # expose annotated real frame to stream

            last_infer_time = (time.time() - t_start) * 1000.0

            for ev in events:
                print(f"[{ev.event_timestamp:.2f}] 🚨 LITTERING CONFIRMED — {ev.object_type} "
                      f"person={ev.person_track_id} object={ev.object_track_id} conf={ev.confidence:.2f}")
            if args.show and has_cv2:
                cv2.imshow("littering", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_count += 1
            
            # Push live state every 5 frames for responsive real-time UI
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
                        "bbox": {"x": x1/w_img, "y": y1/h_img, "w": (x2-x1)/w_img, "h": (y2-y1)/h_img},
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
                        "bbox": {"x": x1/w_img, "y": y1/h_img, "w": (x2-x1)/w_img, "h": (y2-y1)/h_img},
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
                    live_entities=entities
                )

            if frame_count % 100 == 0:
                st = pipe.stats()
                fps = frame_count / max(1e-6, time.time() - t0)
                print(f"[stats] {st} capture_fps={fps:.1f} latency={last_infer_time:.1f}ms")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
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
