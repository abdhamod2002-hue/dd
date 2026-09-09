#!/usr/bin/env python3
"""
Real-video positive-scenario probe (NOT a unit test — an engineering probe).

Builds a REALISTIC littering video from the real photograph:
  frames 0-14   : person holding bottle near hand          (HOLDING)
  frames 15-24  : bottle travels CONTINUOUSLY hand -> ground in a smooth arc
                  (real throws are continuous; ByteTrack keeps ONE stable id)
  frames 25-44  : bottle static on ground; person keeps walking away

Then runs the FULL production pipeline (YOLO + ByteTrack + MoveNet +
association + FSM + voting + evidence) and reports exactly what the AI
decided — no forcing either way.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

REAL_IMAGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tests", "real_video", "person_bottle2.jpg")
OUT = os.path.join(os.path.dirname(__file__), "_probe_positive.mp4")


def build(path: str) -> bool:
    img = cv2.imread(REAL_IMAGE)
    H, W = img.shape[:2]
    from ultralytics import YOLO
    m = YOLO("yolov8n.pt")
    res = m(img, verbose=False)[0]
    person_box = bottle_box = None
    for b in res.boxes:
        name = m.names[int(b.cls[0])]
        if name == "person" and person_box is None:
            person_box = tuple(map(int, b.xyxy[0].tolist()))
        elif name == "bottle" and bottle_box is None:
            bottle_box = tuple(map(int, b.xyxy[0].tolist()))
    if not person_box or not bottle_box:
        print("FAIL: base image must contain person+bottle")
        return False

    bx1, by1, bx2, by2 = bottle_box
    # TIGHT crop: bottle only — no hand/context (a pad of 60 pulled the
    # person's hand into the "flying object", producing garbage detections).
    tpad = 6
    patch = img[max(0, by1 - tpad):min(H, by2 + tpad),
                max(0, bx1 - tpad):min(W, bx2 + tpad)].copy()
    ph, pw = patch.shape[:2]
    # carried photo WITHOUT the bottle: inpaint the region so the background
    # looks natural (a flat gray rectangle broke person detection confidence).
    blanked = img.copy()
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[max(0, by1 - 4):min(H, by2 + 4), max(0, bx1 - 4):min(W, bx2 + 4)] = 255
    blanked = cv2.inpaint(blanked, mask, 7, cv2.INPAINT_TELEA)

    out_W, out_H = 1280, 720
    scale = (out_H - 40) / H
    img_s = cv2.resize(img, (int(W * scale), int(H * scale)))
    blanked_s = cv2.resize(blanked, (int(W * scale), int(H * scale)))
    ph_s, pw_s = max(8, int(ph * scale)), max(8, int(pw * scale))
    patch_s = cv2.resize(patch, (pw_s, ph_s))

    # REALISTIC TIMING @30fps: hold 1s, throw arc 1s (clear downward motion,
    # ~25px/frame), then ground + person walking away for 1.5s.
    fps = 30.0
    n_frames = 105
    hold_end = 30
    arc_end = 60
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_W, out_H))
    start_x = 60
    end_x = out_W - img_s.shape[1] - 60

    # release point = where the hand holds the bottle IN THE SCALED PHOTO;
    # landing point = ground band at bottom-right of the release x.
    hand_x = start_x + int((bx1 + tpad) * scale) + pw_s // 2
    hand_y = iy0 = 20 + int((by1 + tpad) * scale) + ph_s // 2
    ground_cx = min(out_W - pw_s - 10, hand_x + 120)
    ground_cy = min(int(out_H * 0.88), out_H - ph_s // 2 - 2)  # low band, clamped

    for i in range(n_frames):
        frame = np.full((out_H, out_W, 3), 40, dtype=np.uint8)
        frame[int(out_H * 0.75):, :] = (52, 110, 60)  # grass floor band

        t01 = i / (n_frames - 1)
        ix = int(start_x + t01 * (end_x - start_x))
        iy = 20

        if i < hold_end:
            h_, w_ = img_s.shape[:2]
            frame[iy:iy + h_, ix:ix + w_] = img_s
        else:
            h_, w_ = blanked_s.shape[:2]
            frame[iy:iy + h_, ix:ix + w_] = blanked_s

        if hold_end <= i < arc_end:
            # RELEASE: continuous descent hand -> ground (downward + slight drift)
            u = (i - hold_end) / float(arc_end - hold_end - 1)
            cx = int(hand_x + u * (ground_cx - hand_x))
            cy = int(hand_y + (u ** 1.5) * (ground_cy - hand_y))
            tx, ty = cx - pw_s // 2, cy - ph_s // 2
            tx = min(max(0, tx), out_W - pw_s)
            ty = min(max(0, ty), out_H - ph_s)
            frame[ty:ty + ph_s, tx:tx + pw_s] = patch_s
        elif i >= arc_end:
            tx = ground_cx - pw_s // 2
            ty = ground_cy - ph_s // 2
            frame[ty:ty + ph_s, tx:tx + pw_s] = patch_s

        vw.write(frame)
    vw.release()
    return True


def main() -> int:
    if not build(OUT):
        return 1
    print(f"built {OUT}")

    from inference.capture.camera_source import VideoFileSource
    from inference.detection.yolo_detector import YoloDetector
    from inference.pose.movenet_pose import MovenetPose
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from inference.pipeline import InferencePipeline, PipelineConfig
    from scripts.run_pipeline import build_tracks_real

    src = VideoFileSource(OUT)
    assert src.open()
    print(f"video: {src.total_frames} frames @ {src.fps:.1f} fps ({src.duration_seconds:.1f}s)")

    detector = YoloDetector(person_weights="yolov8n.pt", litter_weights="nonexistent.pt",
                            litter_conf=0.25, person_conf=0.30)
    detector.load()
    movenet = MovenetPose(); movenet.load()
    tracker = BytetrackTracker(); tracker.load()

    cfg = PipelineConfig(buffer_seconds=6.0, analysis_fps=30.0,
                         pre_seconds=1.0, post_seconds=1.5)
    cfg.event_detector_config.min_carried_frames = 3
    cfg.event_detector_config.min_stationary_frames = 4
    cfg.event_detector_config.min_departed_frames = 2
    cfg.event_detector_config.confirmation_grace_frames = 2
    cfg.event_detector_config.smoothing_window = 3
    cfg.event_detector_config.stationary_window_frames = 3
    cfg.event_detector_config.max_pair_age_frames = 20
    cfg.event_detector_config.min_event_confidence = 0.65
    cfg.assoc_config.min_persistence = 2
    cfg.assoc_config.bind_radius = 90.0
    cfg.assoc_config.frame_height = 720.0
    pipe = InferencePipeline(cfg)

    persons_seen, objects_seen = set(), {}
    detector_timeline = []
    frame_idx = 0
    for pkt in src:
        tracked = detector.track(pkt.frame, persist=True)
        run_pose = pipe.should_analyze(pkt.timestamp)
        persons, objects = build_tracks_real(pkt.frame, tracked, movenet, tracker, frame_idx, run_pose=run_pose)
        for p in persons:
            persons_seen.add(p.track_id)
        for o in objects:
            objects_seen.setdefault(o.track_id, o.class_name)
        evs = pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
        for ev in evs:
            print(f"  >>> CONFIRMED at t={ev.event_timestamp:.2f}s obj={ev.object_type} "
                  f"person={ev.person_track_id} object={ev.object_track_id} conf={ev.confidence:.2f}")
        states = {k: mem.state.value for k, mem in pipe.event_detector._pairs.items()}
        cur = list(states.values())
        if not detector_timeline or detector_timeline[-1][1] != str(cur):
            detector_timeline.append((round(pkt.timestamp, 2), str(cur)))
        frame_idx += 1
    src.release()
    pipe.finalize()

    print("\n=== PROBE RESULT ===")
    print("frames processed:", frame_idx)
    print("person ids:", persons_seen)
    print("object ids:", {k: v for k, v in objects_seen.items()})
    print("detector timeline:")
    for ts, st in detector_timeline:
        print(f"  t={ts}: {st}")
    print("confirmed events:", len(pipe.events))
    print("detector summary:", pipe.event_detector.summary())
    print("finalized artifacts:", list(pipe.finalized_artifacts.keys()))
    for eid, art in pipe.finalized_artifacts.items():
        ok_s = art.snapshot_path and os.path.getsize(art.snapshot_path) > 0
        ok_v = art.video_path and os.path.exists(art.video_path) and os.path.getsize(art.video_path) > 0
        print(f"  artifact {eid}: snapshot_ok={bool(ok_s)} video_ok={bool(ok_v)} dur={art.duration_seconds:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
