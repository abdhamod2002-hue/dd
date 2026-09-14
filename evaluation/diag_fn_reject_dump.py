#!/usr/bin/env python3
"""Quick reject-reason dump for one frozen clip (matches run_frozen_eval path)."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
import sys

from backend.services.video_normalizer import ensure_cfr_source
from inference.capture.camera_source import VideoFileSource
from inference.detection.novelty_detector import NoveltyConfig, NoveltyDetector
from inference.detection.yolo_detector import YoloDetector
from inference.pipeline import InferencePipeline, PipelineConfig
from inference.pose.movenet_pose import MovenetPose
from inference.runtime_determinism import configure_determinism
from inference.tracking.bytetrack_tracker import BytetrackTracker
from littering_event_detector import EventState
from scripts.run_pipeline import build_tracks_real

VIDEO = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/22/IMG_5290.MOV")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "evaluation/reports/fn_reject_dump.json")


def main() -> None:
    configure_determinism(0)
    sep_latches = 0
    carried_high = 0
    with tempfile.TemporaryDirectory(prefix="fn_dump_") as td:
        cfr = Path(ensure_cfr_source(VIDEO, td, stem="source_CFR"))
        source = VideoFileSource(str(cfr))
        assert source.open()
        detector = YoloDetector()
        detector.load()
        detector.reset_tracking()
        tracker = BytetrackTracker()
        tracker.load()
        nov_cfg = NoveltyConfig.from_yaml()
        nov = NoveltyDetector(nov_cfg) if nov_cfg.enabled else None
        movenet = MovenetPose()
        movenet.load()
        pipe = InferencePipeline(
            PipelineConfig(
                buffer_seconds=8.0,
                analysis_fps=8.0,
                camera_id="fn-dump",
                post_backend_url=None,
                auto_tune=True,
                deterministic=True,
            )
        )
        pipe.event_detector.reset()
        fsm = pipe.event_detector.primary
        orig = fsm._advance_pair
        samples = []

        def wrap(mem, info, ts, fi):
            nonlocal sep_latches, carried_high
            before = mem.state
            before_sep = mem.aidm_separated
            out = orig(mem, info, ts, fi)
            if mem.aidm_separated and not before_sep:
                sep_latches += 1
                samples.append(
                    {
                        "f": fi,
                        "t": round(float(ts), 2),
                        "wd": None
                        if info.wrist_d_norm is None
                        else round(float(info.wrist_d_norm), 4),
                        "st": mem.state.name,
                        "c": mem.carried_frames,
                        "att": mem.ever_aidm_attached,
                        "rel": mem.release_frame,
                    }
                )
            if before == EventState.BAG_CARRIED and info.wrist_d_norm is not None:
                if float(info.wrist_d_norm) >= 0.20:
                    carried_high += 1
            return out

        fsm._advance_pair = wrap  # type: ignore[method-assign]

        n = 0
        last_ts = 0.0
        for pkt in source:
            n += 1
            last_ts = float(pkt.timestamp)
            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            persons, objects = build_tracks_real(
                pkt.frame, tracked, movenet, tracker, n - 1,
                run_pose=run_pose, nov=nov,
            )
            pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
        source.release()
        pipe.finalize(last_ts)

        conf = list(pipe.event_detector.confirmed_events)
        rej = list(pipe.event_detector.rejected_events)
        out = {
            "video": str(VIDEO),
            "frames": n,
            "n_conf": len(conf),
            "reasons": dict(Counter(e.reason for e in rej)),
            "cfg_sep": float(fsm.config.aidm_wrist_separate_ratio),
            "cfg_carry": int(fsm.config.min_carried_frames),
            "sep_latches": sep_latches,
            "carried_high_wd_ticks": carried_high,
            "latch_samples": samples[:20],
            "conf": [
                {
                    "reason": e.reason,
                    "ts": (e.timestamps or {}),
                    "loc": (e.details or {}).get("location_status"),
                }
                for e in conf
            ],
            "rej": [
                {
                    "r": e.reason,
                    "c": (e.details or {}).get("carried_frames"),
                    "max_wd": (e.details or {}).get("max_wrist_d_norm"),
                    "aidm_sep": (e.details or {}).get("aidm_separated"),
                    "att": (e.details or {}).get("ever_aidm_attached"),
                    "rel": (e.frames or {}).get("release"),
                    "loc": (e.details or {}).get("location_status"),
                    "st": getattr(e.state, "value", str(e.state)),
                }
                for e in rej[:15]
            ],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps({k: out[k] for k in [
            "frames", "n_conf", "reasons", "cfg_sep", "cfg_carry",
            "sep_latches", "carried_high_wd_ticks",
        ]}, indent=2))
        print("latches", json.dumps(samples[:8], indent=2))
        print("rej", json.dumps(out["rej"], indent=2))


if __name__ == "__main__":
    main()
