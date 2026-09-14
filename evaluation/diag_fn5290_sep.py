#!/usr/bin/env python3
"""Diagnose why IMG_5290 never leaves BAG_CARRIED after AIDM-sep path."""

from __future__ import annotations

import json
import tempfile
import time
from collections import Counter
from pathlib import Path

from backend.services.video_normalizer import ensure_cfr_source
from inference.capture.camera_source import VideoFileSource
from inference.detection.novelty_detector import NoveltyConfig, NoveltyDetector
from inference.detection.yolo_detector import YoloDetector
from inference.pipeline import InferencePipeline, PipelineConfig
from inference.pose.movenet_pose import MovenetPose
from inference.runtime_determinism import configure_determinism
from inference.tracking.bytetrack_tracker import BytetrackTracker
from littering_event_detector import EventState, LitteringEventDetector
from scripts.run_pipeline import build_tracks_real

VIDEO = Path("/data/22/IMG_5290.MOV")
OUT = Path("evaluation/reports/fn5290_post_sep_diag.json")


def main() -> None:
    configure_determinism(0)
    ticks = []
    with tempfile.TemporaryDirectory(prefix="d5290_") as td:
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
                camera_id="fn-diag",
                post_backend_url=None,
                auto_tune=True,
                deterministic=True,
            )
        )
        pipe.event_detector.reset()
        base = pipe.event_detector
        # Adaptive wrapper may expose underlying FSM via _detectors or itself.
        fsm = base
        if hasattr(base, "_detectors"):
            dets = getattr(base, "_detectors")
            if isinstance(dets, dict) and dets:
                fsm = next(iter(dets.values()))
            elif isinstance(dets, list) and dets:
                fsm = dets[0]
        if not isinstance(fsm, LitteringEventDetector):
            # AdaptiveEventDetector often subclasses / proxies
            fsm = getattr(base, "_detector", None) or getattr(base, "inner", None) or base

        orig = fsm._advance_pair

        def wrap(mem, info, ts, fi):
            before = mem.state
            out = orig(mem, info, ts, fi)
            after = mem.state
            if before == EventState.BAG_CARRIED or after in (
                EventState.BAG_CARRIED,
                EventState.BAG_RELEASED,
                EventState.BAG_ON_GROUND,
            ):
                row = {
                    "f": fi,
                    "t": round(float(ts), 2),
                    "k": [mem.person_uid, mem.bag_id],
                    "before": before.name,
                    "st": after.name,
                    "c": mem.carried_frames,
                    "wd": None
                    if info.wrist_d_norm is None
                    else round(float(info.wrist_d_norm), 4),
                    "max_wd": round(float(mem.max_wrist_d_norm), 4),
                    "wkp": bool(info.wrist_keypoints_available),
                    "att": bool(mem.ever_aidm_attached),
                    "sc": bool(info.carried),
                    "ng": bool(info.near_ground_plane),
                    "mwp": bool(info.moves_with_person),
                    "pm": bool(info.person_moving),
                    "cont": round(float(info.containment), 3),
                    "nd": round(float(info.norm_distance), 3),
                }
                try:
                    row["aidm_sep"] = bool(fsm._aidm_separation_release(mem, info))
                    row["aidm_ok"] = bool(fsm._aidm_allows_release(mem, info))
                except Exception as exc:  # noqa: BLE001
                    row["err"] = str(exc)
                ticks.append(row)
            return out

        fsm._advance_pair = wrap  # type: ignore[method-assign]

        t0 = time.time()
        n = 0
        last_ts = 0.0
        for pkt in source:
            n += 1
            last_ts = float(pkt.timestamp)
            tracked = detector.track(pkt.frame, persist=True)
            run_pose = pipe.should_analyze(pkt.timestamp)
            persons, objects = build_tracks_real(
                pkt.frame,
                tracked,
                movenet,
                tracker,
                n - 1,
                run_pose=run_pose,
                nov=nov,
            )
            pipe.process_frame(pkt.frame, pkt.timestamp, persons, objects)
        source.release()
        pipe.finalize(last_ts)

        conf = list(pipe.event_detector.confirmed_events)
        rej = list(pipe.event_detector.rejected_events)
        reasons = Counter(getattr(e, "reason", None) for e in rej)
        interesting = [
            t
            for t in ticks
            if (t.get("wd") or 0) >= 0.18
            or t["st"] in ("BAG_RELEASED", "BAG_ON_GROUND", "PERSON_DEPARTED")
            or t.get("aidm_sep")
            or t.get("aidm_ok")
        ]
        sep_true = [t for t in ticks if t.get("aidm_sep")]
        high_wd_carried = [
            t
            for t in ticks
            if t["before"] == "BAG_CARRIED"
            and (t.get("wd") or 0) >= 0.20
            and t.get("wkp")
        ]
        cfg = fsm.config
        out = {
            "frames": n,
            "elapsed": round(time.time() - t0, 1),
            "n_conf": len(conf),
            "reasons": dict(reasons),
            "cfg_sep": float(cfg.aidm_wrist_separate_ratio),
            "cfg_attach": float(cfg.aidm_wrist_attach_ratio),
            "cfg_carry": int(cfg.min_carried_frames),
            "ticks": len(ticks),
            "sep_true_n": len(sep_true),
            "high_wd_carried_n": len(high_wd_carried),
            "high_wd_carried": high_wd_carried[:40],
            "interesting": interesting[:60],
            "rej": [
                {
                    "r": e.reason,
                    "c": (e.details or {}).get("carried_frames"),
                    "max_wd": (e.details or {}).get("max_wrist_d_norm"),
                    "wd": (e.details or {}).get("wrist_d_norm"),
                    "loc": (e.details or {}).get("location_status"),
                    "att": (e.details or {}).get("ever_aidm_attached"),
                }
                for e in rej[:12]
            ],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(json.dumps({k: out[k] for k in [
            "frames", "n_conf", "reasons", "cfg_sep", "cfg_carry",
            "ticks", "sep_true_n", "high_wd_carried_n",
        ]}, indent=2))
        print("high_wd sample", json.dumps(high_wd_carried[:8], indent=2))
        print("rej", json.dumps(out["rej"], indent=2))


if __name__ == "__main__":
    main()
