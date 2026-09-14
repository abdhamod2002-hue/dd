#!/usr/bin/env python3
"""Run Motared against the frozen behavioural test set (Section 5c).

Usage (inside littering-backend or host venv with models):

  python evaluation/run_frozen_eval.py
  python evaluation/run_frozen_eval.py --quick          # short clips only
  python evaluation/run_frozen_eval.py --ids IMG_5117,IMG_5303

Writes:
  evaluation/reports/frozen_eval_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.metrics import (  # noqa: E402
    ClipResult,
    apply_temporal_scores,
    evaluate,
)
from inference.runtime_determinism import configure_determinism  # noqa: E402

FROZEN_SET = Path(__file__).resolve().parent / "frozen_test_set.v1.json"
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def _load_frozen(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("frozen"):
        raise RuntimeError(f"{path} is not marked frozen=true")
    return data


def _resolve_video(file_name: str, roots: List[str]) -> Optional[Path]:
    for root in roots:
        p = Path(root) / file_name
        if p.is_file():
            return p
        # uploaded_videos often prefix timestamps: *_IMG_5117.MOV
        parent = Path(root)
        if parent.is_dir():
            matches = sorted(parent.glob(f"*{file_name}"))
            if matches:
                return matches[-1]
    return None


def _run_clip(path: Path) -> Tuple[bool, List[float], float, float]:
    """Return (confirmed, pred_event_times_sec, latency, fps)."""
    from backend.services.video_normalizer import ensure_cfr_source
    from inference.capture.camera_source import VideoFileSource
    from inference.detection.novelty_detector import NoveltyConfig, NoveltyDetector
    from inference.detection.yolo_detector import YoloDetector
    from inference.pipeline import InferencePipeline, PipelineConfig
    from inference.pose.movenet_pose import MovenetPose
    from inference.tracking.bytetrack_tracker import BytetrackTracker
    from scripts.run_pipeline import build_tracks_real
    import tempfile

    configure_determinism(0)
    with tempfile.TemporaryDirectory(prefix="frozen_cfr_") as td:
        cfr = Path(ensure_cfr_source(path, td, stem="source_CFR"))
        source = VideoFileSource(str(cfr))
        if not source.open():
            raise RuntimeError(f"cannot open {cfr}")

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
                camera_id="frozen-eval",
                post_backend_url=None,
                auto_tune=True,
                deterministic=True,
            )
        )
        pipe.event_detector.reset()

        t0 = time.time()
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
        elapsed = max(1e-6, time.time() - t0)
        fps = n / elapsed

        pred_times: List[float] = []
        for ev in pipe.event_detector.confirmed_events:
            ts = None
            try:
                ts = (ev.timestamps or {}).get("confirmed") or (ev.timestamps or {}).get(
                    "departure"
                ) or (ev.timestamps or {}).get("release")
            except Exception:
                ts = None
            if ts is not None:
                pred_times.append(float(ts))
            else:
                pred_times.append(float(last_ts))

        confirmed = len(pred_times) > 0
        latency = 0.0
        if confirmed and pred_times:
            latency = max(0.0, pred_times[0])
        return confirmed, pred_times, latency, fps


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen Motared behavioural eval")
    ap.add_argument("--set", default=str(FROZEN_SET), help="frozen_test_set JSON")
    ap.add_argument("--quick", action="store_true", help="only clips with event_time<=12s or NO_EVENT short")
    ap.add_argument("--ids", default="", help="comma-separated clip ids")
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()

    data = _load_frozen(Path(args.set))
    protocol = data.get("protocol") or {}
    delta_t = float(protocol.get("delta_t_sec", 3.0))
    t_max = float(protocol.get("t_max_sec", 10.0))
    roots = list(data.get("video_roots") or [])

    want_ids = {x.strip() for x in args.ids.split(",") if x.strip()}
    clips = []
    seen_files = set()
    for c in data.get("clips") or []:
        if c.get("skip"):
            continue
        if want_ids and c.get("id") not in want_ids:
            continue
        if c.get("file") in seen_files:
            continue
        seen_files.add(c.get("file"))
        if args.quick:
            et = c.get("event_time_sec")
            label = str(c.get("label") or "")
            # Keep short positives + all hard negatives under ~30s by name heuristic
            if label == "LITTER" and et is not None and float(et) > 12.0:
                continue
            if c.get("id") in {"IMG_5306", "A", "IMG_5295", "IMG_5290"}:
                continue
        clips.append(c)
        if args.max_clips and len(clips) >= args.max_clips:
            break

    results: List[ClipResult] = []
    missing: List[str] = []
    for c in clips:
        path = _resolve_video(str(c["file"]), roots)
        if path is None:
            missing.append(str(c["file"]))
            print(f"[SKIP] missing video: {c['file']}", flush=True)
            continue
        label = str(c.get("label") or "NO_EVENT").upper()
        gt = label == "LITTER"
        print(f"=== {c['id']} ({path.name}) label={label} ===", flush=True)
        try:
            predicted, pred_times, latency, fps = _run_clip(path)
        except Exception as exc:
            print(f"[ERROR] {c['id']}: {exc}", flush=True)
            continue
        et = c.get("event_time_sec")
        results.append(
            ClipResult(
                clip_id=str(c["id"]),
                scenario=str(c.get("scenario") or "unknown"),
                ground_truth=gt,
                predicted=predicted,
                latency_seconds=float(latency),
                fps=float(fps),
                gt_event_time_sec=(float(et) if et is not None else None),
                pred_event_times_sec=list(pred_times),
            )
        )
        print(
            f"  pred={predicted} times={pred_times} fps={fps:.2f}",
            flush=True,
        )

    if not results:
        print("No clips evaluated.", file=sys.stderr)
        if missing:
            print("Missing:", ", ".join(missing), file=sys.stderr)
        return 2

    apply_temporal_scores(results, delta_t_sec=delta_t, t_max_sec=t_max)
    report = evaluate(results)
    if report.temporal is not None:
        report.temporal["delta_t_sec"] = delta_t
        report.temporal["t_max_sec"] = t_max

    print()
    print(report.summary_str())
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORT_DIR / f"frozen_eval_{stamp}.json"
    payload = {
        "frozen_set": str(Path(args.set).name),
        "frozen_set_version": data.get("version"),
        "generated_at": stamp,
        "missing_videos": missing,
        "protocol": protocol,
        "report": {
            "aggregate": report.aggregate.__dict__,
            "confusion_matrix": report.confusion_matrix,
            "temporal": report.temporal,
            "per_scenario": {k: v.__dict__ for k, v in report.per_scenario.items()},
            "all_results": [r.__dict__ for r in report.all_results],
        },
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
