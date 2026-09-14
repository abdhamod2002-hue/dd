"""
Evaluation Metrics — clip-level + MIVIA-style temporal event matching.

Clip-level (behavioural):
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2PR / (P + R)

Temporal (MIVIA-IWDD-500 style, Greco et al. / IllegalWasteDumping):
    A predicted confirmation at t_pred matches a GT event at t_gt when
    |t_pred - t_gt| <= Δt (default 3 s). Predictions farther than T_max
    (default 10 s) from every GT event are false positives. Unmatched GT
    events are false negatives.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass
class ClipResult:
    clip_id: str
    scenario: str
    ground_truth: bool  # True if this clip is a real littering event
    predicted: bool  # True if system confirmed littering
    latency_seconds: float = 0.0
    fps: float = 0.0
    # Optional temporal fields (Section 5c)
    gt_event_time_sec: Optional[float] = None
    pred_event_times_sec: List[float] = field(default_factory=list)
    temporal_tp: int = 0
    temporal_fp: int = 0
    temporal_fn: int = 0


@dataclass
class ScenarioMetrics:
    scenario: str
    n: int
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    fpr: float = 0.0
    mean_latency: float = 0.0
    mean_fps: float = 0.0


@dataclass
class EvaluationReport:
    aggregate: ScenarioMetrics
    per_scenario: Dict[str, ScenarioMetrics] = field(default_factory=dict)
    confusion_matrix: Dict[str, int] = field(default_factory=dict)
    all_results: List[ClipResult] = field(default_factory=list)
    temporal: Optional[Dict[str, float]] = None

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "aggregate": asdict(self.aggregate),
                    "per_scenario": {k: asdict(v) for k, v in self.per_scenario.items()},
                    "confusion_matrix": self.confusion_matrix,
                    "temporal": self.temporal,
                    "all_results": [asdict(r) for r in self.all_results],
                },
                f,
                indent=2,
            )

    def summary_str(self) -> str:
        lines = ["=" * 60, "EVALUATION REPORT", "=" * 60, ""]
        a = self.aggregate
        lines.append(
            f"AGGREGATE (n={a.n}): P={a.precision:.3f} R={a.recall:.3f} "
            f"F1={a.f1:.3f} FPR={a.fpr:.3f}"
        )
        lines.append(f"  TP={a.tp} FP={a.fp} FN={a.fn} TN={a.tn}")
        lines.append(
            f"  mean latency={a.mean_latency:.2f}s  mean FPS={a.mean_fps:.1f}"
        )
        if self.temporal:
            t = self.temporal
            lines.append("")
            lines.append(
                f"TEMPORAL (Δt={t.get('delta_t_sec')}s, T_max={t.get('t_max_sec')}s): "
                f"P={t.get('precision', 0):.3f} R={t.get('recall', 0):.3f} "
                f"F1={t.get('f1', 0):.3f}"
            )
            lines.append(
                f"  TP={t.get('tp')} FP={t.get('fp')} FN={t.get('fn')}"
            )
        lines.append("")
        lines.append("PER-SCENARIO:")
        lines.append(f"  {'scenario':<28} {'n':>3} {'P':>6} {'R':>6} {'F1':>6} {'FPR':>6}")
        for name, m in sorted(self.per_scenario.items()):
            lines.append(
                f"  {name:<28} {m.n:>3} {m.precision:>6.3f} {m.recall:>6.3f} "
                f"{m.f1:>6.3f} {m.fpr:>6.3f}"
            )
        return "\n".join(lines)


def _prf(tp: int, fp: int, fn: int, tn: int = 0) -> Tuple[float, float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return precision, recall, f1, fpr


def match_events_temporal(
    gt_times: Sequence[float],
    pred_times: Sequence[float],
    *,
    delta_t_sec: float = 3.0,
    t_max_sec: float = 10.0,
) -> Tuple[int, int, int]:
    """Greedy 1-1 temporal matching (MIVIA-IWDD style).

    Returns (tp, fp, fn).
    Predictions with min distance to any GT > t_max_sec are FP even if the
    clip is a true littering clip (late / unrelated detections).
    """
    gt = sorted(float(t) for t in gt_times)
    pred = sorted(float(t) for t in pred_times)
    used_gt = set()
    tp = 0
    fp = 0
    for pt in pred:
        best_i = None
        best_d = None
        for i, gt_t in enumerate(gt):
            if i in used_gt:
                continue
            d = abs(pt - gt_t)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        if best_i is None or best_d is None:
            fp += 1
            continue
        if best_d <= float(delta_t_sec):
            used_gt.add(best_i)
            tp += 1
        elif best_d > float(t_max_sec):
            fp += 1
        else:
            # Between Δt and T_max: count as unmatched prediction (FP) —
            # too late/early to credit as the labelled dump.
            fp += 1
    fn = len(gt) - len(used_gt)
    return tp, fp, fn


def evaluate(results: List[ClipResult]) -> EvaluationReport:
    """Compute per-scenario + aggregate clip-level metrics from clip results."""
    by_scenario: Dict[str, List[ClipResult]] = {}
    for r in results:
        by_scenario.setdefault(r.scenario, []).append(r)

    per_scenario: Dict[str, ScenarioMetrics] = {}
    for name, rs in by_scenario.items():
        tp = sum(1 for r in rs if r.predicted and r.ground_truth)
        fp = sum(1 for r in rs if r.predicted and not r.ground_truth)
        fn = sum(1 for r in rs if not r.predicted and r.ground_truth)
        tn = sum(1 for r in rs if not r.predicted and not r.ground_truth)
        p, rec, f1, fpr = _prf(tp, fp, fn, tn)
        latencies = [r.latency_seconds for r in rs if r.predicted and r.latency_seconds > 0]
        fps_vals = [r.fps for r in rs if r.fps > 0]
        per_scenario[name] = ScenarioMetrics(
            scenario=name,
            n=len(rs),
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            precision=p,
            recall=rec,
            f1=f1,
            fpr=fpr,
            mean_latency=sum(latencies) / len(latencies) if latencies else 0.0,
            mean_fps=sum(fps_vals) / len(fps_vals) if fps_vals else 0.0,
        )

    tp = sum(1 for r in results if r.predicted and r.ground_truth)
    fp = sum(1 for r in results if r.predicted and not r.ground_truth)
    fn = sum(1 for r in results if not r.predicted and r.ground_truth)
    tn = sum(1 for r in results if not r.predicted and not r.ground_truth)
    p, rec, f1, fpr = _prf(tp, fp, fn, tn)
    latencies = [r.latency_seconds for r in results if r.predicted and r.latency_seconds > 0]
    fps_vals = [r.fps for r in results if r.fps > 0]
    aggregate = ScenarioMetrics(
        scenario="ALL",
        n=len(results),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=p,
        recall=rec,
        f1=f1,
        fpr=fpr,
        mean_latency=sum(latencies) / len(latencies) if latencies else 0.0,
        mean_fps=sum(fps_vals) / len(fps_vals) if fps_vals else 0.0,
    )

    # Temporal roll-up when any clip carries event times.
    t_tp = sum(int(r.temporal_tp) for r in results)
    t_fp = sum(int(r.temporal_fp) for r in results)
    t_fn = sum(int(r.temporal_fn) for r in results)
    temporal = None
    if any(
        r.gt_event_time_sec is not None or r.pred_event_times_sec for r in results
    ):
        tp_p, tp_r, tp_f1, _ = _prf(t_tp, t_fp, t_fn, 0)
        temporal = {
            "tp": t_tp,
            "fp": t_fp,
            "fn": t_fn,
            "precision": tp_p,
            "recall": tp_r,
            "f1": tp_f1,
            "delta_t_sec": 3.0,
            "t_max_sec": 10.0,
        }

    confusion = {"TP": tp, "FP": fp, "FN": fn, "TN": tn}
    return EvaluationReport(
        aggregate=aggregate,
        per_scenario=per_scenario,
        confusion_matrix=confusion,
        all_results=results,
        temporal=temporal,
    )


def apply_temporal_scores(
    results: Iterable[ClipResult],
    *,
    delta_t_sec: float = 3.0,
    t_max_sec: float = 10.0,
) -> List[ClipResult]:
    """Fill temporal_tp/fp/fn on each ClipResult in-place and return the list."""
    out: List[ClipResult] = []
    for r in results:
        gt_times: List[float] = []
        if r.ground_truth and r.gt_event_time_sec is not None:
            gt_times = [float(r.gt_event_time_sec)]
        elif r.ground_truth and not r.pred_event_times_sec:
            # Positive clip with no labelled time: defer to clip-level only.
            r.temporal_tp = 1 if r.predicted else 0
            r.temporal_fp = 0
            r.temporal_fn = 0 if r.predicted else 1
            out.append(r)
            continue
        tp, fp, fn = match_events_temporal(
            gt_times,
            r.pred_event_times_sec,
            delta_t_sec=delta_t_sec,
            t_max_sec=t_max_sec,
        )
        if not r.ground_truth:
            # Hard negative: every prediction is FP; no FN possible.
            tp, fp, fn = 0, len(r.pred_event_times_sec), 0
        r.temporal_tp = tp
        r.temporal_fp = fp
        r.temporal_fn = fn
        out.append(r)
    return out
