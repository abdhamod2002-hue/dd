"""Promotion gate — promote candidate weights only when holdout improves safely."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

PathLike = Union[str, Path]


@dataclass
class GateDecision:
    promote: bool
    reason: str
    current: Dict[str, float]
    candidate: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promote": self.promote,
            "reason": self.reason,
            "current": self.current,
            "candidate": self.candidate,
        }


def _metric_block(report: Mapping[str, Any]) -> Dict[str, float]:
    """Extract F1 + FPR from a frozen_eval JSON (aggregate section)."""
    if "report" in report and isinstance(report["report"], dict):
        agg = report["report"].get("aggregate")
    else:
        agg = report.get("aggregate")
    if agg is None and "f1" in report:
        agg = report
    if not isinstance(agg, dict):
        raise ValueError("frozen eval report missing aggregate metrics")
    return {
        "precision": float(agg.get("precision") or 0.0),
        "recall": float(agg.get("recall") or 0.0),
        "f1": float(agg.get("f1") or 0.0),
        "fpr": float(agg.get("fpr") or 0.0),
        "tp": float(agg.get("tp") or 0.0),
        "fp": float(agg.get("fp") or 0.0),
        "fn": float(agg.get("fn") or 0.0),
        "tn": float(agg.get("tn") or 0.0),
    }


def _coerce_metrics(payload: Mapping[str, Any]) -> Dict[str, float]:
    if "aggregate" in payload or "report" in payload:
        return _metric_block(payload)
    return {
        "f1": float(payload["f1_score"] if "f1_score" in payload else payload["f1"]),
        "fpr": float(payload.get("false_positive_rate", payload.get("fpr", 0.0))),
        "precision": float(payload.get("precision", 0.0)),
        "recall": float(payload.get("recall", 0.0)),
        "tp": float(payload.get("tp", 0.0)),
        "fp": float(payload.get("fp", 0.0)),
        "fn": float(payload.get("fn", 0.0)),
        "tn": float(payload.get("tn", 0.0)),
    }


def should_promote(
    new_metrics: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    *,
    fpr_slack: float = 1.1,
) -> GateDecision:
    """Section 7e: promote iff F1↑ and FPR ≤ slack × current FPR (default 1.1)."""
    new_m = _coerce_metrics(new_metrics)
    cur_m = _coerce_metrics(current_metrics)

    if cur_m["fpr"] <= 0:
        fpr_cap = 0.0
        fpr_ok = new_m["fpr"] <= 0.0
    else:
        fpr_cap = float(cur_m["fpr"]) * float(fpr_slack)
        fpr_ok = new_m["fpr"] <= fpr_cap + 1e-12

    f1_ok = new_m["f1"] > cur_m["f1"]
    if f1_ok and fpr_ok:
        return GateDecision(
            True,
            f"F1 improved ({cur_m['f1']:.3f} → {new_m['f1']:.3f}) and FPR within "
            f"{fpr_slack:.2f}x ({new_m['fpr']:.3f} ≤ {fpr_cap:.3f})",
            cur_m,
            new_m,
        )
    reasons = []
    if not f1_ok:
        reasons.append(f"F1 not improved ({cur_m['f1']:.3f} → {new_m['f1']:.3f})")
    if not fpr_ok:
        reasons.append(f"FPR regresses beyond slack ({new_m['fpr']:.3f} > {fpr_cap:.3f})")
    return GateDecision(False, "; ".join(reasons), cur_m, new_m)


def should_promote_from_files(
    candidate_report: PathLike,
    current_report: PathLike,
    *,
    fpr_slack: float = 1.1,
) -> GateDecision:
    cand = json.loads(Path(candidate_report).read_text(encoding="utf-8"))
    cur = json.loads(Path(current_report).read_text(encoding="utf-8"))
    return should_promote(cand, cur, fpr_slack=fpr_slack)
