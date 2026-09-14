"""Unit tests for frozen-set temporal P/R/F1 (Section 5c) — no GPU required."""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.metrics import (
    ClipResult,
    apply_temporal_scores,
    evaluate,
    match_events_temporal,
)


def test_match_events_temporal_delta_t():
    tp, fp, fn = match_events_temporal([10.0], [11.5], delta_t_sec=3.0, t_max_sec=10.0)
    assert (tp, fp, fn) == (1, 0, 0)


def test_match_events_temporal_late_is_fp_and_fn():
    # Prediction 12s after GT (> T_max) → FP; GT unmatched → FN
    tp, fp, fn = match_events_temporal([10.0], [25.0], delta_t_sec=3.0, t_max_sec=10.0)
    assert tp == 0 and fp == 1 and fn == 1


def test_clip_level_precision_recall_f1():
    results = [
        ClipResult("a", "pos", True, True),
        ClipResult("b", "pos", True, True),
        ClipResult("c", "neg", False, False),
        ClipResult("d", "neg", False, True),  # FP
        ClipResult("e", "pos", True, False),  # FN
    ]
    report = evaluate(results)
    # TP=2 FP=1 FN=1 TN=1
    assert report.aggregate.tp == 2
    assert report.aggregate.fp == 1
    assert report.aggregate.fn == 1
    assert report.aggregate.tn == 1
    assert abs(report.aggregate.precision - 2 / 3) < 1e-9
    assert abs(report.aggregate.recall - 2 / 3) < 1e-9
    assert abs(report.aggregate.f1 - (2 * (2 / 3) * (2 / 3) / (4 / 3))) < 1e-9


def test_apply_temporal_scores_hard_negative():
    r = ClipResult(
        "n1",
        "hard_neg",
        ground_truth=False,
        predicted=True,
        pred_event_times_sec=[4.0, 5.0],
    )
    apply_temporal_scores([r], delta_t_sec=3.0, t_max_sec=10.0)
    assert r.temporal_tp == 0
    assert r.temporal_fp == 2
    assert r.temporal_fn == 0


def test_frozen_test_set_is_balanced_and_marked_frozen():
    path = Path(__file__).resolve().parents[1] / "evaluation" / "frozen_test_set.v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["frozen"] is True
    clips = [c for c in data["clips"] if not c.get("skip")]
    pos = [c for c in clips if c["label"] == "LITTER"]
    neg = [c for c in clips if c["label"] == "NO_EVENT"]
    assert len(clips) >= 10
    assert len(pos) >= 5
    assert len(neg) >= 3
    assert data["protocol"]["delta_t_sec"] == 3.0
    assert data["protocol"]["t_max_sec"] == 10.0
