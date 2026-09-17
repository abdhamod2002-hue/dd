"""P1-E: frozen eval manifest gate (no videos required).

Pins the promotion contract so CI fails loudly if it is weakened:
- frozen set exists, frozen=true, version pinned
- protocol pins delta_t/t_max (no silent re-scoring)
- report files carry frozen_set_version + aggregate P/R/F1/FPR
- learning_offline/gate.py enforces F1-up AND FPR <= 1.1x (no FP regression)
- all three file-mode entry points share build_shared_file_pipeline
"""
from __future__ import annotations

import json
from pathlib import Path

FROZEN = Path(__file__).resolve().parents[1] / "evaluation" / "frozen_test_set.v1.json"
ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "evaluation" / "reports"


def test_frozen_test_set_is_marked_frozen_and_well_formed():
    data = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert data.get("frozen") is True
    assert data.get("version") == "1.0"
    assert data["protocol"]["delta_t_sec"] == 3.0
    assert data["protocol"]["t_max_sec"] == 10.0
    clips = data.get("clips") or []
    assert len(clips) >= 1
    for clip in clips:
        assert clip.get("id")
        assert clip.get("file")
        assert clip.get("label") in {"LITTER", "NO_EVENT"}
        if clip["label"] == "LITTER":
            assert clip.get("event_time_sec") is not None
        if clip.get("skip"):
            continue


def test_run_frozen_eval_module_imports():
    import evaluation.run_frozen_eval as mod

    assert mod.FROZEN_SET.is_file()
    loaded = mod._load_frozen(mod.FROZEN_SET)
    assert loaded["frozen"] is True


def test_latest_frozen_report_carries_contract_metrics():
    reports = sorted(REPORT_DIR.glob("frozen_eval_*.json"))
    assert reports, "no frozen eval report — run evaluation/run_frozen_eval.py"
    data = json.loads(reports[-1].read_text(encoding="utf-8"))
    assert data.get("frozen_set_version") == "1.0"
    agg = data["report"]["aggregate"]
    for k in ("precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"):
        assert k in agg, k
    assert data["report"]["temporal"]["delta_t_sec"] == 3.0


def test_promotion_gate_requires_f1_up_and_fpr_capped():
    from learning_offline.gate import should_promote

    base = {"f1": 0.857, "fpr": 0.333, "precision": 0.75, "recall": 1.0,
            "tp": 3, "fp": 1, "fn": 0, "tn": 2}
    better = dict(base, f1=0.9)
    assert should_promote(better, base).promote is True
    fp_regression = dict(base, f1=0.95, fpr=0.9)  # F1 up but FPR explodes
    assert should_promote(fp_regression, base).promote is False
    f1_flat = dict(base, f1=0.857)
    assert should_promote(f1_flat, base).promote is False


def test_file_mode_entry_points_share_one_loop():
    import scripts.run_pipeline as rp

    assert callable(rp.build_shared_file_pipeline)
    for rel in ("backend/routers/analysis.py", "evaluation/run_frozen_eval.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert "build_shared_file_pipeline" in text, rel
