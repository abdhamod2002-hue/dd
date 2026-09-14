"""Unit tests for Section 7 offline self-learning (no GPU train)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from learning_offline.dataset_builder import build_mixed_dataset
from learning_offline.gate import should_promote
from learning_offline.ingest import write_verdict
from learning_offline.promote import promote_weights


def _fake_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal valid-ish JPEG via OpenCV if available; else raw bytes placeholder.
    try:
        import cv2  # type: ignore

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = (40, 80, 120)
        cv2.imwrite(str(path), img)
    except Exception:
        path.write_bytes(b"\xff\xd8\xff\xd9")


def test_write_verdict_confirm_and_reject(tmp_path, monkeypatch):
    import learning_offline.ingest as ingest

    monkeypatch.setattr(ingest, "DATA_INCOMING", tmp_path / "incoming")
    waste = tmp_path / "src_waste.jpg"
    _fake_jpg(waste)

    d1 = write_verdict(
        101,
        "confirm",
        crop_paths={"crop_waste.jpg": waste},
        day="2026-09-13",
    )
    assert (d1 / "verdict.json").is_file()
    assert (d1 / "labels.txt").read_text(encoding="utf-8").startswith("0 ")
    assert (d1 / "crop_waste.jpg").is_file()

    d2 = write_verdict(
        102,
        "reject",
        crop_paths={"crop_waste.jpg": waste},
        day="2026-09-13",
    )
    assert (d2 / "labels.txt").read_text(encoding="utf-8") == ""
    meta = json.loads((d2 / "verdict.json").read_text(encoding="utf-8"))
    assert meta["hard_negative"] is True


def test_build_mixed_dataset_50_50(tmp_path, monkeypatch):
    import learning_offline.dataset_builder as db
    import learning_offline.ingest as ingest

    incoming = tmp_path / "incoming"
    replay = tmp_path / "replay"
    runs = tmp_path / "runs"
    monkeypatch.setattr(ingest, "DATA_INCOMING", incoming)
    monkeypatch.setattr(db, "DATA_INCOMING", incoming)
    monkeypatch.setattr(db, "DATA_REPLAY", replay)
    monkeypatch.setattr(db, "OFFLINE_RUNS", runs)

    for i in range(4):
        img = tmp_path / f"w{i}.jpg"
        _fake_jpg(img)
        write_verdict(i, "confirm", crop_paths={"crop_waste.jpg": img}, day="2026-09-13")
    for i in range(4, 8):
        folder = replay / "2026-01-01" / f"event_{i}"
        folder.mkdir(parents=True)
        img = folder / "crop_waste.jpg"
        _fake_jpg(img)
        (folder / "labels.txt").write_text("0 0.5 0.5 0.9 0.9\n", encoding="utf-8")
        (folder / "verdict.json").write_text(
            json.dumps({"verdict": "confirm", "train_image": "crop_waste.jpg"}),
            encoding="utf-8",
        )

    yaml_path = build_mixed_dataset(run_id="t1", new_ratio=0.5, seed=0, max_samples=8)
    assert yaml_path.is_file()
    assert "trash_bag" in yaml_path.read_text(encoding="utf-8")
    assert (runs / "t1" / "dataset" / "images" / "train").is_dir()


def test_gate_promotes_only_when_f1_up_and_fpr_safe():
    current = {"f1": 0.50, "fpr": 0.10, "precision": 0.7, "recall": 0.4}
    better = {"f1": 0.60, "fpr": 0.10, "precision": 0.8, "recall": 0.5}
    worse_fpr = {"f1": 0.60, "fpr": 0.20, "precision": 0.5, "recall": 0.7}
    no_f1 = {"f1": 0.50, "fpr": 0.05, "precision": 0.9, "recall": 0.3}

    assert should_promote(better, current).promote is True
    assert should_promote(worse_fpr, current).promote is False
    assert should_promote(no_f1, current).promote is False

    # Current FPR=0 → candidate must also be 0
    cur0 = {"f1": 0.55, "fpr": 0.0, "precision": 1.0, "recall": 0.38}
    cand_fp = {"f1": 0.70, "fpr": 0.05, "precision": 0.9, "recall": 0.6}
    assert should_promote(cand_fp, cur0).promote is False


def test_promote_versions_best_pt(tmp_path):
    prod = tmp_path / "best.pt"
    cand = tmp_path / "cand.pt"
    prod.write_bytes(b"OLD")
    cand.write_bytes(b"NEW")
    # Point module constants via args
    meta = promote_weights(cand, decision={"promote": True}, production=prod)
    assert prod.read_bytes() == b"NEW"
    assert meta["archived_previous"] is not None
    assert Path(meta["archived_previous"]).read_bytes() == b"OLD"
