# Motared Offline Self-Learning (Section 7)

This loop is **offline and versioned**. It does **not** change FSM thresholds
or YOLO weights during live/upload analysis (that would break determinism).

`adaptive_tuner.LearningStore` (online threshold tweaks) is a **different**
system — keep it separate.

## Flow

1. Operator reviews an event → `POST /api/events/{id}/verdict`
   `{ "verdict": "confirm" | "reject", "notes": "..." }`
2. Files land in `data/incoming/YYYY-MM-DD/event_{id}/`
   - `crop_waste.jpg`, `crop_person.jpg` (from evidence when available)
   - `labels.txt` (YOLO) — empty for **reject** hard-negatives
   - `verdict.json`
3. Weekly (or on demand):
   ```bash
   python -m learning_offline.retrain --epochs 20 --device 0
   ```
   Builds a **50% new / 50% replay** dataset (`data/replay`), fine-tunes with
   `freeze=10` (backbone) and low `lr0=0.001` (Ultralytics + continual-learning
   guidance), writes `inference/detection/weights/candidates/<run_id>.pt`.
4. Evaluate candidate vs production on the **frozen** holdout
   (`evaluation/run_frozen_eval.py`) — point the pipeline at each weight set.
5. Gate + optional promote:
   ```bash
   python -m learning_offline.gate_cli \
     --current-report evaluation/reports/frozen_eval_CURRENT.json \
     --candidate-report evaluation/reports/frozen_eval_CANDIDATE.json \
     --candidate-weights inference/detection/weights/candidates/RUN.pt \
     --promote
   ```
   Promotes only if **F1 increases** and **FPR ≤ 1.1 × current** (or stays 0
   when current FPR is 0). Archives previous `best.pt` as `best.pt.vN`.

## Research notes applied

- Experience replay ~50/50 — Ultralytics continual-learning issues + catastrophic forgetting literature.
- `freeze=10` ≈ YOLOv8 backbone — Ultralytics community guidance.
- Low LR AdamW fine-tune — reduces weight shock on scarce new labels.
- Promotion gate — Motared Section 7e (Precision/Recall/F1 on frozen set).
