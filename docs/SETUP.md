# Setup Guide

## 1. iPhone → Laptop video transport (Camo/Iriun over USB)

The iPhone runs the **Camo** (or **Iriun**) app, which exposes the phone
as a standard UVC webcam over USB. The laptop sees a normal webcam — no
custom protocol, no RTSP.

### Camo (recommended)
1. Install Camo on the laptop: https://reincubate.com/camo/
2. Install the Camo app on the iPhone (App Store).
3. Connect iPhone → laptop with a USB cable.
4. Trust the computer on the iPhone when prompted.
5. Open Camo on both — the laptop preview shows the iPhone camera.
6. Find the OpenCV device index: `python scripts/list_cameras.py`
   (usually 0 or 1).

### Iriun (free alternative)
1. Install Iriun Webcam on the laptop: https://iriunwebcam.com/
2. Install Iriun app on iPhone.
3. USB connect, open Iriun on both.
4. Same `list_cameras.py` step.

## 2. Backend + database (Docker)

```bash
docker-compose up -d postgres backend
# PostgreSQL on :5432, FastAPI on :8000
curl http://localhost:8000/health   # → {"status":"ok"}
```

If you prefer running the backend locally without Docker:
```bash
pip install -r backend/_deps_check.txt
uvicorn backend.main:app --reload
```

## 3. Inference engine (on the laptop — heavy CV deps)

```bash
pip install -r requirements.txt
# download best.pt into inference/detection/weights/ (see repo README of
# the reference project; or train/fine-tune your own)
mkdir -p inference/detection/weights
# place best.pt there
```

## 4. Dashboard

```bash
cd dashboard
npm install
npm run dev    # http://localhost:5173 (proxies /api → :8000)
```

## 5. Run the live pipeline

```bash
# live from iPhone
python scripts/run_pipeline.py --source camo --device 0 --buffer 6 --show

# with backend reporting
python scripts/run_pipeline.py --source camo --device 0 \
    --post-backend http://localhost:8000/api/events
```

## 6. Evaluate on the test dataset

```bash
# after filming + annotating clips (see evaluation/dataset_schema.md)
python scripts/evaluate.py --dataset path/to/dataset --report report.json
```

## 7. Unit tests (no camera, no CV deps)

```bash
pytest tests/ -v
```

These test the 🔴 core contribution layers with synthetic inputs.

## 8. Event detector configuration

The temporal littering event detector reads tunable defaults from:

```bash
config/events.yaml
```

Use `--event-detector` with the live pipeline or evaluation script to run it in
parallel with the existing FSM/voting pipeline:

```bash
python scripts/run_pipeline.py --source file --video clip.mp4 --event-detector
python scripts/evaluate.py --dataset path/to/dataset --event-detector
```

Uploaded video analysis always includes detector diagnostics in the job report.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://litter:litter@localhost:5432/littering` | Backend DB |

## Performance expectations (measure, don't assume)

On CPU-only laptops the full pipeline (YOLO+track+pose+logic) typically
runs below real-time. Design mitigations already in place:

- **Capture ≠ analysis FPS.** Buffer ingests at full camera FPS; analysis
  throttled to `--analysis-fps` (default 10). Evidence clips are smooth
  because they come from the full-FPS buffer.
- **MoveNet is lazy** — only on tracked persons, not every detection.
- **YOLO nano/small** for the demo, not large.
