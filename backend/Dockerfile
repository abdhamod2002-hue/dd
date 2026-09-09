FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10 \
    PYTHONPATH=/app

# Minimal system deps: libpq for psycopg2, libgl/libglib for opencv, ffmpeg
# for video reading. Build-essential is omitted — all wheels are prebuilt.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libpq5 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend-only deps first (smaller layer cache).
COPY backend/_deps_check.txt /tmp/deps_backend.txt
RUN pip install --no-cache-dir -r /tmp/deps_backend.txt

# Full app deps (CV + TensorFlow + ultralytics). Split into stages for
# cache reuse and so we can retry the heavy step independently.
RUN pip install --no-cache-dir --timeout 1000 --retries 10 \
        numpy==1.26.4 \
        opencv-python==4.10.0.84

# The pipeline runs on CPU (device="cpu"). Pre-install CPU-only torch +
# torchvision from the PyTorch CPU index so that `pip install ultralytics`
# satisfies its `torch>=1.8 / torchvision>=0.9` requirements with the already
# installed CPU wheels instead of fetching the multi-GB CUDA/CUDA-capable
# torch + NVIDIA packages from PyPI. (ultralytics 8.3.40 otherwise pulls
# torch 2.13.0-cuN + nvidia-cudnn/cusparselt — huge and unnecessary for CPU.)
RUN pip install --no-cache-dir --timeout 1000 --retries 10 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu

RUN pip install --no-cache-dir --timeout 1000 --retries 10 \
        ultralytics==8.3.40

RUN pip install --no-cache-dir --timeout 1000 --retries 10 \
        tensorflow==2.16.1 \
        tensorflow-hub==0.16.1

# Application code: backend + inference + scripts + models.
RUN pip install --no-cache-dir supervision==0.30.2

COPY backend/ /app/backend/
COPY inference/ /app/inference/
COPY scripts/ /app/scripts/
COPY models/ /app/models/
# CACHEBUST forces re-COPY when the on-disk weights change but the size happens
# to collide with a stale cached layer (bump the value to refresh weights).
# yolov8n_cpu.pt: byte-identical copy of yolov8n.pt (distinct path busts
# BuildKit context/layer caches that served a stale, legacy-format file).
ARG CACHEBUST=yolov8n-2
COPY yolov8n_cpu.pt /app/yolov8n.pt
COPY inference/detection/weights/best.pt /app/inference/detection/weights/best.pt
# Semantic waste-bag detector (human-annotated Roboflow CC BY 4.0 dataset;
# sha256 2f33b83654789ef273151db3a7c1ef89adb405d076e46ca9659bcdd68ba6a824).
# Loaded by YoloDetector._resolve_bag_weights() at startup, once.
COPY inference/detection/weights/garbage_bag_v2.pt /app/inference/detection/weights/garbage_bag_v2.pt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
