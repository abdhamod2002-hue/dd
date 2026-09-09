"""
MoveNet Pose — 🟢 wrapper.

MoveNet (TensorFlow Hub, TFLite-style signature) gives 17 body keypoints
per person. We only consume the subset the associator needs:
    left_wrist, right_wrist, left_shoulder, right_shoulder
and derive ``torso_center`` as the midpoint of the shoulders.

Why MoveNet over a heavier pose model: it is fast on CPU (the demo
target), and the reference project already integrates it. We run it
**lazily** — only on persons that the associator is currently tracking
or considering, not on every detected person every frame. This is the
single biggest CPU saving in the pipeline.

OFFLINE POLICY (important for the demo laptop):
  * First run WITH internet: the model is downloaded ONCE into
    ``<repo>/models/movenet/`` (persistent, git-ignored cache).
  * Every later run — including fully OFFLINE — loads from that local
    cache. No internet required.
  * If the cache is missing AND there is no internet, ``load()`` raises
    a clear RuntimeError. The system must NEVER report AI READY when
    MoveNet cannot load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np  # type: ignore

from inference.association.person_object_assoc import Keypoints

# MoveNet 17-keypoint indices (Thunderbird/SinglePose)
KP_NOSE = 0
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10

# Persistent local cache root: <repo>/models/movenet/
_REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE_ROOT = _REPO_ROOT / "models" / "movenet"

# Deterministic first-run download handles (TF Hub; redirect to Kaggle storage).
MODEL_URLS = {
    "movenet_singlepose_thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
    "movenet_singlepose_lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4",
}


@dataclass
class PoseResult:
    person_index: int
    keypoints: Keypoints
    confidence: float


class MovenetPose:
    """
    Wraps a TF MoveNet model. Loads lazily into a PERSISTENT local cache.

    The inference input is a cropped person bbox (not the full frame) to
    keep the model small and fast.
    """

    def __init__(self, model_name: str = "movenet_singlepose_thunder", input_size: int = 256) -> None:
        self.model_name = model_name
        self.input_size = input_size
        self._model = None
        self._loaded_from: Optional[str] = None  # "cache" | "download"

    # ------------------------------------------------------------------ #
    def _local_model_dir(self) -> Path:
        return LOCAL_CACHE_ROOT / self.model_name

    def _manifest_path(self) -> Path:
        return LOCAL_CACHE_ROOT / "registry.json"

    def _read_manifest(self) -> dict:
        try:
            import json
            with open(self._manifest_path(), "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_manifest(self, key: str, dirname: str) -> None:
        try:
            import json
            data = self._read_manifest()
            data[key] = dirname
            with open(self._manifest_path(), "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # manifest is an optimization; cache-by-URL still works

    def load(self) -> None:
        """Load MoveNet from the local cache; download once if missing.

        Raises RuntimeError with an explicit, actionable message when the
        model cannot be loaded (no cache + no internet). Never silently
        continues — callers must treat a failed load as fatal for pose.
        """
        import tensorflow as tf  # type: ignore

        # 1) fast path: registered local copy (fully offline)
        local_dir = self._local_model_dir()
        if not (local_dir.is_dir() and any(local_dir.iterdir())):
            # fall back to the registry-mapped TF Hub cache dir
            cached = self._read_manifest().get(self.model_name)
            if cached:
                candidate = LOCAL_CACHE_ROOT / cached
                if candidate.is_dir():
                    local_dir = candidate

        if local_dir.is_dir() and any(local_dir.iterdir()):
            import tensorflow_hub as hub  # type: ignore
            self._model = hub.load(str(local_dir))
            self._loaded_from = "cache"
            return

        # 2) first run: download once into the persistent repo-local cache.
        url = MODEL_URLS.get(self.model_name)
        if url is None:
            raise RuntimeError(
                f"Unknown MoveNet model '{self.model_name}'. "
                f"Known models: {sorted(MODEL_URLS)}"
            )
        try:
            # Redirect TF Hub's cache into the repo so the download is
            # persistent and later runs work fully offline.
            LOCAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
            os.environ["TFHUB_CACHE_DIR"] = str(LOCAL_CACHE_ROOT)
            before = {p.name for p in LOCAL_CACHE_ROOT.iterdir() if p.is_dir()}
            import tensorflow_hub as hub  # type: ignore
            self._model = hub.load(url)
            after = {p.name for p in LOCAL_CACHE_ROOT.iterdir() if p.is_dir()}
            new_dirs = sorted(after - before)
            if not new_dirs:
                # pre-existing cache (downloaded before manifest existed):
                # register any dir that looks like a valid SavedModel
                new_dirs = [
                    d for d in sorted(after)
                    if (LOCAL_CACHE_ROOT / d / "saved_model.pb").is_file()
                ]
            if new_dirs:
                self._write_manifest(self.model_name, new_dirs[0])
        except Exception as e:
            self._model = None
            raise RuntimeError(
                f"MoveNet '{self.model_name}' could NOT be loaded. "
                f"Local cache at {LOCAL_CACHE_ROOT} has no copy and the "
                f"first-run download from {url} failed "
                f"({type(e).__name__}: {e}). Connect to the internet ONCE "
                f"to populate the cache, or manually place the extracted "
                f"TF Hub module at {self._local_model_dir()}. The system "
                f"will NOT report AI READY without pose."
            ) from e
        self._loaded_from = "download"

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_from(self) -> Optional[str]:
        return self._loaded_from

    def estimate(self, frame, person_bboxes) -> List[PoseResult]:
        """
        Run pose on each person crop. Returns one PoseResult per person.
        ``person_bboxes`` is a list of (x1,y1,x2,y2).
        """
        if self._model is None:
            raise RuntimeError("MovenetPose.load() must be called first")
        import tensorflow as tf  # type: ignore

        results: List[PoseResult] = []
        for idx, bbox in enumerate(person_bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            inp = tf.image.resize_with_pad(tf.convert_to_tensor(crop)[tf.newaxis, ...], self.input_size, self.input_size)
            inp = tf.cast(inp, dtype=tf.int32)
            outs = self._model.signatures["serving_default"](input=inp)
            kps = outs["output_0"].numpy()[0, 0, :, :]  # (17, 3) — y, x, conf
            results.append(self._to_pose_result(idx, kps, (x1, y1, x2, y2)))
        return results

    @staticmethod
    def _to_pose_result(idx: int, kps: np.ndarray, bbox) -> PoseResult:
        def get(i):
            y, x, c = kps[i]
            if c < 0.2:
                return None
            # map back to frame coords
            x1, y1, x2, y2 = bbox
            h = max(1, y2 - y1)
            w = max(1, x2 - x1)
            return (x1 + float(x) * w, y1 + float(y) * h)

        lw = get(KP_LEFT_WRIST)
        rw = get(KP_RIGHT_WRIST)
        ls = get(KP_LEFT_SHOULDER)
        rs = get(KP_RIGHT_SHOULDER)
        nose = get(KP_NOSE)
        nose_conf = float(kps[KP_NOSE][2]) if nose is not None else 0.0
        tc = None
        if ls is not None and rs is not None:
            tc = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        kp = Keypoints(
            left_wrist=lw,
            right_wrist=rw,
            torso_center=tc,
            left_shoulder=ls,
            right_shoulder=rs,
            nose=nose,
            nose_confidence=nose_conf,
        )
        avg_conf = float(np.mean(kps[:, 2]))
        return PoseResult(person_index=idx, keypoints=kp, confidence=avg_conf)
