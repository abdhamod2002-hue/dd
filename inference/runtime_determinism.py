"""Process-wide determinism for Motared inference.

Same input video must yield the same detection/FSM decisions across runs.
Sources of non-determinism we harden against:

* GPU / cuDNN algorithm selection (benchmark, TF32, FP16)
* Python / NumPy / PyTorch RNGs
* TensorFlow MoveNet ops (separate stack from YOLO/ultralytics)

CUBLAS_WORKSPACE_CONFIG and PYTHONHASHSEED should be set before the first
``import torch`` when possible; we still set them here so long-running
backend workers get them before the next CUDA workspace allocation.

References:
* https://docs.pytorch.org/docs/stable/notes/randomness.html
* ultralytics ``init_seeds(..., deterministic=True)``
* https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
"""

from __future__ import annotations

import os
import random
from typing import Optional

_CONFIGURED = False
_DEFAULT_SEED = 0


def configure_determinism(seed: int = _DEFAULT_SEED) -> dict:
    """Seed RNGs and force deterministic backend knobs. Idempotent."""
    global _CONFIGURED
    # Env must be set even on repeat calls (worker may have cleared them).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    # oneDNN can reorder FP ops across runs (TF itself warns about this).
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    random.seed(seed)
    report: dict = {"seed": seed, "torch": False, "tensorflow": False}

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        report["numpy"] = True
    except Exception as exc:  # pragma: no cover
        report["numpy_error"] = str(exc)

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = False
        report["torch"] = True
        report["cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover
        report["torch_error"] = str(exc)

    try:
        import tensorflow as tf  # type: ignore

        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
        try:
            # TF 2.8+ preferred path; may raise if already initialized oddly.
            tf.config.experimental.enable_op_determinism()
            report["tf_op_determinism"] = True
        except Exception as exc:
            report["tf_op_determinism_error"] = str(exc)
        report["tensorflow"] = True
    except Exception as exc:
        # MoveNet may load later; env TF_DETERMINISTIC_OPS still helps.
        report["tensorflow_error"] = str(exc)

    _CONFIGURED = True
    report["configured"] = True
    return report


def is_configured() -> bool:
    return _CONFIGURED


def determinism_enabled_from_env() -> bool:
    """Default ON. Set MOTARED_DETERMINISTIC=0 to opt out."""
    raw: Optional[str] = os.environ.get("MOTARED_DETERMINISTIC")
    if raw is None or raw.strip() == "":
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}
