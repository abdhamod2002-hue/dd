"""CFR video normalizer — run once per upload before OpenCV/YOLO.

iPhone (and many phone) recordings are Variable Frame Rate (VFR). OpenCV
can decode a slightly different frame sequence on repeated reads, which
flips FSM thresholds that need N consecutive carried/stationary frames.

This module re-encodes to Constant Frame Rate (CFR) via ffmpeg so every
downstream consumer sees the same frame timeline.

Encoding defaults match ``inference/visualization/h264.py`` (veryfast +
CRF 23): fast enough for upload jobs, quality high enough that YOLO boxes
are not materially degraded. Override via env:

* ``MOTARED_CFR_FPS`` (default 30)
* ``MOTARED_CFR_PRESET`` (default veryfast)
* ``MOTARED_CFR_CRF`` (default 23)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Union

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def _ffmpeg_exe() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for cand in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(cand):
            return cand
    return None


def normalize_video(
    input_path: PathLike,
    output_path: PathLike,
    target_fps: Optional[int] = None,
    *,
    timeout_sec: int = 3600,
) -> str:
    """Convert any input to CFR H.264 at ``target_fps``.

    Returns the absolute path of the normalized file.
    Raises ``RuntimeError`` if ffmpeg is missing or the output is empty.
    """
    src = Path(input_path).resolve()
    dst = Path(output_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"normalize_video: input missing: {src}")

    fps = int(target_fps or os.environ.get("MOTARED_CFR_FPS", "30"))
    preset = os.environ.get("MOTARED_CFR_PRESET", "veryfast")
    crf = os.environ.get("MOTARED_CFR_CRF", "23")

    exe = _ffmpeg_exe()
    if exe is None:
        raise RuntimeError("normalize_video: ffmpeg not found on PATH")

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    cmd = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"fps={fps}",
        "-fps_mode", "cfr",
        "-r", str(fps),
        "-an",
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    log.info(
        "normalize_video: CFR %sfps preset=%s crf=%s -> %s",
        fps, preset, crf, dst.name,
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        if dst.exists():
            dst.unlink(missing_ok=True)
        raise RuntimeError(f"normalize_video: ffmpeg timed out for {src}") from exc

    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size <= 0:
        err = (proc.stderr or proc.stdout or "")[:500]
        if dst.exists():
            dst.unlink(missing_ok=True)
        raise RuntimeError(
            f"normalize_video: ffmpeg failed for {src} (rc={proc.returncode}): {err}"
        )
    return str(dst)


def ensure_cfr_source(
    input_path: PathLike,
    output_dir: PathLike,
    *,
    stem: str = "source_CFR",
    target_fps: Optional[int] = None,
) -> str:
    """Write ``{output_dir}/{stem}.mp4`` and return its path.

    If ``MOTARED_SKIP_CFR=1``, returns the original path unchanged (debug only).
    """
    if os.environ.get("MOTARED_SKIP_CFR", "").strip().lower() in {"1", "true", "yes"}:
        log.warning("ensure_cfr_source: MOTARED_SKIP_CFR set — using original %s", input_path)
        return str(Path(input_path).resolve())

    out = Path(output_dir) / f"{stem}.mp4"
    return normalize_video(input_path, out, target_fps=target_fps)
