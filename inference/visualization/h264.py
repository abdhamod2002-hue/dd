"""Video transcoding helpers — H.264 (browser-playable) output.

WHY THIS EXISTS
---------------
The dashboard's `<video>` panels (AI Analyzed Video, Event Clip) were
rendering as empty/black players. Root cause (Phase 2 investigation,
verified with ffprobe): every video this system writes — analyzed.mp4,
event_clip.mp4, evidence_<id>.mp4 — was encoded with OpenCV's ``mp4v``
(MPEG-4 Part 2) fourcc. Chrome/Edge/Firefox do not ship an MPEG-4 Part 2
decoder, so the file exists and streams fine over HTTP but the browser
cannot decode a single frame of it. Byte size was never the problem.

THE FIX
-------
The production containers ship ffmpeg 7.x with libx264 (verified:
``ffmpeg -encoders | grep 264`` → libx264). We therefore write with
OpenCV as before (mp4v, lossless intermediate is fine) and then
re-encode the result to H.264 + AAC-less MP4 via ffmpeg subprocess.
If ffmpeg is unavailable the *original* mp4v file is kept and a warning
logged — playback then degrades exactly to the old behavior instead of
the job dying. No fabricated files: on failure the honest result is the
un-transcoded file plus a log line, never a fake placeholder.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

log = logging.getLogger(__name__)

# H.264 level ceiling for 1080p60; +faststart lets browsers start
# playing before the whole file is downloaded (moov atom up front).
_X264_ARGS = [
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "23",
    "-pix_fmt", "yuv420p",
    "-movflags", "+faststart",
    "-an",
]


def _ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # common container locations when PATH is minimal
    for cand in ("/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.exists(cand):
            return cand
    return None


def extract_clip_h264(
    source_path: str,
    target_path: str,
    start_sec: float,
    end_sec: float,
    timeout_sec: int = 1800,
) -> bool:
    """Cut ``[start_sec, end_sec]`` from ``source_path`` and encode H.264 in ONE ffmpeg pass.

    P1-9: replaces the OpenCV mp4v-write -> ffmpeg-re-encode pipeline, which
    decoded and encoded the same clip footage TWICE (once by OpenCV into the
    mp4v intermediate, once by ffmpeg transcoding it). Here ffmpeg decodes the
    source once and encodes H.264 directly. Returns True only when the output
    exists and is non-empty; callers fall back to the legacy path on False.
    """
    exe = _ffmpeg_exe()
    if exe is None:
        return False
    src = os.path.abspath(source_path)
    if not os.path.exists(src) or os.path.getsize(src) == 0:
        log.warning("extract_clip_h264: source missing/empty: %s", src)
        return False
    start = max(0.0, float(start_sec))
    duration = float(end_sec) - start
    if duration <= 0:
        log.warning("extract_clip_h264: non-positive clip duration (%.3f..%.3f)", start_sec, end_sec)
        return False
    target = os.path.abspath(target_path)
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(target) or ".")
    os.close(tmp_fd)
    os.remove(tmp_path)  # ffmpeg wants to create it itself
    try:
        cmd = [
            exe, "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-i", src, "-t", f"{duration:.3f}",
            "-map", "0:v:0", *_X264_ARGS, tmp_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            log.warning(
                "extract_clip_h264: ffmpeg failed for %s (rc=%s): %s",
                src, proc.returncode, (proc.stderr or "")[:300],
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False
        os.replace(tmp_path, target)
        return True
    except subprocess.TimeoutExpired:
        log.error("extract_clip_h264: ffmpeg timed out for %s — no clip written", src)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    except Exception:
        log.exception("extract_clip_h264: unexpected failure for %s", src)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False


def transcode_to_h264(source_path: str, *, keep_source: bool = True, timeout_sec: int = 1800) -> str | None:
    """Re-encode ``source_path`` in place to browser-playable H.264.

    Returns the final path (which may be the original file if ffmpeg is
    missing or fails — the honest degradation). ``keep_source=False``
    replaces the original mp4v file with the H.264 one to save disk
    (large analyzed videos); ``True`` keeps both during the transcode
    and deletes the original only after verified success.
    """
    exe = _ffmpeg_exe()
    if exe is None:
        log.warning("transcode_to_h264: ffmpeg not found — keeping mp4v file %s", source_path)
        return source_path

    src = os.path.abspath(source_path)
    if not os.path.exists(src) or os.path.getsize(src) == 0:
        log.warning("transcode_to_h264: source missing/empty: %s", src)
        return src

    # Skip if already H.264 — idempotent re-runs must not re-encode.
    probe = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name", "-of", "default=nw=1:nk=1", src],
        capture_output=True, text=True, timeout=60,
    )
    if probe.returncode == 0 and probe.stdout.strip().lower() == "h264":
        return src

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4", dir=os.path.dirname(src) or ".")
    os.close(tmp_fd)
    os.remove(tmp_path)  # ffmpeg wants to create it itself
    try:
        cmd = [exe, "-y", "-v", "error", "-i", src, *_X264_ARGS, tmp_path]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        if proc.returncode != 0 or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            log.error(
                "transcode_to_h264: ffmpeg failed for %s (rc=%s): %s",
                src, proc.returncode, (proc.stderr or "")[:400],
            )
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return src  # honest degradation: un-transcoded mp4v
        # Verified success -> replace the original.
        os.replace(tmp_path, src)
        log.info("transcode_to_h264: %s re-encoded to H.264", src)
        return src
    except subprocess.TimeoutExpired:
        log.error("transcode_to_h264: ffmpeg timed out for %s — keeping original", src)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return src
    except Exception:
        log.exception("transcode_to_h264: unexpected failure for %s", src)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return src
