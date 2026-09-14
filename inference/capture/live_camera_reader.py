"""Threaded live camera reader (Section 6 — Live Camera / RTSP).

Design (OpenCV + RTSP community practice):
  * One dedicated capture thread owns ``cv2.VideoCapture`` and always keeps
    ONLY the newest decoded frame under a lock (drops backlog → low latency).
  * A time-bounded ``CircularFrameBuffer`` retains the last N seconds for
    evidence clips assembled *after* FSM confirmation (pre/post event).
  * On read failure, reconnect with capped exponential backoff instead of
    spinning or exiting the process.

Supports USB indices (Camo/Iriun) and ``rtsp://`` / ``http://`` URLs.
Inference must run on a separate thread/loop that only calls
``get_latest_frame()`` — never share the VideoCapture across threads.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import List, Optional, Tuple, Union

from inference.capture.camera_source import FramePacket
from inference.capture.circular_buffer import BufferedFrame, CircularFrameBuffer

logger = logging.getLogger(__name__)

SourceSpec = Union[int, str]


class LiveCameraReader:
    """Capture-thread reader with latest-frame slot + ring buffer."""

    def __init__(
        self,
        source: SourceSpec,
        *,
        buffer_seconds: float = 10.0,
        target_fps: int = 30,
        max_frames: Optional[int] = None,
        reconnect_delay: float = 1.0,
        reconnect_backoff: float = 2.0,
        max_reconnect_delay: float = 30.0,
        jpeg_encode: bool = False,
    ) -> None:
        self.source = source
        self.target_fps = int(target_fps)
        self.reconnect_delay = float(reconnect_delay)
        self.reconnect_backoff = float(reconnect_backoff)
        self.max_reconnect_delay = float(max_reconnect_delay)
        ceiling = int(max_frames) if max_frames is not None else max(
            60, int(buffer_seconds * target_fps) + 30
        )
        self.ring_buffer = CircularFrameBuffer(
            window_seconds=float(buffer_seconds),
            max_frames=ceiling,
            jpeg_encode=jpeg_encode,
        )
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._running = False
        self._connected = False
        self._thread: Optional[threading.Thread] = None
        self._cap = None
        self._frame_index = 0
        self._reconnect_count = 0
        self._last_error: Optional[str] = None

    # ------------------------------------------------------------------ #
    @property
    def is_connected(self) -> bool:
        return bool(self._connected)

    @property
    def reconnect_count(self) -> int:
        return int(self._reconnect_count)

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="LiveCameraReader",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)
        self._thread = None
        self._release_cap()
        with self._lock:
            self._connected = False

    def get_latest_frame(self) -> Optional[FramePacket]:
        with self._lock:
            pkt = self._latest
            if pkt is None:
                return None
            # Shallow copy of the packet; frame array is the live buffer's
            # current ndarray — callers that mutate must .copy() themselves.
            return FramePacket(
                frame=pkt.frame,
                timestamp=pkt.timestamp,
                frame_index=pkt.frame_index,
                width=pkt.width,
                height=pkt.height,
            )

    def get_buffered_clip(
        self,
        seconds_before: float,
        *,
        around_ts: Optional[float] = None,
        seconds_after: float = 0.0,
    ) -> List[BufferedFrame]:
        """Extract frames for evidence around ``around_ts`` (default: now)."""
        ts = float(around_ts if around_ts is not None else time.time())
        if seconds_after > 0:
            return self.ring_buffer.get_around(
                ts, pre_seconds=float(seconds_before), post_seconds=float(seconds_after)
            )
        return self.ring_buffer.get_window(before_ts=ts, seconds=float(seconds_before))

    # ------------------------------------------------------------------ #
    def _open_capture(self):
        import cv2  # type: ignore
        import sys

        src = self.source
        if isinstance(src, str) and src.lower().startswith("rtsp://"):
            # Prefer TCP for RTSP stability (OpenCV FFMPEG option).
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp"
            )
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap

        if isinstance(src, str) and (
            src.lower().startswith("http://") or src.lower().startswith("https://")
        ):
            return cv2.VideoCapture(src)

        # USB / Camo index
        index = int(src)
        if sys.platform == "win32":
            for backend in (
                getattr(cv2, "CAP_DSHOW", None),
                getattr(cv2, "CAP_MSMF", None),
            ):
                if backend is None:
                    continue
                try:
                    cap = cv2.VideoCapture(index, backend)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                        try:
                            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        except Exception:
                            pass
                        return cap
                    cap.release()
                except Exception:
                    pass
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        return cap

    def _release_cap(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def _capture_loop(self) -> None:
        delay = self.reconnect_delay
        while self._running:
            try:
                self._release_cap()
                self._cap = self._open_capture()
                if self._cap is None or not self._cap.isOpened():
                    raise RuntimeError(f"cannot open source {self.source!r}")
                with self._lock:
                    self._connected = True
                    self._last_error = None
                delay = self.reconnect_delay
                logger.info("LiveCameraReader connected to %r", self.source)

                while self._running:
                    ok, frame = self._cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("capture read failed")
                    h, w = frame.shape[:2]
                    ts = time.time()
                    pkt = FramePacket(
                        frame=frame,
                        timestamp=ts,
                        frame_index=self._frame_index,
                        width=int(w),
                        height=int(h),
                    )
                    self._frame_index += 1
                    # Ring buffer gets a copy so inference can mutate safely.
                    self.ring_buffer.push(frame.copy(), timestamp=ts, frame_index=pkt.frame_index)
                    with self._lock:
                        self._latest = pkt
            except Exception as exc:
                self._reconnect_count += 1
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
                    # Keep last good frame for MJPEG / UI; do not null it out.
                logger.warning(
                    "LiveCameraReader disconnect (%s); reconnect #%d in %.1fs",
                    exc,
                    self._reconnect_count,
                    delay,
                )
                self._release_cap()
                # Interruptible sleep for backoff
                end = time.time() + delay
                while self._running and time.time() < end:
                    time.sleep(0.1)
                delay = min(self.max_reconnect_delay, delay * self.reconnect_backoff)

        self._release_cap()
        with self._lock:
            self._connected = False
