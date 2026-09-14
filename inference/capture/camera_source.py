"""
Camera Source Abstraction — 🟢 wrapper.

Abstracts how frames arrive from the iPhone so the pipeline doesn't care
whether it's Camo/Iriun USB, an RTSP stream, or an offline video file.

The chosen transport for this project is **Camo/Iriun over USB**: the
iPhone runs the Camo/Iriun app, which exposes the phone as a standard
UVC webcam to the laptop. OpenCV reads it with ``cv2.VideoCapture(0)``
(or a configurable device index). No custom protocol — the OS sees a
normal webcam.

This module is a thin wrapper around OpenCV. It is imported lazily so
the rest of the inference package imports without OpenCV installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

# cv2 / numpy imported lazily inside methods so this module (and the
# inference package) imports cleanly without CV libs installed. The core
# logic modules (buffer, association, FSM, voting, evidence, pipeline)
# never import cv2 at module load.


@dataclass
class FramePacket:
    """One frame from the capture loop, ready for the pipeline."""

    frame: object  # np.ndarray (BGR) — opaque to the core logic
    timestamp: float
    frame_index: int
    width: int
    height: int


class CameraSource:
    """
    USB webcam-style source (Camo/Iriun exposes the iPhone this way).

    Parameters
    ----------
    device_index : int
        OpenCV device index. Camo/Iriun usually register as 0 or 1.
        Run ``python scripts/list_cameras.py`` to find the right index.
    target_fps : int
        Capture target. The OS/camera may deliver fewer; we timestamp
        every frame we actually receive.
    resolution : (width, height)
        Requested capture resolution. The camera may not honor it.
    """

    def __init__(
        self,
        device_index: int = 0,
        target_fps: int = 30,
        resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.device_index = device_index
        self.target_fps = target_fps
        self.resolution = resolution
        self._cap = None  # cv2.VideoCapture, set in open()
        self._frame_index = 0

    def open(self) -> bool:
        import sys
        import cv2  # type: ignore
        # On Windows, try CAP_DSHOW and CAP_MSMF explicitly to avoid FFMPEG fallback warning
        if sys.platform == "win32":
            for backend in [getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None)]:
                if backend is not None:
                    try:
                        self._cap = cv2.VideoCapture(self.device_index, backend)
                        if self._cap.isOpened():
                            break
                        self._cap.release()
                    except Exception:
                        pass
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            return False
        if self.resolution:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        return True

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        h, w = frame.shape[:2]
        pkt = FramePacket(
            frame=frame,
            timestamp=time.time(),
            frame_index=self._frame_index,
            width=w,
            height=h,
        )
        self._frame_index += 1
        return pkt

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[FramePacket]:
        while True:
            pkt = self.read()
            if pkt is None:
                break
            yield pkt


class VideoFileSource:
    """
    Offline source: reads frames from a video file. Useful for evaluation
    and the test dataset (100 clips). Same FramePacket contract.
    """

    def __init__(self, path: str, target_fps: Optional[int] = None) -> None:
        self.path = path
        self.target_fps = target_fps
        self._cap = None  # cv2.VideoCapture, set in open()
        self._frame_index = 0
        self._start_ts: Optional[float] = None
        self.total_frames: int = 0
        self.fps: float = 30.0
        self.duration_seconds: float = 0.0
        self.width: int = 0
        self.height: int = 0

    def open(self) -> bool:
        import cv2  # type: ignore
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            return False
        
        # Read video metadata
        cnt = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if orig_fps <= 0 or orig_fps > 120 or not (orig_fps == orig_fps):
            orig_fps = 30.0
        self.fps = float(self.target_fps) if self.target_fps else orig_fps
        self.total_frames = max(0, cnt)
        self.duration_seconds = self.total_frames / self.fps if self.fps > 0 else 0.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def read(self) -> Optional[FramePacket]:
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        # Deterministic timeline for file sources: never anchor to wall-clock
        # (wall-clock start made analysis-tick scheduling differ across runs).
        if self._start_ts is None:
            self._start_ts = 0.0
        ts = float(self._start_ts) + (self._frame_index / max(1.0, self.fps))
        h, w = frame.shape[:2]
        pkt = FramePacket(frame=frame, timestamp=ts, frame_index=self._frame_index, width=w, height=h)
        self._frame_index += 1
        return pkt

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self) -> Iterator[FramePacket]:
        while True:
            pkt = self.read()
            if pkt is None:
                break
            yield pkt
