"""Long-running live session hygiene (Section 6c).

Resets trackers / FSM identity when the scene is empty for a sustained
window, or on an hourly wall-clock cadence, so ByteTrack IDs and pair
memory cannot grow unbounded across days of continuous capture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LiveHygieneConfig:
    empty_scene_seconds: float = 30.0
    max_session_seconds: float = 3600.0  # hourly hard reset
    min_reset_interval_seconds: float = 60.0  # debounce


class LiveSessionHygiene:
    def __init__(self, config: Optional[LiveHygieneConfig] = None) -> None:
        self.config = config or LiveHygieneConfig()
        self._empty_since: Optional[float] = None
        self._session_started = time.time()
        self._last_reset_at = 0.0

    def observe(self, *, person_count: int, now: Optional[float] = None) -> bool:
        """Return True when a reset should run after this observation."""
        ts = float(now if now is not None else time.time())
        if person_count <= 0:
            if self._empty_since is None:
                self._empty_since = ts
        else:
            self._empty_since = None

        if (ts - self._last_reset_at) < self.config.min_reset_interval_seconds:
            return False

        empty_long_enough = (
            self._empty_since is not None
            and (ts - self._empty_since) >= self.config.empty_scene_seconds
        )
        hourly = (ts - self._session_started) >= self.config.max_session_seconds
        return bool(empty_long_enough or hourly)

    def mark_reset(self, now: Optional[float] = None) -> None:
        ts = float(now if now is not None else time.time())
        self._last_reset_at = ts
        self._session_started = ts
        self._empty_since = None


def reset_live_trackers(
    *,
    detector: Any,
    tracker_ns: Any,
    event_detector: Any,
    novelty: Any = None,
) -> None:
    """Hard-reset YOLO ByteTrack state, TrackStore, FSM, and novelty BG."""
    if detector is not None and hasattr(detector, "reset_tracking"):
        detector.reset_tracking()
    if tracker_ns is not None and hasattr(tracker_ns, "reset"):
        tracker_ns.reset()
    if event_detector is not None and hasattr(event_detector, "reset"):
        event_detector.reset()
    if novelty is not None and hasattr(novelty, "reset"):
        novelty.reset()
