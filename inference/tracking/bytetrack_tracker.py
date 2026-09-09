"""
ByteTrack Tracker + TrackStore — 🟢 real adapter around ultralytics.

This is NO LONGER a stub. The YoloDetector.track() method calls
ultralytics' ``model.track(..., tracker="bytetrack.yaml", persist=True)``
which runs the real ByteTrack algorithm and returns stable ``box.id``
values across frames. This module:

  1. namespaces the raw ByteTrack ids so person ids and object ids never
     collide (persons: 1..N, objects: 10001..N),
  2. maintains a TrackStore — a rolling history of (track_id → recent
     centroids/keypoints) so the association layer and the pipeline can
     look up where a tracked entity was recently,
  3. converts TrackedDetection (from the detector) into Track objects
     (consumed by the association engine) after namespacing.

The actual tracking algorithm lives inside ultralytics. We do not
reimplement ByteTrack — we wrap it correctly and persist its state.

Why two separate tracker streams: persons are large and slow; litter
objects are small and, once thrown, fast. Keeping them in separate
``model.track(...)`` calls (one per YOLO model) gives each its own
ByteTrack instance and ID space, which is exactly what the association
engine's namespaced IDs expect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from inference.association.person_object_assoc import Keypoints, Track
from inference.detection.yolo_detector import TrackedDetection


@dataclass
class TrackHistory:
    """Rolling history for one tracked entity."""

    track_id: int
    class_name: str
    centroids: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    last_seen_frame: int = -1
    is_person: bool = False


class BytetrackTracker:
    """
    Real ByteTrack adapter + TrackStore.

    The detector already ran ``model.track(...)`` and produced
    ``TrackedDetection`` objects with stable raw ids. This class
    namespaces those ids and records them in the store.

    ID convention:
      person raw id 5  → namespaced 5
      object raw id 5  → namespaced 10005
    """

    PERSON_ID_OFFSET = 0
    OBJECT_ID_OFFSET = 10_000

    def __init__(self, person_tracker_cfg: str = "bytetrack.yaml", object_tracker_cfg: str = "bytetrack.yaml") -> None:
        # cfg strings are informational; the detector passes tracker="bytetrack.yaml"
        self.person_cfg = person_tracker_cfg
        self.object_cfg = object_tracker_cfg
        self._store: Dict[int, TrackHistory] = {}
        self._frame_index = 0

    def load(self) -> None:
        """No-op now — tracking is performed by the detector's model.track().
        Kept for API compatibility with the pipeline that calls load()."""
        return None

    # ------------------------------------------------------------------ #
    # Namespacing (pure, unit-testable)
    # ------------------------------------------------------------------ #
    def namespace(self, raw_id: int, is_person: bool) -> int:
        return raw_id + (self.PERSON_ID_OFFSET if is_person else self.OBJECT_ID_OFFSET)

    def denamespace(self, ns_id: int) -> Tuple[int, bool]:
        if ns_id >= self.OBJECT_ID_OFFSET:
            return ns_id - self.OBJECT_ID_OFFSET, False
        return ns_id - self.PERSON_ID_OFFSET, True

    # ------------------------------------------------------------------ #
    # Store operations
    # ------------------------------------------------------------------ #
    def update(self, tracked: List[TrackedDetection], frame_index: int) -> None:
        """Record the current frame's tracked detections into the store."""
        self._frame_index = frame_index
        for td in tracked:
            ns_id = self.namespace(td.track_id, td.is_person)
            hist = self._store.get(ns_id)
            if hist is None:
                hist = TrackHistory(track_id=ns_id, class_name=td.class_name, is_person=td.is_person)
                self._store[ns_id] = hist
            hist.centroids.append(td.centroid)
            hist.last_bbox = td.bbox
            hist.last_seen_frame = frame_index

    def get_history(self, namespaced_id: int) -> Optional[TrackHistory]:
        return self._store.get(namespaced_id)

    def active_ids(self, within_frames: int = 5) -> List[int]:
        """Return ids seen within the last ``within_frames`` frames."""
        cutoff = self._frame_index - within_frames
        return [tid for tid, h in self._store.items() if h.last_seen_frame >= cutoff]

    def prune(self, max_age_frames: int = 60) -> None:
        """Drop tracks not seen for max_age_frames."""
        cutoff = self._frame_index - max_age_frames
        for tid in list(self._store.keys()):
            if self._store[tid].last_seen_frame < cutoff:
                del self._store[tid]

    @property
    def store_size(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------ #
    # Conversion: TrackedDetection → Track (for the association engine)
    # ------------------------------------------------------------------ #
    def to_tracks(
        self,
        tracked: List[TrackedDetection],
        keypoints_by_person_ns: Optional[Dict[int, Keypoints]] = None,
    ) -> Tuple[List[Track], List[Track]]:
        """
        Convert namespaced tracked detections into the Track dataclass the
        association engine consumes, splitting persons vs objects.

        ``keypoints_by_person_ns`` maps namespaced person id → Keypoints
        (produced by MoveNet on the person crop). Optional because pose
        is lazy — if not provided, persons get keypoints=None and the
        associator's hand-occlusion fallback (torso distance) is unused.
        """
        persons: List[Track] = []
        objects: List[Track] = []
        kp = keypoints_by_person_ns or {}
        for td in tracked:
            ns_id = self.namespace(td.track_id, td.is_person)
            t = Track(
                track_id=ns_id,
                class_name=td.class_name,
                centroid=td.centroid,
                bbox=td.bbox,
                keypoints=kp.get(ns_id) if td.is_person else None,
                confidence=td.confidence,
                source=getattr(td, "source", "yolo"),
            )
            if td.is_person:
                persons.append(t)
            else:
                objects.append(t)
        return persons, objects
