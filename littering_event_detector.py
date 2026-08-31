"""
Temporal Littering Event Detector.

This module implements a stronger, explainable replacement for the simple
"trash near person, then person leaves" rule found in notebook-style
illegal-dumping demos.

It consumes person tracks and bag/trash tracks produced by the existing
YOLO + ByteTrack pipeline and maintains a temporal state machine per
(person_track_id, bag_track_id) pair.

Design principles
-----------------
1. Prefer temporal evidence over single-frame decisions.
2. Emit explainable rejection reasons instead of silent failures.
3. Keep thresholds configurable per camera/scene.
4. Allow short CSRT fallback gaps, but never let CSRT alone confirm a
   violation; YOLO must re-confirm the bag before final evidence.
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


CONFIG_PATH = os.path.join("config", "events.yaml")


class EventState(str, Enum):
    NO_BAG = "NO_BAG"
    BAG_NEAR_PERSON = "BAG_NEAR_PERSON"
    BAG_CARRIED = "BAG_CARRIED"
    BAG_RELEASED = "BAG_RELEASED"
    BAG_ON_GROUND = "BAG_ON_GROUND"
    PERSON_DEPARTED = "PERSON_DEPARTED"
    VIOLATION_CONFIRMED = "VIOLATION_CONFIRMED"


class RejectionReason(str, Enum):
    PERSON_NOT_DETECTED = "PERSON_NOT_DETECTED"
    BAG_NOT_DETECTED = "BAG_NOT_DETECTED"
    LOW_BAG_CONFIDENCE = "LOW_BAG_CONFIDENCE"
    CSRT_NOT_RECONFIRMED = "CSRT_NOT_RECONFIRMED"
    ASSOCIATION_AMBIGUOUS = "ASSOCIATION_AMBIGUOUS"
    NOT_ENOUGH_CARRIED_FRAMES = "NOT_ENOUGH_CARRIED_FRAMES"
    NO_RELEASE_TRANSITION = "NO_RELEASE_TRANSITION"
    BAG_NOT_STATIONARY = "BAG_NOT_STATIONARY"
    PERSON_DID_NOT_DEPART = "PERSON_DID_NOT_DEPART"
    OTHER_PERSON_CLOSER = "OTHER_PERSON_CLOSER"
    EVENT_CONFIDENCE_TOO_LOW = "EVENT_CONFIDENCE_TOO_LOW"


@dataclass
class EventDetectorConfig:
    """
    Tunable event-detector parameters.

    These are intentionally conservative defaults, not final truths. They
    should be tuned per camera/scene using real validation clips.
    """

    analysis_fps: float = 8.0
    detection_low_conf: float = 0.10
    detection_high_conf: float = 0.25

    min_carried_frames: int = 6
    min_stationary_frames: int = 8
    min_departed_frames: int = 2
    confirmation_grace_frames: int = 8

    near_distance_ratio: float = 0.25
    departure_distance_ratio: float = 1.25
    departure_motion_ratio: float = 0.65
    release_distance_ratio: float = 0.35
    stationary_distance_ratio: float = 0.15
    stationary_window_frames: int = 3
    # ABSOLUTE pixel cap on centroid drift across the stationarity window.
    # Complements (does NOT replace) ``stationary_distance_ratio``: that ratio
    # is normalised by the bag's own bbox size, so a LARGE bag is permitted to
    # drift many more real pixels than a small bag while still reading as
    # "stationary". This cap bounds the drift in absolute pixels regardless of
    # object size. ``None`` disables the extra gate (legacy behaviour).
    stationary_max_pixel_step: Optional[float] = None
    # Generous spatial radius (in person-heights) within which a (person, bag)
    # pair is kept alive. Without this, a bag dropped/left at a moderate
    # distance (e.g. into a bin a meter from the person) stops producing
    # pair-info and the half-finished event dies before reaching the ground
    # state. The STATE MACHINE still decides near/carried/departed; this only
    # controls association continuity.
    association_radius_ratio: float = 1.0
    # Once a bag is on the ground and stationary, if it is NOT re-grabbed for
    # this many analysis-frames the event is confirmed as abandonment. This
    # captures littering where the person stays near the discarded object
    # (e.g. drops it at a bin) instead of walking far away.
    min_abandonment_frames: int = 8

    min_event_confidence: float = 0.80
    max_fallback_tracker_gap_frames: int = 16
    max_pair_age_frames: int = 30
    max_other_person_closer_frames: int = 3

    smoothing_window: int = 5
    association_margin: float = 0.15
    carrying_zone_vertical_start: float = 0.45
    carrying_zone_horizontal_margin: float = 0.75

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventDetectorConfig":
        allowed = {f.name for f in fields(cls)}
        normalized: Dict[str, Any] = {}
        for key, value in data.items():
            name = str(key).lower()
            if name in allowed:
                normalized[name] = value
        return cls(**normalized)


def _coerce_scalar(raw: str) -> Any:
    value = raw.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none", "~"}:
        return None
    try:
        if any(ch in value for ch in {".", "e", "E"}) and not value.isdigit():
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_simple_yaml(path: str) -> Dict[str, Any]:
    """
    Minimal flat YAML parser for config/events.yaml.

    This avoids adding PyYAML as a hard dependency for a small flat config.
    If PyYAML is installed, load_event_config() uses it instead.
    """
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith(("-", " ", "\t")) or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value == "":
                data[key] = None
            else:
                data[key] = _coerce_scalar(value)
    return data


def load_event_config(path: str = CONFIG_PATH) -> EventDetectorConfig:
    """Load detector config from YAML, falling back to defaults."""
    if not os.path.exists(path):
        return EventDetectorConfig()

    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except ImportError:
        data = _parse_simple_yaml(path)
    except Exception:
        data = {}

    if not isinstance(data, dict):
        return EventDetectorConfig()
    return EventDetectorConfig.from_dict(data)


@dataclass
class DetectorKeypoints:
    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    torso_center: Optional[Tuple[float, float]] = None


@dataclass
class DetectorPerson:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float = 1.0
    timestamp: float = 0.0
    frame_index: int = 0
    keypoints: Optional[DetectorKeypoints] = None


@dataclass
class DetectorBag:
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float = 0.0
    timestamp: float = 0.0
    frame_index: int = 0
    class_name: str = "trash_bag"
    source: str = "yolo"  # "yolo" or "csrt_fallback"
    yolo_confirmed: bool = True


def _keypoints_from_any(kp: Any) -> Optional[DetectorKeypoints]:
    if kp is None:
        return None
    return DetectorKeypoints(
        left_wrist=getattr(kp, "left_wrist", None),
        right_wrist=getattr(kp, "right_wrist", None),
        torso_center=getattr(kp, "torso_center", None),
    )


def detector_person_from_track(track: Any, confidence: Optional[float] = None) -> DetectorPerson:
    """Convert an existing Track-like object into DetectorPerson."""
    return DetectorPerson(
        track_id=int(track.track_id),
        bbox=tuple(float(v) for v in track.bbox),  # type: ignore[arg-type]
        confidence=float(confidence if confidence is not None else getattr(track, "confidence", 1.0)),
        timestamp=float(getattr(track, "timestamp", 0.0) or 0.0),
        frame_index=int(getattr(track, "frame_index", 0) or 0),
        keypoints=_keypoints_from_any(getattr(track, "keypoints", None)),
    )


def detector_bag_from_track(
    track: Any,
    confidence: Optional[float] = None,
    source: str = "yolo",
    yolo_confirmed: Optional[bool] = None,
) -> DetectorBag:
    """Convert an existing Track-like object into DetectorBag."""
    conf = float(confidence if confidence is not None else getattr(track, "confidence", 0.0))
    if yolo_confirmed is None:
        yolo_confirmed = source != "csrt_fallback"
    return DetectorBag(
        track_id=int(track.track_id),
        bbox=tuple(float(v) for v in track.bbox),  # type: ignore[arg-type]
        confidence=conf,
        timestamp=float(getattr(track, "timestamp", 0.0) or 0.0),
        frame_index=int(getattr(track, "frame_index", 0) or 0),
        class_name=str(getattr(track, "class_name", "trash_bag")),
        source=source,
        yolo_confirmed=bool(yolo_confirmed),
    )


@dataclass
class EventEvidence:
    carry_score: float
    release_score: float
    stationary_score: float
    departure_score: float
    association_score: float
    detection_score: float
    confidence: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class LitteringEvent:
    event_id: str
    person_track_id: int
    bag_track_id: int
    state: EventState
    confirmed: bool
    reason: str
    confidence: float
    evidence: Dict[str, float]
    frames: Dict[str, Optional[int]]
    timestamps: Dict[str, Optional[float]]
    bag_class: str
    fallback_used: bool
    yolo_reconfirmed: bool
    other_person_closer: bool
    detector_source: str = "yolo"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data


@dataclass
class _PairInfo:
    person_id: int
    bag_id: int
    timestamp: float
    frame_index: int
    distance: float
    norm_distance: float
    person_centroid: Tuple[float, float]
    person_height: float
    near: bool
    carried: bool
    stationary: bool
    departed: bool
    score: float
    containment: float
    bag_confidence: float
    bag_class: str
    fallback: bool
    yolo_confirmed: bool
    is_primary: bool = False
    ambiguous: bool = False
    other_person_closer: bool = False


@dataclass
class _PairMemory:
    person_id: int
    bag_id: int
    state: EventState = EventState.NO_BAG
    first_frame: int = 0
    last_frame: int = 0
    last_timestamp: float = 0.0
    missing_frames: int = 0
    person_seen_frames: int = 0
    bag_seen_frames: int = 0

    near_frames: int = 0
    carried_frames: int = 0
    stationary_frames: int = 0
    departed_frames: int = 0
    release_frames: int = 0

    carry_start_frame: Optional[int] = None
    carry_start_ts: Optional[float] = None
    release_frame: Optional[int] = None
    release_ts: Optional[float] = None
    ground_frame: Optional[int] = None
    ground_ts: Optional[float] = None
    departure_frame: Optional[int] = None
    departure_ts: Optional[float] = None
    confirmed_frame: Optional[int] = None
    confirmed_ts: Optional[float] = None

    max_confidence: float = 0.0
    confidence_sum: float = 0.0
    confidence_count: int = 0
    fallback_frames: int = 0
    fallback_gap_frames: int = 0
    ever_yolo_confirmed: bool = False
    yolo_reconfirmed_after_fallback: bool = False

    other_person_closer_frames: int = 0
    ambiguous_frames: int = 0
    departure_baseline: float = 0.0
    max_departure_ratio: float = 0.0
    abandonment_frames: int = 0
    release_person_centroid: Optional[Tuple[float, float]] = None
    release_person_height: float = 0.0
    bag_class: str = "trash_bag"
    emitted: bool = False

    near_flags: Deque[bool] = field(default_factory=deque)
    carried_flags: Deque[bool] = field(default_factory=deque)
    stationary_flags: Deque[bool] = field(default_factory=deque)
    departed_flags: Deque[bool] = field(default_factory=deque)
    regrab_flags: Deque[bool] = field(default_factory=deque)
    distances: Deque[float] = field(default_factory=deque)
    norm_distances: Deque[float] = field(default_factory=deque)
    association_scores: Deque[float] = field(default_factory=deque)

    def mean_confidence(self) -> float:
        if self.confidence_count <= 0:
            return 0.0
        return self.confidence_sum / float(self.confidence_count)

    def mean_association_score(self) -> float:
        if not self.association_scores:
            return 0.0
        return sum(self.association_scores) / float(len(self.association_scores))


def _centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_height(bbox: Tuple[float, float, float, float]) -> float:
    return max(1.0, float(bbox[3]) - float(bbox[1]))


def _box_width(bbox: Tuple[float, float, float, float]) -> float:
    return max(1.0, float(bbox[2]) - float(bbox[0]))


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _intersection_over_bag(person_box: Tuple[float, float, float, float], bag_box: Tuple[float, float, float, float]) -> float:
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bag_box
    ix1 = max(px1, bx1)
    iy1 = max(py1, by1)
    ix2 = min(px2, bx2)
    iy2 = min(py2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    bag_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return min(1.0, inter / bag_area)


def _in_carrying_zone(person: DetectorPerson, bag: DetectorBag, cfg: EventDetectorConfig) -> bool:
    px1, py1, px2, py2 = person.bbox
    bx, by = _centroid(bag.bbox)
    ph = _box_height(person.bbox)
    pw = _box_width(person.bbox)
    lower_start = py1 + ph * cfg.carrying_zone_vertical_start
    lower_end = py2 + ph * 0.25
    left = px1 - pw * cfg.carrying_zone_horizontal_margin
    right = px2 + pw * cfg.carrying_zone_horizontal_margin
    return lower_start <= by <= lower_end and left <= bx <= right


def _wrist_distance(person: DetectorPerson, bag: DetectorBag) -> Tuple[float, bool]:
    kp = person.keypoints
    if kp is None:
        return math.inf, False
    bc = _centroid(bag.bbox)
    wrists = [w for w in (kp.left_wrist, kp.right_wrist) if w is not None]
    if not wrists:
        return math.inf, False
    return min(_dist(w, bc) for w in wrists), True


def _smooth(flags: Iterable[bool], min_votes: int) -> bool:
    return sum(1 for f in flags if f) >= min_votes


class LitteringEventDetector:
    """
    Stateful detector for temporal littering events.

    Usage:
        detector = LitteringEventDetector(load_event_config())
        for frame_index, timestamp in ...:
            events = detector.update(persons, bags, timestamp, frame_index)
        events.extend(detector.finalize())
    """

    def __init__(self, config: Optional[EventDetectorConfig] = None) -> None:
        self.config = config or EventDetectorConfig()
        self._pairs: Dict[Tuple[int, int], _PairMemory] = {}
        self._bag_history: Dict[int, Deque[Tuple[float, Tuple[float, float]]]] = {}
        self._frame_index = 0
        self._last_timestamp = 0.0
        self.confirmed_events: List[LitteringEvent] = []
        self.rejected_events: List[LitteringEvent] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        persons: List[DetectorPerson],
        bags: List[DetectorBag],
        timestamp: float,
        frame_index: Optional[int] = None,
    ) -> List[LitteringEvent]:
        cfg = self.config
        if frame_index is None:
            frame_index = self._frame_index + 1
        self._frame_index = int(frame_index)
        self._last_timestamp = float(timestamp)

        valid_bags = [b for b in bags if float(b.confidence) >= cfg.detection_low_conf]
        self._update_bag_history(valid_bags)

        infos: List[_PairInfo] = []
        for person in persons:
            if float(person.confidence) < cfg.detection_low_conf:
                continue
            for bag in valid_bags:
                info = self._evaluate_pair(person, bag, timestamp, frame_index)
                if info is not None:
                    infos.append(info)

        primary_infos = self._select_primary_associations(infos)

        active_keys: set = set()
        emitted: List[LitteringEvent] = []
        current_person_ids = {int(p.track_id) for p in persons}
        current_bag_ids = {int(b.track_id) for b in valid_bags}

        # --- Rebind orphaned pairs across color-tracker ID churn ------------
        # The HSV fallback tracker emits unstable IDs: the same physical bag is
        # frequently re-born under a new track id. Without continuity the
        # (person, old_id) pair is killed as BAG_NOT_DETECTED while a brand-new
        # (person, new_id) pair restarts the state machine from NO_BAG and can
        # never accumulate enough carried/released/ground frames before the clip
        # ends. Here an existing, progressed pair whose bag vanished adopts the
        # currently-primary bag for that same person (same class, still
        # associated) and KEEPS all accumulated temporal evidence.
        rebound_bag_ids: set = set()
        for key, mem in list(self._pairs.items()):
            if key in active_keys:
                continue
            if key[1] in current_bag_ids:
                continue  # bag still present this frame -> normal advance path
            if key[0] not in current_person_ids:
                continue  # person gone -> let the missing loop finalize it
            if mem.state not in (
                EventState.BAG_NEAR_PERSON,
                EventState.BAG_CARRIED,
                EventState.BAG_RELEASED,
                EventState.BAG_ON_GROUND,
            ):
                continue
            candidate = None
            for info in primary_infos:
                if (info.person_id, info.bag_id) in rebound_bag_ids:
                    continue
                if info.person_id != mem.person_id:
                    continue
                if info.bag_class != mem.bag_class:
                    continue
                # primary_infos only contains (person, bag) pairs that already
                # passed the association gating, so any entry here is a plausible
                # continuation of the orphaned pair's physical object — including
                # a bag re-born at a moderate distance (e.g. on the ground).
                candidate = info
                break
            if candidate is None:
                continue
            new_key = (mem.person_id, candidate.bag_id)
            existing = self._pairs.get(new_key)
            if existing is not None and existing.carried_frames >= mem.carried_frames:
                # a more advanced pair already exists for the new key; keep it
                del self._pairs[key]
                continue
            # migrate memory to the new bag id (preserve all temporal counters)
            self._pairs[new_key] = mem
            mem.bag_id = candidate.bag_id
            del self._pairs[key]
            active_keys.add(new_key)
            rebound_bag_ids.add(candidate.bag_id)
            emitted.extend(self._advance_pair(mem, candidate, timestamp, frame_index))

        # --- Advance / create pairs for current primary associations --------
        for info in primary_infos:
            key = (info.person_id, info.bag_id)
            if key in rebound_bag_ids:
                continue  # already advanced via the rebind path above
            mem = self._pairs.get(key)
            if mem is None:
                mem = self._create_pair(info)
                self._pairs[key] = mem
            active_keys.add(key)
            emitted.extend(self._advance_pair(mem, info, timestamp, frame_index))

        for key, mem in list(self._pairs.items()):
            if key in active_keys:
                continue
            mem.missing_frames += 1
            person_present = key[0] in current_person_ids
            bag_present = key[1] in current_bag_ids
            if person_present:
                mem.person_seen_frames += 1
            if bag_present:
                mem.bag_seen_frames += 1

            if mem.missing_frames > cfg.max_pair_age_frames:
                forced_reason = None
                if not person_present and mem.state in (EventState.BAG_CARRIED, EventState.BAG_RELEASED, EventState.BAG_ON_GROUND):
                    forced_reason = RejectionReason.PERSON_NOT_DETECTED.value
                elif not bag_present and mem.state in (EventState.BAG_CARRIED, EventState.BAG_RELEASED):
                    forced_reason = RejectionReason.BAG_NOT_DETECTED.value
                emitted.extend(self._finalize_pair(mem, forced_reason=forced_reason))
                del self._pairs[key]

        return emitted

    def finalize(self) -> List[LitteringEvent]:
        emitted: List[LitteringEvent] = []
        for key, mem in list(self._pairs.items()):
            emitted.extend(self._finalize_pair(mem))
            del self._pairs[key]
        return emitted

    def summary(self) -> Dict[str, Any]:
        candidates = len(self.confirmed_events) + len(self.rejected_events)
        acceptance_rate = (len(self.confirmed_events) / candidates) if candidates else 0.0
        reason_counts = Counter(event.reason for event in self.rejected_events)
        return {
            "total_candidates": candidates,
            "confirmed_violations": len(self.confirmed_events),
            "rejected_candidates": len(self.rejected_events),
            "acceptance_rate": round(acceptance_rate, 4),
            "rejection_reason_counts": dict(reason_counts),
            "active_pairs": len(self._pairs),
        }

    def reset(self) -> None:
        self._pairs.clear()
        self._bag_history.clear()
        self.confirmed_events.clear()
        self.rejected_events.clear()
        self._frame_index = 0
        self._last_timestamp = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _update_bag_history(self, bags: List[DetectorBag]) -> None:
        maxlen = max(3, int(self.config.smoothing_window) + 2, int(self.config.stationary_window_frames) + 2)
        for bag in bags:
            hist = self._bag_history.get(bag.track_id)
            if hist is None:
                hist = deque(maxlen=maxlen)
                self._bag_history[bag.track_id] = hist
            hist.append((float(bag.timestamp), _centroid(bag.bbox)))

    def _is_stationary(self, bag: DetectorBag, person_height: float) -> bool:
        hist = self._bag_history.get(bag.track_id)
        if not hist or len(hist) < 2:
            return False
        window = max(2, int(self.config.stationary_window_frames))
        prev_ts, prev_centroid = hist[-min(window, len(hist))]
        curr_ts, curr_centroid = hist[-1]
        dt = max(1e-6, curr_ts - prev_ts)
        # Stationarity is an object property, not a person-height property.
        # Using the bag's own bbox size keeps the threshold stable when the
        # person moves closer to/farther from the camera.
        bag_size = max(1.0, _box_height(bag.bbox), _box_width(bag.bbox))
        step_px = _dist(prev_centroid, curr_centroid)
        step_norm = step_px / bag_size
        # Normalize by the expected analysis window so stationary means
        # "small movement over the window", not an absolute pixel threshold.
        period = 1.0 / max(1e-6, self.config.analysis_fps)
        expected_dt = period * max(1, window - 1)
        if not step_norm < self.config.stationary_distance_ratio * max(1.0, dt / expected_dt):
            return False
        # Absolute-pixel gate (complementary, off unless configured).
        # Bounds the raw drift so a physically large object cannot buy itself a
        # large real-pixel movement budget purely by being large on screen.
        cap = self.config.stationary_max_pixel_step
        if cap is not None and step_px > float(cap):
            return False
        return True

    def _evaluate_pair(
        self,
        person: DetectorPerson,
        bag: DetectorBag,
        timestamp: float,
        frame_index: int,
    ) -> Optional[_PairInfo]:
        cfg = self.config
        pc = _centroid(person.bbox)
        bc = _centroid(bag.bbox)
        ph = _box_height(person.bbox)
        distance = _dist(pc, bc)
        norm_distance = distance / ph
        containment = _intersection_over_bag(person.bbox, bag.bbox)
        carry_zone = _in_carrying_zone(person, bag, cfg)
        wrist_dist, wrist_hit = _wrist_distance(person, bag)
        wrist_near = wrist_hit and (wrist_dist / ph) <= cfg.near_distance_ratio
        stationary = self._is_stationary(bag, ph)

        near = norm_distance <= cfg.near_distance_ratio or containment >= 0.25 or carry_zone
        carried = (wrist_near or carry_zone) and norm_distance <= cfg.near_distance_ratio * 1.5
        departed = norm_distance >= cfg.departure_distance_ratio
        associated = norm_distance <= cfg.association_radius_ratio

        if not (near or departed or containment > 0.0 or associated):
            return None

        proximity = max(0.0, 1.0 - norm_distance / max(cfg.near_distance_ratio * 2.0, 1e-6))
        conf_score = min(1.0, float(bag.confidence) / max(cfg.detection_high_conf, 1e-6))
        score = 0.55 * proximity + 0.25 * containment + 0.20 * conf_score

        return _PairInfo(
            person_id=int(person.track_id),
            bag_id=int(bag.track_id),
            timestamp=float(timestamp),
            frame_index=int(frame_index),
            distance=distance,
            norm_distance=norm_distance,
            person_centroid=pc,
            person_height=ph,
            near=bool(near),
            carried=bool(carried),
            stationary=bool(stationary),
            departed=bool(departed),
            score=float(score),
            containment=float(containment),
            bag_confidence=float(bag.confidence),
            bag_class=str(bag.class_name),
            fallback=(bag.source in ("csrt_fallback", "color")),
            yolo_confirmed=bool(bag.yolo_confirmed),
        )

    def _select_primary_associations(self, infos: List[_PairInfo]) -> List[_PairInfo]:
        by_bag: Dict[int, List[_PairInfo]] = {}
        for info in infos:
            by_bag.setdefault(info.bag_id, []).append(info)

        primary: List[_PairInfo] = []
        for infos_for_bag in by_bag.values():
            infos_for_bag.sort(key=lambda x: x.score, reverse=True)
            best = infos_for_bag[0]
            best.is_primary = True
            if len(infos_for_bag) > 1:
                second = infos_for_bag[1]
                if best.score - second.score < self.config.association_margin:
                    best.ambiguous = True
                best.other_person_closer = second.norm_distance < best.norm_distance * 0.95
            primary.append(best)
        return primary

    def _create_pair(self, info: _PairInfo) -> _PairMemory:
        cfg = self.config
        mem = _PairMemory(
            person_id=info.person_id,
            bag_id=info.bag_id,
            first_frame=info.frame_index,
            last_frame=info.frame_index,
            last_timestamp=info.timestamp,
            bag_class=info.bag_class,
        )
        mem.near_flags = deque(maxlen=cfg.smoothing_window)
        mem.carried_flags = deque(maxlen=cfg.smoothing_window)
        mem.stationary_flags = deque(maxlen=cfg.smoothing_window)
        mem.departed_flags = deque(maxlen=cfg.smoothing_window)
        mem.regrab_flags = deque(maxlen=cfg.smoothing_window)
        mem.distances = deque(maxlen=cfg.smoothing_window)
        mem.norm_distances = deque(maxlen=cfg.smoothing_window)
        mem.association_scores = deque(maxlen=cfg.smoothing_window)
        return mem

    def _advance_pair(
        self,
        mem: _PairMemory,
        info: _PairInfo,
        timestamp: float,
        frame_index: int,
    ) -> List[LitteringEvent]:
        cfg = self.config
        mem.last_frame = frame_index
        mem.last_timestamp = timestamp
        mem.missing_frames = 0
        mem.person_seen_frames += 1
        mem.bag_seen_frames += 1
        mem.bag_class = info.bag_class

        mem.near_flags.append(info.near)
        mem.carried_flags.append(info.carried)
        mem.stationary_flags.append(info.stationary)
        person_motion = 0.0
        if mem.release_person_centroid is not None and mem.release_person_height > 0.0:
            person_motion = _dist(info.person_centroid, mem.release_person_centroid) / mem.release_person_height
        departure_threshold = (
            max(cfg.departure_distance_ratio, mem.departure_baseline * cfg.departure_distance_ratio)
            if mem.departure_baseline > 0.0
            else math.inf
        )
        norm_departed = mem.release_frame is not None and info.norm_distance >= departure_threshold
        motion_departed = mem.release_frame is not None and person_motion >= cfg.departure_motion_ratio
        departed_now = norm_departed or motion_departed
        mem.departed_flags.append(departed_now)
        mem.distances.append(info.distance)
        mem.norm_distances.append(info.norm_distance)
        mem.association_scores.append(info.score)

        mem.confidence_sum += info.bag_confidence
        mem.confidence_count += 1
        mem.max_confidence = max(mem.max_confidence, info.bag_confidence)

        if info.fallback:
            mem.fallback_frames += 1
            mem.fallback_gap_frames += 1
        else:
            mem.ever_yolo_confirmed = True
            if mem.fallback_gap_frames > 0:
                mem.yolo_reconfirmed_after_fallback = True
            mem.fallback_gap_frames = 0

        if info.other_person_closer:
            mem.other_person_closer_frames += 1
        if info.ambiguous:
            mem.ambiguous_frames += 1

        min_votes = max(2, int(math.ceil(cfg.smoothing_window / 2.0)))
        smooth_near = _smooth(mem.near_flags, min_votes)
        smooth_carried = _smooth(mem.carried_flags, min_votes)
        smooth_stationary = _smooth(mem.stationary_flags, min_votes)
        smooth_departed = _smooth(mem.departed_flags, min_votes)
        regrab_candidate = smooth_carried and not smooth_stationary
        mem.regrab_flags.append(regrab_candidate)
        smooth_regrab = _smooth(mem.regrab_flags, min_votes)

        if smooth_near:
            mem.near_frames += 1
        if smooth_carried:
            mem.carried_frames += 1
            if mem.carry_start_frame is None:
                mem.carry_start_frame = frame_index
                mem.carry_start_ts = timestamp
        if mem.release_frame is not None:
            if smooth_stationary:
                mem.stationary_frames += 1
                mem.release_frames += 1
            else:
                mem.stationary_frames = 0
            if smooth_departed:
                mem.departed_frames += 1
                gain_norm = info.norm_distance / max(mem.departure_baseline, 1e-6)
                gain_motion = person_motion / max(cfg.departure_motion_ratio, 1e-6)
                mem.max_departure_ratio = max(mem.max_departure_ratio, gain_norm, gain_motion)
            else:
                mem.departed_frames = 0

        distance_increasing = len(mem.distances) >= 2 and mem.distances[-1] > mem.distances[-2]

        if mem.emitted:
            return []

        if mem.state == EventState.NO_BAG:
            if smooth_near:
                mem.state = EventState.BAG_NEAR_PERSON

        if mem.state == EventState.BAG_NEAR_PERSON:
            if smooth_carried and mem.carried_frames >= cfg.min_carried_frames:
                mem.state = EventState.BAG_CARRIED
                mem.carry_start_frame = mem.carry_start_frame or frame_index
                mem.carry_start_ts = mem.carry_start_ts or timestamp

        if mem.state == EventState.BAG_CARRIED:
            if not smooth_carried and info.norm_distance >= cfg.release_distance_ratio and distance_increasing:
                mem.state = EventState.BAG_RELEASED
                mem.release_frame = frame_index
                mem.release_ts = timestamp
                mem.departure_baseline = max(1e-6, info.norm_distance)
                mem.release_person_centroid = info.person_centroid
                mem.release_person_height = info.person_height
                mem.max_departure_ratio = 0.0
                mem.stationary_frames = 0
                mem.departed_frames = 0

        if mem.state == EventState.BAG_RELEASED:
            if smooth_regrab:
                mem.state = EventState.BAG_CARRIED
                mem.release_frame = None
                mem.release_ts = None
                mem.stationary_frames = 0
                mem.departed_frames = 0
                mem.departure_baseline = 0.0
                mem.release_person_centroid = None
                mem.release_person_height = 0.0
                mem.max_departure_ratio = 0.0
            elif mem.stationary_frames >= cfg.min_stationary_frames:
                mem.state = EventState.BAG_ON_GROUND
                mem.ground_frame = frame_index
                mem.ground_ts = timestamp

        if mem.state == EventState.BAG_ON_GROUND:
            if smooth_regrab:
                mem.state = EventState.BAG_CARRIED
                mem.release_frame = None
                mem.release_ts = None
                mem.ground_frame = None
                mem.ground_ts = None
                mem.stationary_frames = 0
                mem.departed_frames = 0
                mem.abandonment_frames = 0
                mem.departure_baseline = 0.0
                mem.release_person_centroid = None
                mem.release_person_height = 0.0
                mem.max_departure_ratio = 0.0
            elif mem.departed_frames >= cfg.min_departed_frames and mem.max_departure_ratio >= 1.0:
                mem.state = EventState.PERSON_DEPARTED
                mem.departure_frame = frame_index
                mem.departure_ts = timestamp
                return self._evaluate_confirmation(mem)
            else:
                # Bag is on the ground, stationary, and not being re-grabbed.
                # The person need not walk far away for this to be littering:
                # leaving an object behind (and not reclaiming it) IS the
                # violation. Accumulate an abandonment timer and confirm once
                # it is clear the object was abandoned.
                if not smooth_carried:
                    mem.abandonment_frames += 1
                else:
                    mem.abandonment_frames = 0
                if mem.abandonment_frames >= cfg.min_abandonment_frames:
                    mem.state = EventState.PERSON_DEPARTED
                    mem.departure_frame = mem.departure_frame or frame_index
                    mem.departure_ts = mem.departure_ts or timestamp
                    return self._evaluate_confirmation(mem)

        if mem.state == EventState.PERSON_DEPARTED:
            if smooth_regrab:
                mem.state = EventState.BAG_CARRIED
                mem.release_frame = None
                mem.ground_frame = None
                mem.departure_frame = None
                mem.stationary_frames = 0
                mem.departed_frames = 0
                mem.departure_baseline = 0.0
                mem.release_person_centroid = None
                mem.release_person_height = 0.0
                mem.max_departure_ratio = 0.0
            else:
                return self._evaluate_confirmation(mem)

        return []

    def _compute_evidence(self, mem: _PairMemory) -> EventEvidence:
        cfg = self.config
        carry_score = min(1.0, mem.carried_frames / max(1, cfg.min_carried_frames))
        release_score = 1.0 if mem.release_frame is not None else 0.0
        stationary_score = min(1.0, mem.stationary_frames / max(1, cfg.min_stationary_frames))
        departure_score = min(1.0, mem.max_departure_ratio)
        # A bag left on the ground and not reclaimed is the departure-equivalent
        # proof of littering even when the person does not walk far away
        # (e.g. drops it at a bin). Credit the departure component in that case.
        if mem.stationary_frames >= cfg.min_stationary_frames and mem.state in (
            EventState.BAG_ON_GROUND,
            EventState.PERSON_DEPARTED,
        ):
            departure_score = 1.0
        # Abandonment: the object was released, grounded, and left un-reclaimed
        # for the abandonment window. That IS a genuine littering pattern. A
        # flaky per-frame stationarity signal must not erase it — credit BOTH
        # the grounded and the departed components so the confidence reflects
        # the real behavior instead of being capped artificially low.
        if mem.abandonment_frames >= cfg.min_abandonment_frames:
            departure_score = 1.0
            stationary_score = 1.0
        association_score = mem.mean_association_score()
        detection_score = min(1.0, mem.mean_confidence() / max(cfg.detection_high_conf, 1e-6))

        # The color/HSV fallback is a legitimate (lower-precision) detector for
        # the yellow waste-bag class that best.pt does NOT contain. It must NOT
        # be zeroed: that would structurally disable every event. We apply a
        # modest, transparent confidence discount and surface fallback_used.
        fallback_factor = 1.0
        if mem.fallback_frames > 0:
            fallback_factor = 0.95 if mem.yolo_reconfirmed_after_fallback else 0.90

        other_factor = 1.0
        if mem.other_person_closer_frames > cfg.max_other_person_closer_frames:
            other_factor = 0.0
        elif mem.other_person_closer_frames > 0:
            other_factor = max(0.50, 1.0 - (mem.other_person_closer_frames / float(cfg.max_other_person_closer_frames + 1)) * 0.5)

        base = (
            0.25 * carry_score
            + 0.15 * release_score
            + 0.25 * stationary_score
            + 0.25 * departure_score
            + 0.10 * association_score
        )
        confidence = max(0.0, min(1.0, base * detection_score * fallback_factor * other_factor))

        return EventEvidence(
            carry_score=round(carry_score, 4),
            release_score=round(release_score, 4),
            stationary_score=round(stationary_score, 4),
            departure_score=round(departure_score, 4),
            association_score=round(association_score, 4),
            detection_score=round(detection_score, 4),
            confidence=round(confidence, 4),
        )

    def _rejection_reason(self, mem: _PairMemory, evidence: EventEvidence) -> Optional[str]:
        cfg = self.config
        if mem.bag_seen_frames == 0:
            return RejectionReason.BAG_NOT_DETECTED.value
        if mem.mean_confidence() < cfg.detection_low_conf:
            return RejectionReason.LOW_BAG_CONFIDENCE.value
        if mem.person_seen_frames < cfg.min_carried_frames:
            return RejectionReason.PERSON_NOT_DETECTED.value
        if mem.carried_frames < cfg.min_carried_frames:
            return RejectionReason.NOT_ENOUGH_CARRIED_FRAMES.value
        if mem.release_frame is None:
            return RejectionReason.NO_RELEASE_TRANSITION.value
        if mem.stationary_frames < cfg.min_stationary_frames:
            # If the object never became stationary on the ground, littering is
            # not established; the person must at least have left the area.
            # Abandonment (grounded + un-reclaimed) is the departure-equivalent,
            # so it must not be reported as "person did not depart".
            if mem.abandonment_frames >= cfg.min_abandonment_frames:
                pass  # fall through to the confidence gate
            elif mem.departure_frame is None or mem.max_departure_ratio < 1.0:
                return RejectionReason.PERSON_DID_NOT_DEPART.value
            else:
                return RejectionReason.BAG_NOT_STATIONARY.value
        # Ground + stationary reached: abandonment (bag left behind, not
        # reclaimed) is a valid violation even without the person walking far.
        if mem.other_person_closer_frames > cfg.max_other_person_closer_frames:
            return RejectionReason.OTHER_PERSON_CLOSER.value
        if mem.ambiguous_frames > max(2, cfg.smoothing_window):
            return RejectionReason.ASSOCIATION_AMBIGUOUS.value
        if evidence.confidence < cfg.min_event_confidence:
            return RejectionReason.EVENT_CONFIDENCE_TOO_LOW.value
        return None

    def _evaluate_confirmation(self, mem: _PairMemory) -> List[LitteringEvent]:
        if mem.emitted:
            return []
        evidence = self._compute_evidence(mem)
        reason = self._rejection_reason(mem, evidence)
        if reason is None:
            mem.state = EventState.VIOLATION_CONFIRMED
            mem.confirmed_frame = mem.last_frame
            mem.confirmed_ts = mem.last_timestamp
            mem.emitted = True
            event = self._make_event(mem, evidence, confirmed=True, reason="VIOLATION_CONFIRMED")
            self.confirmed_events.append(event)
            return [event]

        if mem.departed_frames >= self.config.min_departed_frames + self.config.confirmation_grace_frames:
            mem.emitted = True
            event = self._make_event(mem, evidence, confirmed=False, reason=reason)
            self.rejected_events.append(event)
            return [event]
        return []

    def _finalize_pair(self, mem: _PairMemory, forced_reason: Optional[str] = None) -> List[LitteringEvent]:
        if mem.emitted:
            return []
        if mem.carried_frames <= 0:
            return []
        evidence = self._compute_evidence(mem)
        reason = forced_reason or self._rejection_reason(mem, evidence) or RejectionReason.EVENT_CONFIDENCE_TOO_LOW.value
        mem.emitted = True
        event = self._make_event(mem, evidence, confirmed=False, reason=reason)
        self.rejected_events.append(event)
        return [event]

    def _make_event(
        self,
        mem: _PairMemory,
        evidence: EventEvidence,
        confirmed: bool,
        reason: str,
    ) -> LitteringEvent:
        return LitteringEvent(
            event_id=uuid.uuid4().hex[:12],
            person_track_id=mem.person_id,
            bag_track_id=mem.bag_id,
            state=mem.state,
            confirmed=confirmed,
            reason=reason,
            confidence=evidence.confidence,
            evidence=evidence.to_dict(),
            frames={
                "carry_start": mem.carry_start_frame,
                "release": mem.release_frame,
                "ground": mem.ground_frame,
                "departure": mem.departure_frame,
                "confirmed": mem.confirmed_frame,
            },
            timestamps={
                "carry_start": mem.carry_start_ts,
                "release": mem.release_ts,
                "ground": mem.ground_ts,
                "departure": mem.departure_ts,
                "confirmed": mem.confirmed_ts,
            },
            bag_class=mem.bag_class,
            fallback_used=mem.fallback_frames > 0,
            yolo_reconfirmed=mem.yolo_reconfirmed_after_fallback or (mem.fallback_frames == 0),
            detector_source=("color_fallback" if mem.fallback_frames > 0 else "yolo"),
            other_person_closer=mem.other_person_closer_frames > self.config.max_other_person_closer_frames,
            details={
                "carried_frames": mem.carried_frames,
                "stationary_frames": mem.stationary_frames,
                "departed_frames": mem.departed_frames,
                "max_departure_ratio": round(mem.max_departure_ratio, 4),
                "mean_bag_confidence": round(mem.mean_confidence(), 4),
                "fallback_frames": mem.fallback_frames,
                "other_person_closer_frames": mem.other_person_closer_frames,
                "ambiguous_frames": mem.ambiguous_frames,
            },
        )


def annotate_detector_frame(
    frame: Any,
    persons: List[DetectorPerson],
    bags: List[DetectorBag],
    event: Optional[LitteringEvent] = None,
    detector: Optional[LitteringEventDetector] = None,
) -> Any:
    """
    Draw detector boxes and state labels on a BGR numpy frame.

    OpenCV is imported lazily so the detector remains importable in pure
    logic tests without CV dependencies.
    """
    import cv2  # type: ignore

    out = frame.copy()
    for p in persons:
        x1, y1, x2, y2 = [int(v) for v in p.bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"P{p.track_id}", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for b in bags:
        x1, y1, x2, y2 = [int(v) for v in b.bbox]
        color = (0, 165, 255) if b.source == "csrt_fallback" else (0, 255, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"B{b.track_id} {b.confidence:.2f}"
        if b.source == "csrt_fallback":
            label += " CSRT"
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    if detector is not None:
        y = 25
        for mem in list(detector._pairs.values())[:8]:
            text = f"P{mem.person_id}->B{mem.bag_id} {mem.state.value} C{mem.carried_frames} S{mem.stationary_frames} D{mem.departed_frames}"
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 16

    if event is not None:
        color = (0, 0, 255) if event.confirmed else (0, 165, 255)
        text = f"{'CONFIRMED' if event.confirmed else 'REJECTED'} {event.reason} conf={event.confidence:.2f}"
        cv2.putText(out, text, (10, out.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    return out


@dataclass
class _PendingEventEvidence:
    event: LitteringEvent
    directory: str
    event_timestamp: float
    pre_frame: Optional[Any] = None
    release_frame: Optional[Any] = None
    event_frame: Optional[Any] = None
    post_frame: Optional[Any] = None
    finalized: bool = False


class EventEvidenceCollector:
    """
    Small evidence writer for detector events.

    It stores recent annotated frames, then writes pre/release/event/post
    snapshots and a short annotated clip for confirmed events.
    """

    def __init__(
        self,
        store_root: str = os.path.join("evidence_store", "event_detector"),
        pre_seconds: float = 3.0,
        post_seconds: float = 3.0,
        max_frames: int = 120,
    ) -> None:
        self.store_root = store_root
        self.pre_seconds = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        self._frames: Deque[Tuple[float, int, Any]] = deque(maxlen=max_frames)
        self._pending: List[_PendingEventEvidence] = []
        self.completed_artifacts: List[Dict[str, Any]] = []

    def observe(self, timestamp: float, frame_index: int, annotated_frame: Any) -> None:
        self._frames.append((float(timestamp), int(frame_index), annotated_frame.copy()))

    def add_confirmed_event(self, event: LitteringEvent, annotated_frame: Any) -> str:
        directory = os.path.join(self.store_root, event.event_id)
        os.makedirs(directory, exist_ok=True)
        pending = _PendingEventEvidence(
            event=event,
            directory=directory,
            event_timestamp=float(event.timestamps.get("confirmed") or event.timestamps.get("departure") or 0.0),
            pre_frame=self._closest_frame(event.timestamps.get("carry_start")),
            release_frame=self._closest_frame(event.timestamps.get("release")),
            event_frame=annotated_frame.copy(),
        )
        self._write_frame_if_present(directory, "pre_event.jpg", pending.pre_frame)
        self._write_frame_if_present(directory, "release_event.jpg", pending.release_frame)
        self._write_frame_if_present(directory, "event.jpg", pending.event_frame)
        self._write_metadata(pending, finalized=False)
        self._pending.append(pending)
        return directory

    def finalize_due(self, timestamp: float, force: bool = False) -> List[Dict[str, Any]]:
        completed: List[Dict[str, Any]] = []
        remaining: List[_PendingEventEvidence] = []
        for pending in self._pending:
            due = force or timestamp >= pending.event_timestamp + self.post_seconds
            if not due:
                remaining.append(pending)
                continue
            pending.post_frame = self._closest_frame(pending.event_timestamp + self.post_seconds)
            self._write_frame_if_present(pending.directory, "post_event.jpg", pending.post_frame)
            self._write_clip(pending)
            pending.finalized = True
            artifact = {
                "event_id": pending.event.event_id,
                "directory": pending.directory,
                "pre_event": os.path.join(pending.directory, "pre_event.jpg"),
                "release_event": os.path.join(pending.directory, "release_event.jpg"),
                "event": os.path.join(pending.directory, "event.jpg"),
                "post_event": os.path.join(pending.directory, "post_event.jpg"),
                "clip": os.path.join(pending.directory, "event_clip.mp4"),
                "metadata": os.path.join(pending.directory, "metadata.json"),
            }
            self.completed_artifacts.append(artifact)
            completed.append(artifact)
        self._pending = remaining
        return completed

    def summary(self) -> Dict[str, Any]:
        return {
            "pending": len(self._pending),
            "completed": len(self.completed_artifacts),
            "store_root": self.store_root,
        }

    def _closest_frame(self, target_ts: Optional[float]) -> Optional[Any]:
        if target_ts is None or not self._frames:
            return None
        return min(self._frames, key=lambda item: abs(item[0] - float(target_ts)))[2]

    def _write_frame_if_present(self, directory: str, filename: str, frame: Any) -> None:
        if frame is None:
            return
        import cv2  # type: ignore

        path = os.path.join(directory, filename)
        cv2.imwrite(path, frame)

    def _write_clip(self, pending: _PendingEventEvidence) -> None:
        start = pending.event_timestamp - self.pre_seconds
        end = pending.event_timestamp + self.post_seconds
        frames = [f for ts, _, f in self._frames if start <= ts <= end]
        if not frames:
            frames = [f for f in (pending.pre_frame, pending.release_frame, pending.event_frame, pending.post_frame) if f is not None]
        if not frames:
            return
        import cv2  # type: ignore

        h, w = frames[0].shape[:2]
        path = os.path.join(pending.directory, "event_clip.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(5.0, self.pre_seconds + self.post_seconds), (w, h))
        if not writer.isOpened():
            return
        for frame in frames:
            writer.write(frame)
        writer.release()

    def _write_metadata(self, pending: _PendingEventEvidence, finalized: bool) -> None:
        data = {
            "event": pending.event.to_dict(),
            "finalized": finalized,
            "created_at": time.time(),
            "pre_seconds": self.pre_seconds,
            "post_seconds": self.post_seconds,
        }
        with open(os.path.join(pending.directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
