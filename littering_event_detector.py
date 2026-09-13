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

Known limitation (MASTER_REPAIR_PLAN P1-4 — documented, pinned by test)
-----------------------------------------------------------------------
The FSM enforces ONE bag per person: ``_select_primary_associations()``
picks a single primary object per person per tick, so a person carrying
TWO objects simultaneously can only ever generate events for one of them
(the other is structurally invisible to the event engine for that tick).
Cases D/E (two simultaneous objects per actor, one released while the
other stays carried) are therefore NOT supported by production. This is a
deliberate product limitation, not a bug — ``tests/test_correctness_audit.py``
pins the behavior so a future change cannot silently alter it.
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
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple


from inference.tracking.object_identity import ObjectIdentityManager
from inference.tracking.person_identity import PersonIdentityManager

CONFIG_PATH = os.path.join("config", "events.yaml")


class EventState(str, Enum):
    NO_BAG = "NO_BAG"
    BAG_NEAR_PERSON = "BAG_NEAR_PERSON"
    BAG_CARRIED = "BAG_CARRIED"
    BAG_RELEASED = "BAG_RELEASED"
    BAG_ON_GROUND = "BAG_ON_GROUND"
    PERSON_DEPARTED = "PERSON_DEPARTED"
    PICKED_BACK_UP = "PICKED_BACK_UP"
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
    PICKED_BACK_UP = "PICKED_BACK_UP"
    # Spec rule "BIN VS GROUND": the object was released and came to rest, but
    # the resting position was NEVER observed at/below the actor's ground line
    # (feet). It could be inside a trash container, on a ledge, or the ground
    # plane cannot be established — the location is ambiguous, so no confident
    # violation is emitted (require_ground_confirmation gate).
    NO_CONFIDENT_EVENT = "NO_CONFIDENT_EVENT"
    # P1-5: the resting position was observed inside an operator-configured
    # bin zone (static-camera polygon drawn once during setup), so the deposit
    # is a proper bin use, NOT ground littering.
    BIN_ZONE_DEPOSIT = "BIN_ZONE_DEPOSIT"
    # Mandatory spec distinction "BIN VS GROUND": the waste was deposited into
    # a container, dumpster, or trash receptacle (or disappeared into a bin opening
    # above the ground line) rather than released onto the open ground plane.
    BIN_DISPOSAL = "BIN_DISPOSAL"
    # Object never physically left the actor (clothing latch / hip-attached
    # color blob / held item). Not ground littering.
    NO_PHYSICAL_SEPARATION = "NO_PHYSICAL_SEPARATION"


@dataclass(frozen=True)
class BinZone:
    """Operator-configured bin zone for a STATIC camera (P1-5).

    ``polygon`` is a closed polygon in NORMALIZED frame coordinates
    (x, y each in [0, 1], origin top-left) so the zone survives resolution
    changes. Drawn once per camera during setup; ``camera_id`` selects which
    camera the zone belongs to (``None``/omitted = applies to every camera).
    """

    polygon: Tuple[Tuple[float, float], ...] = ()
    camera_id: Optional[str] = None
    name: str = "bin"

    @staticmethod
    def _coerce(entry: Any) -> "BinZone":
        """Accept a dict (YAML/config) or a BinZone; polygons may also be
        given as ``bbox: [x1, y1, x2, y2]`` (normalized) for convenience."""
        if isinstance(entry, BinZone):
            return entry
        if not isinstance(entry, dict):
            raise ValueError(f"bin_zones entry must be a mapping, got {type(entry)!r}")
        poly = entry.get("polygon")
        if poly is None and entry.get("bbox") is not None:
            x1, y1, x2, y2 = (float(v) for v in entry["bbox"])
            poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        polygon = tuple((float(p[0]), float(p[1])) for p in (poly or ()))
        if len(polygon) < 3:
            raise ValueError("bin zone polygon needs at least 3 points")
        camera = entry.get("camera_id")
        return BinZone(
            polygon=polygon,
            camera_id=None if camera is None else str(camera),
            name=str(entry.get("name") or "bin"),
        )

    def contains_normalized(self, x: float, y: float) -> bool:
        """Ray-casting point-in-polygon test on normalized coordinates."""
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if (yi > y) != (yj > y):
                x_cross = (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi
                if x < x_cross:
                    inside = not inside
            j = i
        return inside


# P1-8 (audit): the config fields the adaptive tuner (adaptive_tuner.py)
# may learn/override via LEARNING_STEPS + TIER_SPECS. Every confirmed or
# rejected event discloses the EFFECTIVE runtime values of exactly these
# fields in details["active_thresholds"], so an operator can see which
# thresholds actually produced the decision without guessing the YAML.
TUNABLE_THRESHOLD_KEYS: tuple = (
    "min_carried_frames",
    "smoothing_window",
    "release_distance_ratio",
    "release_distance_floor",
    "release_window_frames",
    "feet_release_frames",
    "departure_motion_ratio",
    "departure_distance_ratio",
    "min_abandonment_frames",
    "stationary_max_pixel_step",
    "stationary_grace_frames",
    "min_stationary_frames",
    "max_pair_age_frames",
    "max_fallback_tracker_gap_frames",
    "min_event_confidence",
    "min_departed_frames",
    "confirmation_grace_frames",
)


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
    # How many consecutive non-stationary ticks the stationarity counter can
    # absorb before it resets to zero. Single-tick detection flicker (HSV mask
    # jitter, tracker id churn) previously reset an 8-tick stationary streak
    # back to zero, making BAG_ON_GROUND nearly unreachable on real footage.
    stationary_grace_frames: int = 3

    # ------------------------------------------------------------------ #
    # Release robustness. The legacy release required, on ONE tick:
    # carried-flipped-false AND norm_distance >= release_distance_ratio AND
    # strictly-increasing distance. Real throws violate each conjunct: the
    # bag can be put down at the person's feet (distance never reaches the
    # threshold), distance can dip for a tick mid-throw (tracking jitter),
    # and the person can walk PAST the dropped bag so the 2D centroid
    # distance DECREASES while they depart in 3D. The additions below make
    # release a short temporal window instead of a single tick.
    #
    # Ticks sampled for the windowed distance-growth test.
    release_window_frames: int = 4
    # Consecutive ticks of "bag on the ground at the person's feet, hand not
    # near, person moving" before a feet-put-down release is declared.
    feet_release_frames: int = 3
    # A person is considered "moving" when their per-tick centroid step
    # exceeds this many pixels (calibrated per video when auto_calibrate).
    carry_motion_norm_px: float = 8.0
    # A carried object must share its carrier's motion: when the person is
    # moving, the bag's per-tick step must be at least this fraction of the
    # person's step, otherwise the object is NOT being carried (it is a
    # static object the person is merely passing). This is the gate that
    # separates "holding a bag" from "walking next to a bag on the ground".
    carry_motion_sync_ratio: float = 0.4
    # Anti-furniture gate: a release (static-bag / feet variant) is only
    # trusted if the object was at some point genuinely handled — it either
    # moved by at least this fraction of the person's height since the pair
    # first saw it, or a wrist was ever near it.
    bag_move_norm: float = 0.05
    # Per-pair adaptive release distance: while carried, the threshold is
    # clamped between this floor and ``release_distance_ratio``.
    release_distance_floor: float = 0.15
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

    # ------------------------------------------------------------------ #
    # BIN VS GROUND gate (spec: "Is the final location actually ground rather
    # than inside the bin?"). While True, a candidate is only CONFIRMED when
    # the resting object was observed at/below the actor's feet line (the
    # ground plane at the actor) at least once between release and
    # confirmation. A release whose resting position was never on the ground
    # plane (possible bin deposit / ledge) is ambiguous -> NO_CONFIDENT_EVENT.
    # The per-event status is ALWAYS attached to details["location_status"];
    # this flag only controls whether ambiguity blocks confirmation.
    # NOTE: no bin-detection source exists yet, so ambiguity = "resting
    # position never observed on the actor's ground plane".
    # P1-7 (audit): the dataclass default now MATCHES production
    # (config/events.yaml sets require_ground_confirmation: true). The old
    # False default made bare `EventDetectorConfig()` construction — e.g. in
    # forensic replays and ad-hoc scripts — silently run with the bin/ground
    # gate OFF, diverging from every production run. Tests that want the
    # legacy no-gate behaviour set it explicitly.
    require_ground_confirmation: bool = True
    # Ground-plane margin for the BIN VS GROUND evidence, in person-heights:
    # the resting object's centroid counts as "on the actor's ground plane"
    # when it is within this margin ABOVE the actor's feet line. Real camera
    # perspective projects a bag resting on the ground BEHIND the actor well
    # above their feet line (measured on IMG_5305: the real confirmed event's
    # bag never crossed the strict 0.05·ph feet line in 15 post-release
    # ticks), while a bin deposit typically rests ~0.5-0.8·ph above the feet
    # line. 0.40 separates the two without rejecting real ground put-downs.
    ground_plane_margin_ratio: float = 0.40

    # P1-5 — operator-configured bin zones for static cameras. Each zone is a
    # normalized polygon (optionally per camera_id, see BinZone). While the
    # require_ground_confirmation gate is on, an object resting inside a bin
    # zone provides NO ground-litter evidence: the resting location is a bin
    # deposit, so the candidate is rejected as BIN_ZONE_DEPOSIT instead of
    # being confirmed as ground littering. Empty by default (no zones
    # configured -> behaviour identical to before).
    bin_zones: Tuple[BinZone, ...] = ()

    def __post_init__(self) -> None:
        # Accept raw YAML dicts (from_dict passes lists straight through).
        if self.bin_zones and not all(isinstance(z, BinZone) for z in self.bin_zones):
            self.bin_zones = tuple(BinZone._coerce(z) for z in self.bin_zones)

    # ------------------------------------------------------------------ #
    # Per-video auto-calibration. During the first ``calibration_frames``
    # analysis ticks the detector measures the scene (median person height,
    # median person walking step) and adapts the motion thresholds. This
    # removes the per-camera hand-tuning this module used to require: a
    # slow-ambling scene and a fast-walking scene get different departure /
    # motion-synchrony thresholds, with hard clamps so no single video can
    # push the thresholds into nonsense.
    auto_calibrate: bool = True
    calibration_frames: int = 24

    # 2026-09-10 baseline regression fix: 0.80 gated out every real event
    # (evidence.confidence tops out ~0.6-0.7 on genuine footage). 0.50 is the
    # resilient operational baseline (0.45-0.50 band).
    min_event_confidence: float = 0.50
    # 2026-09-10: 16 analysis-ticks (~2 s @8 fps) rejected real events after
    # brief occlusions; 40 ticks (~5 s) tolerates transient visual occlusions.
    max_fallback_tracker_gap_frames: int = 40
    # Static ground-clutter suppression: a bag whose confidence is below this
    # AND that is not currently near/carried AND that sits on the ground plane
    # never enters FSM pair memory (e.g. a stationary NESCAFE tin on the floor).
    min_carry_origin_confidence: float = 0.40
    # CARRY-ORIGIN CONSTRAINT: when wrist pose keypoints are available, an
    # object can only transition to CARRIED if the person's wrist/hand was
    # spatially linked to it (ever_wrist_near). Falls back to geometric
    # evidence (containment / motion-synchrony) when no pose is available.
    require_carry_origin_link: bool = True
    max_pair_age_frames: int = 30
    max_other_person_closer_frames: int = 3

    smoothing_window: int = 5
    association_margin: float = 0.15
    carrying_zone_vertical_start: float = 0.45
    carrying_zone_horizontal_margin: float = 0.75

    # ------------------------------------------------------------------ #
    # Phase 2 — pose-based association score.
    #
    # The legacy association score is purely geometric (centroid proximity +
    # box containment + detection confidence). It cannot tell "person is
    # holding the bag" from "person happens to be standing next to a bag on
    # the ground". The pose score adds three body-relative cues and is BLENDED
    # with the legacy score, never replacing it outright:
    #
    #     score = (1 - blend) * legacy_score + blend * pose_score
    #
    # ``pose_association_enabled = False`` (and/or ``pose_score_blend = 0.0``)
    # reproduces the exact pre-Phase-2 behaviour.
    pose_association_enabled: bool = False
    pose_score_blend: float = 0.0        # 0.0 = legacy only, 1.0 = pose only
    pose_weight_wrist: float = 0.4
    pose_weight_zone: float = 0.3
    pose_weight_motion: float = 0.3
    # wrist distance normaliser, in person-heights. 1/(1 + d/(ph*this)).
    pose_wrist_norm_ratio: float = 0.5
    # carrying-zone horizontal margin as a fraction of person bbox width.
    pose_zone_horizontal_margin: float = 0.30
    # vertical band: shoulder line .. bbox bottom. Both margins are 0.0, so the
    # zone is exactly "from the shoulders down to the bottom of the person's
    # bounding box" (the Phase-2 redefinition forced by MoveNet lacking knees).
    pose_zone_vertical_above: float = 0.0
    pose_zone_vertical_below: float = 0.0
    # motion-similarity normaliser, in pixels of velocity difference per tick.
    pose_motion_norm_px: float = 20.0
    # how many analysis ticks of centroid history motion similarity averages over
    pose_motion_window: int = 3

    # Per-pair adaptive release distance: threshold = clamp(
    # release_distance_adaptive_ratio * max_carry_norm_distance,
    # release_distance_floor, release_distance_ratio)
    release_distance_adaptive_ratio: float = 1.6

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventDetectorConfig":
        allowed = {f.name for f in fields(cls)}
        normalized: Dict[str, Any] = {}
        # P2-1 (audit): NOVELTY_* keys are NOT event-detector config — they are
        # consumed separately by inference/detection/novelty_detector.py
        # (NoveltyConfig.from_yaml). Anything ELSE that does not map to a
        # dataclass field is silently ignored today; that turns a config typo
        # into a silent no-op, so surface it loudly instead.
        for key, value in data.items():
            name = str(key).lower()
            if name in allowed:
                normalized[name] = value
                continue
            if str(key).upper().startswith("NOVELTY_"):
                continue
            try:
                import logging

                logging.getLogger(__name__).warning(
                    "config/events.yaml: key %r does not match any "
                    "EventDetectorConfig field and was IGNORED (typo?)",
                    key,
                )
            except Exception:
                pass
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
    """Load detector config from YAML, falling back to defaults.

    Cross-video learning is NOT applied here: adaptive_tuner.LearningStore
    (learning/learning.json, fed by every analyzed video) owns it and injects
    it through the tier ladder / learned overrides, with hard clamps.
    """
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
    # Shoulders are needed by the Phase-2 pose association score: the carrying
    # zone is anchored on the shoulder line. MoveNet exposes them, but the
    # pipeline used to drop them here.
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None


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
    # P2-2: live sources are "yolo" | "color" | "novelty" (pipeline mapping).
    # The legacy "csrt_fallback" string is retired — no live code assigns it.
    # Phase B: NON-yolo sources are proposals and NEVER form event pairs, so
    # the pair-level fallback discount below is currently DORMANT (kept as
    # defense-in-depth; re-enabling a real fallback-confirmation path is a
    # Layer-2 change requiring explicit approval).
    source: str = "yolo"
    yolo_confirmed: bool = True
    object_uid: Optional[int] = None  # stable logical id across track-id churn (Phase B)


def _is_semantic_waste(bag: DetectorBag) -> bool:
    """Layer 1 semantic admission (REPAIR-P0-02).

    SEMANTIC_WASTE (may drive pairs / littering events):
      - source == "yolo" with a real class (not color_candidate* / detected_object)
      - source == "color" with a real class — discounted HSV path for bags
        YOLO weights miss (e.g. yellow waste bag). Confidence is reduced via
        fallback_frames in _compute_evidence.

    PROPOSAL-ONLY (telemetry / debug render; NEVER event objects alone):
      - source == "novelty"
      - class detected_object
      - class color_candidate*
    """
    src = str(getattr(bag, "source", "yolo") or "yolo").lower()
    cls = str(getattr(bag, "class_name", "") or "").lower()
    if cls.startswith("color_candidate"):
        return False
    if cls == "detected_object":
        return False
    if src == "novelty":
        return False
    if src in ("yolo", "color"):
        return True
    return False


def _keypoints_from_any(kp: Any) -> Optional[DetectorKeypoints]:
    if kp is None:
        return None
    return DetectorKeypoints(
        left_wrist=getattr(kp, "left_wrist", None),
        right_wrist=getattr(kp, "right_wrist", None),
        torso_center=getattr(kp, "torso_center", None),
        left_shoulder=getattr(kp, "left_shoulder", None),
        right_shoulder=getattr(kp, "right_shoulder", None),
    )


def _median(values: List[float]) -> float:
    """Median of a sample list; 0.0 when the sample is empty."""
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


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
        object_uid=getattr(track, "object_uid", None),
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
    bag_uid: Optional[int] = None  # stable logical object id across churn (Phase B)
    # Phase C — authoritative event-actor record (spec rules #11/#17). Frozen at
    # the carry transition; every evidence artifact must reference these exact IDs
    # so a passing person/car can NEVER inherit this event.
    event_actor_person_track_id: Optional[int] = None
    event_actor_person_uid: Optional[int] = None  # STABLE person uid (authoritative actor)
    event_object_track_id: Optional[int] = None
    event_object_uid: Optional[int] = None
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
    person_uid: int = 0  # stable logical person identity across track-id switches
    bag_uid: Optional[int] = None  # stable logical object id (Phase B)
    is_primary: bool = False
    ambiguous: bool = False
    other_person_closer: bool = False
    # Phase 2 — pose-based association. None when no pose cue was available
    # for this pair this frame (no keypoints yet / no motion history yet).
    pose_score: Optional[float] = None
    pose_components: Dict[str, Optional[float]] = field(default_factory=dict)
    legacy_score: float = 0.0
    # bag centroid this frame (for the pair's first-sighting displacement check).
    bag_centroid: Optional[Tuple[float, float]] = None
    # bag centre at/below the person's feet line (strict ground line, used by
    # the carry / feet-put-down logic).
    bag_below_feet: bool = False
    # bag centre within ground_plane_margin_ratio ABOVE the feet line (relaxed
    # BIN VS GROUND ground-plane evidence flag).
    near_ground_plane: bool = False
    # person's per-tick step exceeded carry_motion_norm_px.
    person_moving: bool = False
    # when the person is moving, the bag shares the motion (carry evidence).
    moves_with_person: bool = True
    wrist_near: bool = False
    # Clutter/origin gating: wrist keypoints were actually available for this
    # person this tick (lets the FSM distinguish "wrist far" from "no pose").
    wrist_keypoints_available: bool = False
    # P1-5: the bag centroid is inside an operator-configured bin zone this
    # tick (static camera). Suppresses ground-litter evidence.
    in_bin_zone: bool = False


@dataclass
class _PairMemory:
    person_id: int
    bag_id: int
    person_uid: int = 0  # stable logical person identity across track-id switches
    bag_uid: Optional[int] = None  # stable logical object id across churn (Phase B)
    # Phase C — authoritative IDs FROZEN at the carry transition, i.e. once the
    # continuous temporal person<->object association is first established. These
    # are the IDs every piece of evidence must reference (spec rules #11/#17): the
    # actor can NEVER be swapped for a passer-by, and the object id is the one
    # actually carried, not a later churned id.
    carry_person_track_id: Optional[int] = None
    carry_person_uid: Optional[int] = None  # STABLE person uid frozen at carry (authoritative actor)
    carry_object_track_id: Optional[int] = None
    carry_object_uid: Optional[int] = None
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
    # BIN VS GROUND evidence: analysis ticks between release and confirmation
    # where the resting object was observed at/below the ACTOR's feet line
    # (the ground plane at the actor). > 0 => GROUND_CONFIRMED, else the
    # resting location is ambiguous (possible bin deposit / ledge).
    ground_evidence_frames: int = 0
    # P1-5: post-release ticks in which the resting object's centroid was
    # inside an operator-configured bin zone. > 0 with ground_evidence_frames
    # == 0 means the deposit happened INSIDE a bin -> BIN_ZONE_DEPOSIT.
    bin_zone_frames: int = 0
    release_person_centroid: Optional[Tuple[float, float]] = None
    release_person_height: float = 0.0
    # Bag centroid at the moment of release — used to detect a true pick-up lift.
    release_bag_centroid: Optional[Tuple[float, float]] = None
    # Deepest bag centroid observed when first entering BAG_ON_GROUND.
    ground_bag_centroid: Optional[Tuple[float, float]] = None
    bag_class: str = "trash_bag"
    emitted: bool = False

    # --- release-robustness state ------------------------------------- #
    # Consecutive non-stationary ticks seen since the last stationary tick
    # (stationarity grace counter).
    stationary_miss_streak: int = 0
    # Consecutive ticks matching the feet-put-down pattern.
    feet_release_streak: int = 0
    # Consecutive ticks matching motion-desync put-down (bag left behind).
    desync_release_streak: int = 0
    # Consecutive stationary ticks while still in BAG_CARRIED after a walking
    # carry — real put-downs often never enter near_ground_plane (perspective).
    carried_stationary_streak: int = 0
    # Frames after release where the bag is no longer motion-synced — allows
    # BAG_ON_GROUND without the strict pixel-stationary gate (jittery tracks).
    post_release_settle_streak: int = 0
    # Consecutive post-ground lift+wrist ticks before accepting a reclaim.
    # Short centroid flicker after put-down must not undo littering (IMG_5290).
    regrab_lift_streak: int = 0
    # First bag centroid this pair ever saw — used by the anti-furniture gate
    # (an object that NEVER moved and was never wrist-near is furniture).
    first_bag_centroid: Optional[Tuple[float, float]] = None
    ever_wrist_near: bool = False
    # CARRY-ORIGIN evidence (clutter suppression): accumulated handling cues.
    ever_contained: bool = False            # bag bbox ever >=25% inside person bbox
    ever_moved_with_person: bool = False    # bag ever moved in sync while carried
    pose_ever_available: bool = False       # wrist keypoints ever present
    # True once the bag was carried above the shared ground plane — required
    # before crit_ground may fire (avoids false release for a low stationary hold).
    ever_off_ground_while_carried: bool = False
    # True once the person walked while this pair was carried — arms the
    # desync put-down criterion for bags that stay in the ground-plane band
    # for the entire carry (common with low handheld bags on iPhone footage).
    ever_person_moved_while_carried: bool = False
    # Largest norm_distance observed while actually carried — the reference
    # for the per-pair adaptive release threshold.
    max_carry_norm_distance: float = 0.0
    # Adaptive release threshold, updated while carried.
    release_distance_threshold: float = 0.0
    # Post-release physical separation evidence (clothing / held-item rejection).
    # max_post_release_norm_distance: peak centroid separation after release.
    # separated_frames: ticks after release with low containment OR bag outside
    # the person box (true discard, not a hip-attached color latch).
    max_post_release_norm_distance: float = 0.0
    separated_frames: int = 0
    # Peak bag displacement (pixels) from first_bag_centroid — clothing /
    # static clutter associations barely move the "object" while the person
    # walks away and inflates norm_distance.
    max_bag_displacement_px: float = 0.0

    near_flags: Deque[bool] = field(default_factory=deque)
    carried_flags: Deque[bool] = field(default_factory=deque)
    stationary_flags: Deque[bool] = field(default_factory=deque)
    departed_flags: Deque[bool] = field(default_factory=deque)
    regrab_flags: Deque[bool] = field(default_factory=deque)
    # Set True whenever a sustained regrab (bag reclaimed after being released /
    # grounded) reverts the pair to BAG_CARRIED. Used to explicitly reclassify the
    # pair as PICKED_BACK_UP (NOT_ABANDONED) at finalization instead of emitting a
    # generic NO_RELEASE_TRANSITION rejection.
    reclaimed: bool = False
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
    lower_end = py2 - ph * 0.05
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


# ---------------------------------------------------------------------- #
# Phase 2 — pose-based association score
# ---------------------------------------------------------------------- #
def _shoulder_line(person: DetectorPerson) -> Optional[float]:
    """
    Y coordinate of the shoulder line, or None if unavailable.

    DEVIATION FROM THE ORIGINAL PLAN: the brief specified the carrying zone as
    the vertical band "between the shoulders and the knees". MoveNet (our pose
    backend) does not expose knees -- it only returns nose, shoulders, wrists
    and a derived torso centre. The band is therefore anchored on the shoulder
    line at the top and on the person's bbox bottom at the bottom, which is the
    closest available proxy for "somewhere on/near the body below the arms".
    Falls back to the torso centre, then to a fixed fraction of the bbox.
    """
    kp = person.keypoints
    if kp is not None:
        ys = [s[1] for s in (kp.left_shoulder, kp.right_shoulder) if s is not None]
        if ys:
            return sum(ys) / len(ys)
        if kp.torso_center is not None:
            return float(kp.torso_center[1])
    return None


def _mean_step(history: Sequence[Tuple[float, float]], window: int) -> Optional[float]:
    """Mean per-tick centroid displacement over the last ``window`` samples.

    Deliberately expressed in PIXELS PER ANALYSIS TICK rather than pixels per
    second: production never populates ``DetectorBag.timestamp`` (it is always
    0.0), so any per-second velocity would divide by a near-zero dt. Per-tick
    displacement is well defined and comparable between person and bag.
    """
    if history is None or len(history) < 2:
        return None
    n = min(max(2, int(window)), len(history))
    recent = list(history)[-n:]
    steps = [
        _dist(recent[i - 1], recent[i]) for i in range(1, len(recent))
    ]
    if not steps:
        return None
    return sum(steps) / len(steps)


def _pose_association_score(
    person: DetectorPerson,
    bag: DetectorBag,
    person_history: Optional[Sequence[Tuple[float, float]]],
    bag_history: Optional[Sequence[Tuple[float, float]]],
    cfg: EventDetectorConfig,
) -> Tuple[Optional[float], Dict[str, Optional[float]]]:
    """
    Body-relative association evidence for one (person, bag) pair.

    Returns ``(score, components)`` in [0, 1]. ``score`` is None when NO pose
    cue was computable at all, which tells the caller to fall back to the
    legacy geometric score entirely.

    Components
    ----------
    wrist   : proximity of the bag centroid to the nearer wrist — the hand is
              the acting agent, so this is weighted highest.
    zone    : binary — is the bag inside the body-relative carrying zone
              (shoulder line .. bbox bottom, body width + margin)?
    motion  : do the person and the bag move together? A carried object shares
              its carrier's velocity; a background object does not.

    When only some components are available (e.g. wrists occluded but shoulders
    visible) the remaining weights are renormalised over the available subset
    rather than zeroing the missing ones — a person with hidden hands still
    yields a usable zone and motion score. This is the documented fallback for
    occlusion; see the Phase 2 report.
    """
    ph = _box_height(person.bbox)
    pw = _box_width(person.bbox)
    px1, py1, px2, py2 = person.bbox
    bc = _centroid(bag.bbox)
    kp = person.keypoints

    parts: Dict[str, Optional[float]] = {"wrist": None, "zone": None, "motion": None}

    # --- 1. wrist proximity ------------------------------------------------
    if kp is not None:
        wrists = [w for w in (kp.left_wrist, kp.right_wrist) if w is not None]
        if wrists:
            d = min(_dist(w, bc) for w in wrists)
            nd = d / max(1e-6, ph * max(1e-6, cfg.pose_wrist_norm_ratio))
            parts["wrist"] = 1.0 / (1.0 + nd)

    # --- 2. carrying zone (binary, shoulder line .. bbox bottom) -----------
    top = _shoulder_line(person)
    if top is None:
        top = py1 + ph * cfg.carrying_zone_vertical_start
    band_top = top - ph * cfg.pose_zone_vertical_above
    band_bottom = py2 + ph * cfg.pose_zone_vertical_below
    left = px1 - pw * cfg.pose_zone_horizontal_margin
    right = px2 + pw * cfg.pose_zone_horizontal_margin
    in_zone = (band_top <= bc[1] <= band_bottom) and (left <= bc[0] <= right)
    parts["zone"] = 1.0 if in_zone else 0.0

    # --- 3. motion similarity ---------------------------------------------
    p_step = _mean_step(person_history, cfg.pose_motion_window)
    b_step = _mean_step(bag_history, cfg.pose_motion_window)
    if p_step is not None and b_step is not None:
        diff = abs(p_step - b_step)
        parts["motion"] = 1.0 / (1.0 + diff / max(1e-6, cfg.pose_motion_norm_px))

    # --- weighted blend with graceful degradation --------------------------
    weights = {
        "wrist": cfg.pose_weight_wrist,
        "zone": cfg.pose_weight_zone,
        "motion": cfg.pose_weight_motion,
    }
    total_w = sum(weights[k] for k, v in parts.items() if v is not None)
    if total_w <= 0.0:
        return None, parts
    score = sum(weights[k] * parts[k] for k, v in parts.items() if v is not None) / total_w
    return max(0.0, min(1.0, score)), parts


class LitteringEventDetector:
    """
    Stateful detector for temporal littering events.

    Usage:
        detector = LitteringEventDetector(load_event_config())
        for frame_index, timestamp in ...:
            events = detector.update(persons, bags, timestamp, frame_index)
        events.extend(detector.finalize())
    """

    def __init__(
        self,
        config: Optional[EventDetectorConfig] = None,
        camera_id: Optional[str] = None,
    ) -> None:
        self.config = config or EventDetectorConfig()
        # P1-5: bin zones are configured per camera; the pipeline passes the
        # camera identity so only this camera's zones (plus camera-agnostic
        # ones) apply here.
        self.camera_id = camera_id
        self._active_bin_zones: Tuple[BinZone, ...] = tuple(
            z for z in self.config.bin_zones
            if z.camera_id is None or z.camera_id == camera_id
        )
        # P1-5: frame size (w, h) in pixels, needed to normalize detections
        # against the normalized zone polygons. Optional — when the caller
        # does not provide it, zone membership cannot be evaluated.
        self._frame_size: Optional[Tuple[int, int]] = None
        self._pairs: Dict[Tuple[int, int], _PairMemory] = {}
        self._bag_history: Dict[int, Deque[Tuple[float, Tuple[float, float]]]] = {}
        # Phase 2: rolling person centroids, for the motion-similarity cue.
        # Keyed by person track id; holds raw pixel centroids, no timestamps
        # (see _mean_step for why velocity is expressed per tick).
        self._person_history: Dict[int, Deque[Tuple[float, float]]] = {}
        self._frame_index = 0
        self._last_timestamp = 0.0
        self.confirmed_events: List[LitteringEvent] = []
        self.rejected_events: List[LitteringEvent] = []
        # Monotonic counter for STABLE logical object identity (Phase B root-cause
        # fix). A fresh uid is allocated ONCE per physical object carried by a
        # person and reused for the whole carry->release->ground arc, so the
        # colour/ByteTrack track-id churn never splits one physical bag into many
        # identities (spec rule #11).
        self._bag_uid_counter = 0
        # person_id -> stable object uid. A littering actor carries one bag, so one
        # stable uid per person keeps the pair + evidence anchored to the same
        # physical object across the entire event.
        self._person_obj_uid: Dict[int, int] = {}
        # Stable person-identity manager: resolves raw ByteTrack person track ids
        # to stable logical person uids so a track-id switch (P2->P7) in a crowd
        # does NOT orphan the pair or hand the event to a bystander (Layer-2 spec:
        # raw tracker ids are NOT authoritative event identity).
        self._person_identity = PersonIdentityManager()
        # Stable OBJECT identity: spatial/temporal nearest-neighbour matching
        # across tracker-id churn (the 807-physical / 1036-switch Layer-1
        # reality). This is the AUTHORITATIVE object-uid source: one physical
        # object keeps one uid across short detector gaps, and two distinct
        # physical objects never share one. The per-person counter below is
        # only the fallback for callers that provide no object uid.
        self._object_identity = ObjectIdentityManager()
        # Diagnostics of the LAST update() tick (raw track id -> stable uid).
        # Consumed by the real-video identity audit / evidence tooling; not
        # used in decision logic.
        self.last_person_uid_map: Dict[int, int] = {}
        self.last_object_uid_map: Dict[int, Optional[int]] = {}
        self.last_proposal_count: int = 0
        self.last_semantic_count: int = 0
        # Per-video auto-calibration state (see EventDetectorConfig.auto_calibrate).
        self._calibration_start = 0
        self._person_height_samples: List[float] = []
        self._person_step_samples: List[float] = []
        self._calibrated = False
        # What the auto-calibrator changed for THIS video (exposed in reports).
        self.calibration_report: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(
        self,
        persons: List[DetectorPerson],
        bags: List[DetectorBag],
        timestamp: float,
        frame_index: Optional[int] = None,
        frame_size: Optional[Tuple[int, int]] = None,
    ) -> List[LitteringEvent]:
        cfg = self.config
        if frame_size is not None:
            # P1-5: (width, height) of the current frame, for bin-zone tests.
            self._frame_size = (int(frame_size[0]), int(frame_size[1]))
        if frame_index is None:
            frame_index = self._frame_index + 1
        self._frame_index = int(frame_index)
        self._last_timestamp = float(timestamp)

        valid_bags = [b for b in bags if float(b.confidence) >= cfg.detection_low_conf]
        # Phase B: strict semantic split — only SEMANTIC_WASTE (source==yolo,
        # non-candidate) may drive temporal events. Proposals are kept for
        # debug rendering but never enter pair evaluation (spec: color/novelty
        # proposals alone must never become WASTE events).
        semantic_bags = [b for b in valid_bags if _is_semantic_waste(b)]
        proposal_bags = [b for b in valid_bags if not _is_semantic_waste(b)]
        # History + object identity operate on semantic bags only, so a flood of
        # proposals cannot create stable UIDs or pollute the temporal state.
        self._update_bag_history(semantic_bags)

        # --- Assign STABLE object uids (BEFORE pair evaluation) -------------
        # Spatial/temporal nearest-neighbour matching across tracker-id churn.
        # One physical object keeps one uid across short detector gaps; two
        # distinct physical objects never merge (per-frame reservation +
        # class-family + distance gate).
        if semantic_bags:
            obj_assignments = self._object_identity.update(
                [
                    (str(b.class_name), tuple(float(v) for v in b.bbox), _centroid(b.bbox))
                    for b in semantic_bags
                ]
            )
            for i, b in enumerate(semantic_bags):
                uid = obj_assignments.get(i)
                if uid is not None:
                    b.object_uid = uid

        # --- Resolve raw person track ids -> stable person uids (BEFORE history) -
        # A track-id switch (ByteTrack re-assigns P2 -> P7) must NOT orphan the
        # (person, object) pair. We map every raw id to a stable uid here; the
        # person centroid history + pairs are then keyed by the STABLE uid, so a
        # switch preserves motion history and pair continuity. Raw track id is
        # retained in info.person_id for the frozen carry_person_track_id /
        # emitted person_track_id (legacy reference only — NOT authoritative).
        person_uid_map: Dict[int, int] = {}
        for p in persons:
            if float(p.confidence) < cfg.detection_low_conf:
                continue
            pc = _centroid(p.bbox)
            uid = self._person_identity.resolve(
                int(p.track_id), pc, p.bbox,
                getattr(p, "keypoints", None), self._frame_index, timestamp)
            person_uid_map[int(p.track_id)] = uid

        # Person centroid history is needed EVERY tick now: the motion-sync
        # carry cue and the auto-calibrator both read it, not just the
        # (optional) pose association score. Keyed by STABLE uid.
        self._update_person_history(persons, person_uid_map)
        if cfg.auto_calibrate and self._frame_index - self._calibration_start < cfg.calibration_frames:
            self._calibrate(persons)

        infos: List[_PairInfo] = []
        for person in persons:
            if float(person.confidence) < cfg.detection_low_conf:
                continue
            for bag in semantic_bags:
                info = self._evaluate_pair(person, bag, timestamp, frame_index)
                if info is not None:
                    info.person_uid = person_uid_map.get(int(info.person_id), int(info.person_id))
                    infos.append(info)

        primary_infos = self._select_primary_associations(infos)

        active_keys: set = set()
        emitted: List[LitteringEvent] = []
        current_person_uids = set(person_uid_map.values())
        current_bag_ids = {int(b.track_id) for b in semantic_bags}

        # --- Rebind orphaned pairs across color-tracker ID churn ------------
        # Pairs are now keyed by a STABLE per-person object uid, so a churned
        # track id does not split one physical bag into many identities. When a
        # pair's current track id vanishes but the actor is still present and a
        # valid primary association exists, we simply refresh mem.bag_id to the
        # new track id (the stable key is unchanged, so all temporal evidence is
        # preserved).
        rebound_bag_ids: set = set()
        rebound_info_keys: set = set()
        for key, mem in list(self._pairs.items()):
            if key in active_keys:
                continue
            if mem.bag_id in current_bag_ids:
                continue  # bag still present under its track id -> advance path
            if key[0] not in current_person_uids:
                continue  # person gone -> let the missing loop finalize it
            if mem.state not in (
                EventState.BAG_NEAR_PERSON,
                EventState.BAG_CARRIED,
                EventState.BAG_RELEASED,
                EventState.BAG_ON_GROUND,
            ):
                continue
            same_class = [
                info for info in primary_infos
                if (info.person_uid, info.bag_id) not in rebound_info_keys
                and info.person_uid == mem.person_uid
                and info.bag_class == mem.bag_class
            ]
            other = [
                info for info in primary_infos
                if (info.person_uid, info.bag_id) not in rebound_info_keys
                and info.person_uid == mem.person_uid
                and info.bag_class != mem.bag_class
            ]
            # REPAIR-P1-02: prefer highest association score; never silently
            # steal a different-class object when a same-class candidate exists.
            # Cross-class rebind is refused (safer than other[0] first-candidate).
            if same_class:
                candidate = max(same_class, key=lambda i: float(i.score))
            else:
                candidate = None
            if candidate is None:
                continue
            # Keep the SAME stable key; only refresh the live track id.
            mem.bag_id = candidate.bag_id
            active_keys.add(key)
            rebound_bag_ids.add(key)
            rebound_info_keys.add((candidate.person_uid, candidate.bag_id))
            emitted.extend(self._advance_pair(mem, candidate, timestamp, frame_index))

        # --- Advance / create pairs for current primary associations --------
        # Key by (STABLE person_uid, stable object bag_uid) so the whole event
        # arc maps to one pair + one evidence identity, regardless of tracker
        # id churn (object OR person). Raw track ids are legacy reference only.
        by_person: Dict[int, List] = {}
        for info in primary_infos:
            by_person.setdefault(int(info.person_uid), []).append(info)

        for person_uid, infos_p in by_person.items():
            person_current_bag_ids = {int(i.bag_id) for i in infos_p}
            first_slot = self._stable_uid_for_person(person_uid)
            for idx, info in enumerate(infos_p):
                # The STABLE OBJECT UID is the pair's object anchor when the
                # object-identity manager produced one (production path). The
                # per-person counter is only a fallback for uid-less callers.
                if info.bag_uid is not None:
                    slot = int(info.bag_uid)
                else:
                    slot = first_slot if idx == 0 else self._bag_uid_counter + 1
                    if idx != 0:
                        self._bag_uid_counter += 1
                key = (person_uid, slot)
                if key in rebound_bag_ids:
                    continue
                mem = self._pairs.get(key)
                if mem is not None and mem.bag_id != int(info.bag_id):
                    if int(mem.bag_id) in current_bag_ids and int(mem.bag_id) in person_current_bag_ids:
                        self._bag_uid_counter += 1
                        key = (person_uid, self._bag_uid_counter)
                        mem = None
                    else:
                        mem.bag_id = int(info.bag_id)
                if mem is None:
                    # STATIC GROUND-CLUTTER GATE: a low-confidence object that
                    # is already resting on the ground plane and is NOT near or
                    # carried by this person has no person origin — it must
                    # never enter FSM pair memory (e.g. a stationary 0.36-conf
                    # NESCAFE tin on the floor that people merely walk past).
                    if (
                        info.bag_confidence < self.config.min_carry_origin_confidence
                        and not info.carried
                        and not info.near
                        and (info.bag_below_feet or info.near_ground_plane)
                    ):
                        continue
                    mem = self._create_pair(info, bag_uid=key[1])
                    self._pairs[key] = mem
                active_keys.add(key)
                emitted.extend(self._advance_pair(mem, info, timestamp, frame_index))

        for key, mem in list(self._pairs.items()):
            if key in active_keys:
                continue
            mem.missing_frames += 1
            person_present = key[0] in current_person_uids
            bag_present = mem.bag_id in current_bag_ids
            if person_present:
                mem.person_seen_frames += 1
            if bag_present:
                mem.bag_seen_frames += 1

            # --- Person exited the frame while the discarded object remains --
            # A real littering arc often ends with the actor walking OUT of the
            # camera view right after the put-down. The pair then stops
            # producing info (no person) and used to be finalized as
            # PERSON_NOT_DETECTED even when the object was already released
            # and sitting on the ground. That exit IS the departure: keep the
            # release->ground->abandonment evidence flowing while the bag is
            # still detected, and confirm through the normal gates.
            if (
                not person_present
                and bag_present
                and mem.state in (EventState.BAG_RELEASED, EventState.BAG_ON_GROUND)
            ):
                bag_track = next(
                    (b for b in semantic_bags if int(b.track_id) == int(mem.bag_id)), None
                )
                if bag_track is not None:
                    if self._is_stationary(bag_track, 0.0):
                        mem.stationary_frames += 1
                        mem.stationary_miss_streak = 0
                    else:
                        mem.stationary_miss_streak += 1
                        if mem.stationary_miss_streak > cfg.stationary_grace_frames:
                            mem.stationary_frames = 0
                    if mem.state == EventState.BAG_RELEASED and mem.stationary_frames >= cfg.min_stationary_frames:
                        mem.state = EventState.BAG_ON_GROUND
                        mem.ground_frame = mem.ground_frame or frame_index
                        mem.ground_ts = mem.ground_ts or timestamp
                    if mem.state == EventState.BAG_ON_GROUND:
                        mem.abandonment_frames += 1
                        if mem.abandonment_frames >= cfg.min_abandonment_frames:
                            mem.departure_frame = mem.departure_frame or frame_index
                            mem.departure_ts = mem.departure_ts or timestamp
                            mem.state = EventState.PERSON_DEPARTED
                            emitted.extend(self._evaluate_confirmation(mem))

            # REPAIR-ORACLE-5290: after put-down, handheld color promotion often
            # stops (bag left the person carry band) so the pair loses its
            # semantic bag while still BAG_RELEASED / BAG_ON_GROUND. Continue
            # abandonment on missing-bag ticks so ground litter can confirm.
            if (
                not bag_present
                and mem.state in (EventState.BAG_RELEASED, EventState.BAG_ON_GROUND)
                and mem.release_frame is not None
            ):
                if mem.state == EventState.BAG_RELEASED:
                    mem.post_release_settle_streak += 1
                    need = max(2, int(cfg.min_stationary_frames) // 2)
                    if mem.post_release_settle_streak >= need:
                        mem.state = EventState.BAG_ON_GROUND
                        mem.ground_frame = mem.ground_frame or frame_index
                        mem.ground_ts = mem.ground_ts or timestamp
                        mem.stationary_frames = max(mem.stationary_frames, need)
                if mem.state == EventState.BAG_ON_GROUND:
                    mem.abandonment_frames += 1
                    mem.stationary_frames = max(
                        mem.stationary_frames, mem.abandonment_frames
                    )
                    if mem.abandonment_frames >= cfg.min_abandonment_frames:
                        mem.departure_frame = mem.departure_frame or frame_index
                        mem.departure_ts = mem.departure_ts or timestamp
                        mem.state = EventState.PERSON_DEPARTED
                        emitted.extend(self._evaluate_confirmation(mem))

            if mem.missing_frames > cfg.max_pair_age_frames:
                forced_reason = None
                if not person_present and mem.state in (EventState.BAG_CARRIED, EventState.BAG_RELEASED, EventState.BAG_ON_GROUND):
                    forced_reason = RejectionReason.PERSON_NOT_DETECTED.value
                elif not bag_present and mem.state in (EventState.BAG_CARRIED, EventState.BAG_RELEASED):
                    forced_reason = RejectionReason.BAG_NOT_DETECTED.value
                emitted.extend(self._finalize_pair(mem, forced_reason=forced_reason))
                del self._pairs[key]

        # Diagnostics snapshot (raw track id -> stable uid) for this tick.
        # Semantic-only: proposals do not get stable UIDs nor pollute diagnostics.
        self.last_person_uid_map = dict(person_uid_map)
        self.last_object_uid_map = {
            int(b.track_id): (int(b.object_uid) if b.object_uid is not None else None)
            for b in semantic_bags
        }
        # Proposal telemetry for audit (not used in decision logic)
        self.last_proposal_count = len(proposal_bags)
        self.last_semantic_count = len(semantic_bags)

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
        self._person_history.clear()
        self._person_identity.reset()
        self._object_identity.reset()
        self.last_person_uid_map = {}
        self.last_object_uid_map = {}
        self.last_proposal_count = 0
        self.last_semantic_count = 0
        self.confirmed_events.clear()
        self.rejected_events.clear()
        self._frame_index = 0
        self._last_timestamp = 0.0
        self._calibrated = False
        self._person_height_samples = []
        self._person_step_samples = []
        self.calibration_report = {}

    # ------------------------------------------------------------------ #
    # Per-video auto-calibration                                          #
    # ------------------------------------------------------------------ #
    def _calibrate(self, persons: List[DetectorPerson]) -> None:
        """Per-video auto-calibration (EventDetectorConfig.auto_calibrate).

        During the first ``calibration_frames`` analysis ticks this collects:
          * person bounding-box heights   -> scene-distance proxy
          * per-tick person centroid steps -> walking-speed proxy
        and, at the end of the window, adapts the MOTION threshold
        (``carry_motion_norm_px``) with hard clamps so no single video can
        push it into nonsense:

          * a far scene (small persons) lowers the px bar, a close scene
            raises it (scaled from the reference 160 px person height,
            clamped to x0.5..x3.0, final value clamped to [2, 48] px);
          * the bar never exceeds a third of the observed median walking
            step (a genuinely walking person must register as "moving"),
            and never falls below 2 px (jitter immunity).
        """
        cfg = self.config
        for person in persons:
            x1, y1, x2, y2 = person.bbox
            h = float(y2) - float(y1)
            if h > 8.0:
                self._person_height_samples.append(h)
            uid = self._person_uid_of(int(person.track_id))
            hist = self._person_history.get(uid)
            if hist is not None and len(hist) >= 2:
                (px, py), (cx, cy) = hist[-2], hist[-1]
                self._person_step_samples.append(math.hypot(cx - px, cy - py))

        if self._frame_index - self._calibration_start < cfg.calibration_frames - 1:
            return  # still collecting samples
        if self._calibrated:
            return
        self._calibrated = True

        med_h = _median(self._person_height_samples)
        med_step = _median(self._person_step_samples)
        if med_h:
            scale = max(0.5, min(3.0, med_h / 160.0))
            cfg.carry_motion_norm_px = max(2.0, min(48.0, 8.0 * scale))
        if med_step:
            bar = max(2.0, min(cfg.carry_motion_norm_px, med_step / 3.0))
            cfg.carry_motion_norm_px = bar
            # Departure: require roughly ONE SECOND of the scene's typical
            # walking (median step x analysis_fps), expressed in person
            # heights, clamped so it can neither vanish nor become impossible.
            if med_h:
                scene_step_ph = med_step / med_h
                cfg.departure_motion_ratio = max(
                    0.30, min(0.80, scene_step_ph * max(1.0, cfg.analysis_fps))
                )
        self.calibration_report = {
            "person_height_median_px": round(med_h, 1) if med_h else None,
            "person_step_median_px": round(med_step, 2) if med_step else None,
            "carry_motion_norm_px": round(cfg.carry_motion_norm_px, 2),
            "departure_motion_ratio": round(cfg.departure_motion_ratio, 3),
        }

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

    def _person_uid_of(self, track_id: int) -> int:
        """Best-effort stable uid for a raw person track id."""
        return self._person_identity._raw_to_uid.get(int(track_id), int(track_id))

    def _update_person_history(self, persons: List[DetectorPerson],
                               person_uid_map: Optional[Dict[int, int]] = None) -> None:
        """Phase 2: keep a short rolling centroid history per person.

        Keyed by STABLE person uid when ``person_uid_map`` is supplied, so a
        track-id switch preserves the motion history (carry cue continuity).
        """
        maxlen = max(3, int(self.config.pose_motion_window) + 2)
        for person in persons:
            uid = person_uid_map.get(int(person.track_id), int(person.track_id)) if person_uid_map else int(person.track_id)
            hist = self._person_history.get(uid)
            if hist is None or hist.maxlen != maxlen:
                hist = deque(maxlen=maxlen)
                self._person_history[uid] = hist
            hist.append(_centroid(person.bbox))

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
        pw = _box_width(person.bbox)

        # Reject objects physically too large to be handheld litter items carried by a person.
        # Dumpsters, trash bins, vehicles, benches, etc. are rejected immediately.
        bw = _box_width(bag.bbox)
        bh = _box_height(bag.bbox)
        bag_area = bw * bh
        person_area = max(1.0, pw * ph)
        if bag_area > 0.40 * person_area or bw > 1.15 * pw or bh > 0.80 * ph:
            return None
        if bag_area > 65000:
            return None

        distance = _dist(pc, bc)
        norm_distance = distance / ph
        containment = _intersection_over_bag(person.bbox, bag.bbox)
        carry_zone = _in_carrying_zone(person, bag, cfg)
        wrist_dist, wrist_hit = _wrist_distance(person, bag)
        wrist_near = wrist_hit and (wrist_dist / ph) <= cfg.near_distance_ratio
        kp = person.keypoints
        wrist_kp_available = bool(
            kp is not None
            and (kp.left_wrist is not None or kp.right_wrist is not None)
        )
        stationary = self._is_stationary(bag, ph)

        # --- motion-synchrony cues --------------------------------------- #
        # A genuinely carried object shares its carrier's velocity. A static
        # object on the ground does not, no matter how close the person walks
        # past it. Person steps come from the rolling person history (kept
        # every tick); bag steps from the bag history.
        p_step = _mean_step(self._person_history.get(self._person_uid_of(int(person.track_id))), 3)
        b_hist = self._bag_history.get(int(bag.track_id))
        # _mean_step consumes a list of centroid POINTS (same as the person
        # history); the bag history stores (timestamp, centroid) tuples, so
        # strip the timestamps here.
        b_step = _mean_step([c for _, c in b_hist] if b_hist else None, 3)
        person_moving = p_step is not None and p_step > cfg.carry_motion_norm_px
        if person_moving:
            if stationary:
                moves_with_person = False
            elif b_step is not None:
                moves_with_person = b_step >= cfg.carry_motion_sync_ratio * p_step
            else:
                moves_with_person = False
        else:
            # Person standing still: only consider carried if wrist is near
            # or the object is not stationary on the ground
            if stationary and not wrist_near:
                moves_with_person = False
            else:
                moves_with_person = True

        # Bag centre on the ground plane at the person's feet.
        # Strict band (bag_move_norm) kept for telemetry / clutter heuristics.
        # Carry unlatch and feet-put-down MUST use near_ground_plane
        # (ground_plane_margin_ratio) — REPAIR-P0-01. Using the 0.05·ph band
        # for carry exclusion permanently latched real put-downs (perspective
        # + bbox padding) so release_score stayed 0 forever.
        bag_below_feet = bc[1] >= (person.bbox[3] - cfg.bag_move_norm * ph)
        near_ground_plane = bc[1] >= (
            person.bbox[3] - cfg.ground_plane_margin_ratio * ph
        )
        # P1-5: is the bag centroid inside an operator-configured bin zone?
        in_bin_zone = False
        if self._active_bin_zones and self._frame_size:
            fw, fh = self._frame_size
            in_bin_zone = any(
                zone.contains_normalized(bc[0] / max(fw, 1), bc[1] / max(fh, 1))
                for zone in self._active_bin_zones
            )

        near = norm_distance <= cfg.near_distance_ratio or containment >= 0.25 or carry_zone
        # Zone carry: above the ground plane. When wrist keypoints exist and
        # the hands are away, also require the bag to be above the ground-plane
        # band — a stationary object in the 0.05–0.40 gap with hands clear is a
        # put-down, not a carry (REPAIR-P0-01). Without pose, keep the legacy
        # zone path so mid-body carries still latch.
        zone_carry = carry_zone and not bag_below_feet and not near_ground_plane
        if wrist_kp_available and (not wrist_near):
            zone_carry = False
        if stationary and (not wrist_near):
            zone_carry = False
        carried = (
            (wrist_near or zone_carry)
            and norm_distance <= cfg.near_distance_ratio * 1.5
            and moves_with_person
        )
        departed = norm_distance >= cfg.departure_distance_ratio
        associated = norm_distance <= cfg.association_radius_ratio

        if not (near or departed or containment > 0.0 or associated):
            return None

        proximity = max(0.0, 1.0 - norm_distance / max(cfg.near_distance_ratio * 2.0, 1e-6))
        conf_score = min(1.0, float(bag.confidence) / max(cfg.detection_high_conf, 1e-6))
        legacy_score = 0.55 * proximity + 0.25 * containment + 0.20 * conf_score

        # Phase 2 — blend in the body-relative pose evidence. Never replaces the
        # legacy score outright: with pose_score_blend=0 (or the feature flag
        # off) this is bit-identical to the pre-Phase-2 behaviour.
        pose_score: Optional[float] = None
        pose_components: Dict[str, Optional[float]] = {}
        score = legacy_score
        if cfg.pose_association_enabled and cfg.pose_score_blend > 0.0:
            bag_hist = self._bag_history.get(int(bag.track_id))
            bag_centroids = [c for _, c in bag_hist] if bag_hist else None
            pose_score, pose_components = _pose_association_score(
                person, bag,
                self._person_history.get(self._person_uid_of(int(person.track_id))),
                bag_centroids,
                cfg,
            )
            if pose_score is not None:
                w = max(0.0, min(1.0, cfg.pose_score_blend))
                score = (1.0 - w) * legacy_score + w * pose_score

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
            fallback=(bag.source != "yolo"),
            yolo_confirmed=bool(bag.yolo_confirmed),
            bag_uid=None if bag.object_uid is None else int(bag.object_uid),
            pose_score=(float(pose_score) if pose_score is not None else None),
            pose_components=pose_components,
            legacy_score=float(legacy_score),
            bag_centroid=bc,
            bag_below_feet=bool(bag_below_feet),
            near_ground_plane=bool(near_ground_plane),
            in_bin_zone=bool(in_bin_zone),
            person_moving=bool(person_moving),
            moves_with_person=bool(moves_with_person),
            wrist_near=bool(wrist_near),
            wrist_keypoints_available=bool(wrist_kp_available),
        )

    def _select_primary_associations(self, infos: List[_PairInfo]) -> List[_PairInfo]:
        by_bag: Dict[int, List[_PairInfo]] = {}
        for info in infos:
            by_bag.setdefault(info.bag_id, []).append(info)

        # Bag OWNERSHIP (the core person-centric guarantee). Once a pair has
        # anchored to an actor -- i.e. reached BAG_CARRIED or any later state --
        # that actor OWNS the bag for the remainder of the event. Geometry (who
        # is momentarily closer) must NOT override the established temporal
        # identity: a passer-by or bystander who walks nearer the dropped bag can
        # NEVER steal the association, and therefore can NEVER inherit (or block)
        # the actor's littering event. Without this, the closest-person rule
        # below would drop the real carrier's association during departure the
        # instant someone else is nearer, freezing the pair before confirmation.
        owners: Dict[int, int] = {}
        for key, mem in self._pairs.items():
            if int(mem.bag_id) in owners:
                continue
            if mem.state in (
                EventState.BAG_CARRIED,
                EventState.BAG_RELEASED,
                EventState.BAG_ON_GROUND,
                EventState.PERSON_DEPARTED,
            ):
                owners[int(mem.bag_id)] = int(key[0])

        primary: List[_PairInfo] = []
        for bag_id, infos_for_bag in by_bag.items():
            # Highest score first; on a tie (e.g. both persons far/departed so
            # their scores are equally low) the CLOSER person wins, not whichever
            # happened to be iterated first. Otherwise a passer-by processed
            # earlier could steal the bag association from the actual carrier.
            infos_for_bag.sort(key=lambda x: (-x.score, x.norm_distance))
            # Ownership override: if this bag already belongs to an anchored
            # actor and that actor is still present this frame, force the owner
            # to remain the primary association -- even if someone else is
            # currently closer. The bag stays bound to the actor who carried it.
            owner_pid = owners.get(int(bag_id))
            if owner_pid is not None:
                owner_info = next(
                    (i for i in infos_for_bag if int(i.person_uid) == owner_pid), None
                )
                if owner_info is not None:
                    best = owner_info
                    best.is_primary = True
                    for other in infos_for_bag:
                        if other is not best:
                            other.is_primary = False
                    primary.append(best)
                    continue
            best = infos_for_bag[0]
            best.is_primary = True
            if len(infos_for_bag) > 1:
                second = infos_for_bag[1]
                # Ambiguity / closer-person only matter when the bag is actually
                # being carried or is near a person (a genuine carry association).
                # A far "departed" blip — e.g. the bag is on the ground and BOTH
                # people have walked away, so their scores are equally low — must
                # NOT be treated as a competing association (it was causing
                # confirmed littering events to be wrongly rejected as
                # ASSOCIATION_AMBIGUOUS during the departure phase).
                if best.near or best.carried:
                    if best.score - second.score < self.config.association_margin:
                        best.ambiguous = True
                    best.other_person_closer = second.norm_distance < best.norm_distance * 0.95
            primary.append(best)

        # Enforce ONE BAG PER PERSON. Littering is a single-actor /
        # single-object interaction: a person carries at most one litter object
        # at a time. Without this, a person momentarily linked to several
        # churned/duplicate bag detections (YOLO + novelty + colour fallback
        # all firing on the same physical bag) would spawn one pair per bag id,
        # fragmenting the temporal evidence and defeating the stable-identity
        # goal (spec rule #11). Among the bags that each chose the same person
        # as their best match, keep only the highest-scoring one; the rest are
        # demoted to non-primary so the FSM tracks exactly one object per actor.
        by_person: Dict[int, List[_PairInfo]] = {}
        for info in primary:
            by_person.setdefault(int(info.person_uid), []).append(info)
        final: List[_PairInfo] = []
        for infos_for_person in by_person.values():
            if len(infos_for_person) <= 1:
                final.extend(infos_for_person)
                continue
            # Prefer YOLO-confirmed bags over color/novelty latches when the
            # same actor has multiple overlapping associations — prevents a
            # hip clothing color blob from stealing the pair from a real bag.
            infos_for_person.sort(
                key=lambda x: (
                    0 if x.yolo_confirmed else 1,
                    -x.score,
                    x.norm_distance,
                )
            )
            kept = infos_for_person[0]
            final.append(kept)
            for other in infos_for_person[1:]:
                # A "competing second bag" only matters when it is actually near /
                # carried / within the association radius of this actor. A merely
                # "departed" association to a FAR bag (e.g. the other actor's bag
                # the person walked away from) is spurious and must NOT be marked
                # ambiguous — otherwise it pollutes the primary pair's
                # ambiguous_frames and wrongly rejects a genuine littering event.
                if other.near or other.carried:
                    other.is_primary = False
                    other.ambiguous = True  # suppressed: a genuine second bag for this actor
                # else: spurious far association — drop silently, do not flag.
        return final

    def _stable_uid_for_person(self, person_uid: int) -> int:
        """Return the single stable object uid for this actor's carried bag.

        Littering is a one-bag-per-person interaction, so one stable uid per
        person anchors the whole event to the same physical object despite the
        underlying tracker ids churning every few frames (spec rule #11).
        """
        if person_uid not in self._person_obj_uid:
            self._bag_uid_counter += 1
            self._person_obj_uid[person_uid] = self._bag_uid_counter
        return self._person_obj_uid[person_uid]

    def _create_pair(self, info: _PairInfo, bag_uid: Optional[int] = None) -> _PairMemory:
        cfg = self.config
        # Use the supplied stable object uid (per-person) or mint a fresh one.
        if bag_uid is None:
            self._bag_uid_counter += 1
            bag_uid = self._bag_uid_counter
        mem = _PairMemory(
            person_id=info.person_id,
            person_uid=info.person_uid,
            bag_id=info.bag_id,
            bag_uid=bag_uid,
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
        if mem.first_bag_centroid is None and info.bag_centroid is not None:
            mem.first_bag_centroid = info.bag_centroid
        if info.wrist_near and info.carried:
            mem.ever_wrist_near = True
        # CARRY-ORIGIN evidence (clutter suppression): accumulate handling cues
        # so the CARRIED transition can verify the object truly originates
        # from a person rather than being static ground clutter.
        if info.containment >= 0.25:
            mem.ever_contained = True
        if info.carried and info.moves_with_person:
            mem.ever_moved_with_person = True
        if mem.first_bag_centroid is not None and info.bag_centroid is not None:
            mem.max_bag_displacement_px = max(
                mem.max_bag_displacement_px,
                _dist(mem.first_bag_centroid, info.bag_centroid),
            )
        # Off-ground arming uses the WIDE ground-plane band: a carry that never
        # left near_ground_plane (typical low hold at ~0.25·ph) must not arm
        # standing crit_ground (that would false-release carry-only clips).
        # Put-downs that began above the band (y < feet - 0.40·ph) still arm.
        if info.carried and not info.near_ground_plane:
            mem.ever_off_ground_while_carried = True
        if info.carried and info.person_moving:
            mem.ever_person_moved_while_carried = True
        if info.wrist_keypoints_available:
            mem.pose_ever_available = True
        person_motion = 0.0
        if mem.release_person_centroid is not None and mem.release_person_height > 0.0:
            person_motion = _dist(info.person_centroid, mem.release_person_centroid) / mem.release_person_height
        departure_threshold = (
            max(cfg.departure_distance_ratio, mem.departure_baseline * cfg.departure_distance_ratio)
            if mem.departure_baseline > 0.0
            else math.inf
        )
        # Distance growth alone is NOT departure while the actor stands still
        # and the bag is still falling (false "departure" that beat regrab).
        # Credit departure when the person moves away from the release pose,
        # or when a settled/grounded bag's separation grows while the person
        # is walking.
        motion_departed = (
            mem.release_frame is not None
            and person_motion >= cfg.departure_motion_ratio
        )
        bag_settled_for_depart = bool(
            info.stationary or mem.ground_frame is not None
        )
        norm_departed = (
            mem.release_frame is not None
            and info.norm_distance >= departure_threshold
            and bag_settled_for_depart
            and (
                info.person_moving
                or motion_departed
                or person_motion >= 0.5 * cfg.departure_motion_ratio
            )
        )
        departed_now = bool(norm_departed or motion_departed)
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
        regrab_candidate = False
        if mem.release_frame is not None and not info.stationary:
            # REPAIR-P0-01d / ORACLE-5290: do NOT use near_ground_plane as the
            # regrab gate. Require a real LIFT from the put-down pose.
            # Before BAG_ON_GROUND, only a STRONG lift may reclaim — otherwise
            # aggressive learned release thresholds cause release↔regrab
            # flicker and finalize as PICKED_BACK_UP (container IMG_5290 miss).
            # After ground, measure lift from the deeper of release vs current
            # bag depth so a pick-up from the ground plane counts even when
            # release_centroid was mid-drop.
            lifted = False
            strong_lift = False
            if (
                mem.release_bag_centroid is not None
                and info.bag_centroid is not None
                and mem.release_person_height > 1.0
            ):
                baseline_y = float(mem.release_bag_centroid[1])
                if mem.ground_bag_centroid is not None:
                    baseline_y = max(baseline_y, float(mem.ground_bag_centroid[1]))
                upward = baseline_y - float(info.bag_centroid[1])
                lifted = upward >= 0.15 * mem.release_person_height
                strong_lift = upward >= 0.25 * mem.release_person_height
            walking_reclaim = (
                mem.ground_frame is not None
                and info.person_moving
                and info.moves_with_person
                and (smooth_carried or info.wrist_near)
            )
            if mem.ground_frame is None:
                regrab_candidate = bool(
                    strong_lift and (smooth_carried or info.wrist_near or info.near)
                )
                mem.regrab_lift_streak = 0
            else:
                # Count pure strong_lift ticks toward the streak (bag may leave
                # the ground before it re-enters the wrist band). Accept reclaim
                # only after sustained lift — walking_reclaim alone used to fire
                # on person-jitter + bag reassociation (Docker IMG_5290).
                if strong_lift:
                    mem.regrab_lift_streak += 1
                else:
                    mem.regrab_lift_streak = 0
                sustained = mem.regrab_lift_streak >= max(2, min_votes)
                associated = bool(
                    smooth_carried
                    or info.wrist_near
                    or info.near
                    or walking_reclaim
                )
                regrab_candidate = bool(sustained and associated)
        mem.regrab_flags.append(regrab_candidate)
        smooth_regrab = _smooth(mem.regrab_flags, min_votes)

        if smooth_near:
            mem.near_frames += 1
        if smooth_carried:
            mem.carried_frames += 1
            if mem.carry_start_frame is None:
                mem.carry_start_frame = frame_index
                mem.carry_start_ts = timestamp
            # Per-pair adaptive release threshold: the further the object was
            # actually carried from the body, the more separation a release
            # requires — but never more than the configured cap, and never
            # less than the floor. A close-carried bottle (0.09 person-heights)
            # gets a proportional, reachable threshold instead of the global
            # 0.35 that only a full-stretch throw can satisfy.
            mem.max_carry_norm_distance = max(mem.max_carry_norm_distance, info.norm_distance)
            mem.release_distance_threshold = min(
                cfg.release_distance_ratio,
                max(cfg.release_distance_floor, cfg.release_distance_adaptive_ratio * mem.max_carry_norm_distance),
            )
        if mem.release_frame is not None:
            # Ground/bin evidence only while the bag is resting (stationary).
            # Counting mid-flight frames lets a bin deposit accumulate false
            # ground evidence while falling through the 0.40·ph band outside
            # the zone polygon (P1-5 / multiperson bin regressions).
            resting = bool(smooth_stationary or info.stationary)
            if resting and info.in_bin_zone:
                mem.bin_zone_frames += 1
            if (
                resting
                and (info.near_ground_plane or info.bag_below_feet)
                and not info.in_bin_zone
            ):
                mem.ground_evidence_frames += 1
            # Physical separation: clothing/hip color latches stay high-
            # containment forever; a real discard leaves the person box.
            mem.max_post_release_norm_distance = max(
                mem.max_post_release_norm_distance, float(info.norm_distance)
            )
            if info.containment < 0.25 or info.norm_distance >= max(
                cfg.release_distance_floor, mem.release_distance_threshold * 0.5
            ):
                mem.separated_frames += 1
            if smooth_stationary:
                mem.stationary_frames += 1
                mem.stationary_miss_streak = 0
                mem.release_frames += 1
            else:
                # Grace: one or two flickery ticks (HSV jitter, tracker churn)
                # must not erase a solid stationary streak.
                mem.stationary_miss_streak += 1
                # Once the bag is ON_GROUND, do not wipe stationary evidence —
                # post-drop color jitter regularly trips non-stationary and was
                # erasing the settle credit before abandonment could confirm
                # (IMG_5290 oracle: ground reached, PERSON_DID_NOT_DEPART).
                if (
                    mem.ground_frame is None
                    and mem.stationary_miss_streak > cfg.stationary_grace_frames
                ):
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
                # CARRY-ORIGIN CONSTRAINT (clutter suppression): when wrist pose
                # keypoints are available, the object must have been spatially
                # linked to a wrist/hand before it may become CARRIED. Without
                # pose availability, fall back to geometric handling evidence
                # (containment in the person box, or motion-synchrony while
                # carried) so production without MoveNet still works.
                pose_available = mem.pose_ever_available
                origin_ok = (
                    mem.ever_wrist_near
                    if pose_available
                    else (mem.ever_wrist_near or mem.ever_moved_with_person)
                )
                if not cfg.require_carry_origin_link or origin_ok:
                    mem.state = EventState.BAG_CARRIED
                    mem.carry_start_frame = mem.carry_start_frame or frame_index
                    mem.carry_start_ts = mem.carry_start_ts or timestamp
                # Phase C: freeze the authoritative actor + object identity at the
                # moment the carry is established (continuous temporal association
                # is now locked). The object track id may churn later, but this is
                # the id that was actually carried, and the stable uid travels with
                # the pair regardless of churn.
                if mem.carry_person_track_id is None:
                    mem.carry_person_track_id = int(info.person_id)
                    mem.carry_person_uid = int(info.person_uid)
                    mem.carry_object_track_id = int(info.bag_id)
                    mem.carry_object_uid = mem.bag_uid

        if mem.state == EventState.BAG_CARRIED:
            if self._release_detected(mem, info, smooth_carried, distance_increasing):
                mem.state = EventState.BAG_RELEASED
                mem.release_frame = frame_index
                mem.release_ts = timestamp
                mem.departure_baseline = max(1e-6, info.norm_distance)
                mem.release_person_centroid = info.person_centroid
                mem.release_person_height = info.person_height
                mem.release_bag_centroid = info.bag_centroid
                mem.max_departure_ratio = 0.0
                mem.stationary_frames = 0
                mem.stationary_miss_streak = 0
                mem.departed_frames = 0
                mem.feet_release_streak = 0
                mem.desync_release_streak = 0
                mem.carried_stationary_streak = 0
                mem.post_release_settle_streak = 0
                # REPAIR-P0-01b: after a ground put-down, wrist_near can keep
                # smooth_carried True and immediately regrab the same bag.
                # Clear the carry smoothing window so release can stick.
                mem.carried_flags.clear()
                for _ in range(max(1, cfg.smoothing_window)):
                    mem.carried_flags.append(False)

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
                mem.release_bag_centroid = None
                mem.ground_bag_centroid = None
                mem.max_departure_ratio = 0.0
                mem.reclaimed = True
                mem.post_release_settle_streak = 0
                mem.regrab_lift_streak = 0
            elif mem.release_frame == frame_index:
                # Same-tick fall-through after BAG_CARRIED→RELEASED: seed settle
                # but do not complete BAG_ON_GROUND (protects regrab window).
                left_behind = not (info.person_moving and info.moves_with_person)
                if info.stationary or left_behind or info.near_ground_plane:
                    mem.post_release_settle_streak = max(
                        mem.post_release_settle_streak, 1
                    )
            elif mem.stationary_frames >= cfg.min_stationary_frames:
                mem.state = EventState.BAG_ON_GROUND
                mem.ground_frame = frame_index
                mem.ground_ts = timestamp
                if info.bag_centroid is not None:
                    mem.ground_bag_centroid = info.bag_centroid
            else:
                # After release, always make progress toward BAG_ON_GROUND.
                # Real iPhone tracks often keep a false motion-sync and never
                # trip the strict stationary gate — starving settle blocked
                # IMG_5117. Faster when clearly left-behind; slow otherwise.
                left_behind = not (info.person_moving and info.moves_with_person)
                if info.stationary or left_behind:
                    mem.post_release_settle_streak += 2
                else:
                    mem.post_release_settle_streak += 1
                need = max(2, int(cfg.min_stationary_frames) // 2)
                if mem.post_release_settle_streak >= need:
                    mem.state = EventState.BAG_ON_GROUND
                    mem.ground_frame = frame_index
                    mem.ground_ts = timestamp
                    if info.bag_centroid is not None:
                        mem.ground_bag_centroid = info.bag_centroid
                    mem.stationary_frames = max(
                        mem.stationary_frames,
                        min(mem.post_release_settle_streak, need),
                    )
                    # Ground evidence: strict band, or a slightly looser
                    # perspective band (0.25·ph) used only at settle time so
                    # elevated bin rests (~0.5-0.7·ph) stay BIN_DISPOSAL.
                    loose_ground = False
                    if (
                        info.bag_centroid is not None
                        and info.person_height > 1.0
                    ):
                        feet_y = (
                            info.person_centroid[1] + 0.5 * info.person_height
                        )
                        loose_ground = info.bag_centroid[1] >= (
                            feet_y - 0.25 * info.person_height
                        )
                    if not info.in_bin_zone and (
                        info.near_ground_plane
                        or info.bag_below_feet
                        or loose_ground
                    ):
                        mem.ground_evidence_frames += 1

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
                mem.release_bag_centroid = None
                mem.ground_bag_centroid = None
                mem.max_departure_ratio = 0.0
                mem.reclaimed = True
                mem.post_release_settle_streak = 0
                mem.regrab_lift_streak = 0
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
                # Pause while a reclaim lift is building (streak) or pending
                # smooth_regrab — otherwise abandonment races the lift path.
                if (
                    not smooth_regrab
                    and not regrab_candidate
                    and mem.regrab_lift_streak == 0
                ):
                    mem.abandonment_frames += 1
                elif smooth_regrab:
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
                mem.reclaimed = True
            else:
                return self._evaluate_confirmation(mem)

        return []

    def _bag_was_handled(self, mem: _PairMemory, info: _PairInfo) -> bool:
        """Anti-furniture gate for the alternative release criteria.

        A release that is NOT based on the object flying away from the person
        must only be trusted when the object was genuinely HANDLED at some
        point: it either moved with the pair (displacement since first
        sighting >= ``bag_move_norm`` person-heights) or a wrist was near it
        while it was carried. A park bench the person stood next to is
        furniture and must never "release".
        """
        if mem.ever_wrist_near:
            return True
        if mem.first_bag_centroid is not None and mem.max_carry_norm_distance > 0.0:
            moved = _dist(mem.first_bag_centroid, info.bag_centroid) if info.bag_centroid else 0.0
            if moved >= self.config.bag_move_norm * max(info.person_height, 1.0):
                return True
        return False

    def _release_detected(
        self,
        mem: _PairMemory,
        info: _PairInfo,
        smooth_carried: bool,
        distance_increasing: bool,
    ) -> bool:
        """Multi-observation release detection (spec: never a single frame).

        Four criteria, ANY of which declares the release:

        1. distance   — the classic: carried flipped false AND the object is
           beyond the per-pair adaptive threshold AND the separation grew
           (this tick or net over the release window — tolerates one jitter
           tick mid-throw). Fast-drop path: distance may fire with
           ``max(2, min_carried_frames - 2)`` carried ticks.
        2. static-bag — the object reads stationary while the person walks
           and no wrist is near it. This catches the walk-past-the-dropped-
           bag geometry where the 2D centroid distance DECREASES (the person
           walks toward the camera, the bag stays behind) and the classic
           criterion can never fire.
        3. feet put-down — sustained ticks of "object at the feet line, no
           wrist near it, not carried": the person set it down at their feet
           and the hand moved away, even while they stand still.

        Criteria 2/3 are gated by the anti-furniture check.
        Alt-release paths (feet / ground / held-static / standing) must NOT
        fire while the bag remains body-attached (high containment + tiny
        separation) — that pattern is clothing / hip color latch, not discard.
        """
        cfg = self.config
        # Fast-drop: allow the classic distance criterion with a slightly
        # reduced carry count so a brief carry→throw is not hard-blocked.
        min_carry_distance = max(2, int(cfg.min_carried_frames) - 2)
        if mem.carried_frames < min_carry_distance:
            return False

        # 1) classic distance-based release, windowed + adaptive threshold.
        # P2-1 wiring: the window is BOUNDED by release_window_frames — the
        # growth test looks at the most recent ticks only. The old
        # full-history test kept firing long after the object settled (any
        # old rise counted forever), making the key dead config.
        window = list(mem.distances)[-max(2, int(cfg.release_window_frames)):]
        grew_over_window = len(window) >= 3 and window[-1] > window[0]
        crit_distance = (
            not smooth_carried
            and info.norm_distance >= mem.release_distance_threshold
            and (distance_increasing or grew_over_window)
        )
        if crit_distance and mem.carried_frames >= min_carry_distance:
            return True
        # Remaining alt-release criteria still require the full carry count.
        if mem.carried_frames < cfg.min_carried_frames:
            return False

        handled = self._bag_was_handled(mem, info)

        # Body-attached latch: bag still mostly inside the person box and
        # never reached a meaningful release distance. Alt criteria must not
        # invent a "release" for clothing / pocket / hip color blobs.
        still_attached = (
            info.containment >= 0.25
            and info.norm_distance
            < max(cfg.release_distance_floor, mem.release_distance_threshold * 0.5)
        )

        # 2) static-bag put-down while the person moves away
        crit_static = (
            not still_attached
            and not smooth_carried
            and info.stationary
            and not info.carried
            and info.person_moving
            and not info.wrist_near
            and handled
        )

        # 3) feet put-down streak (works while the person stands still).
        # Use near_ground_plane (REPAIR-P0-01) so real camera put-downs that
        # never enter the 0.05·ph band can still release while standing.
        if (
            info.near_ground_plane
            and not info.wrist_near
            and not info.carried
            and info.stationary
            and not still_attached
        ):
            mem.feet_release_streak += 1
        else:
            mem.feet_release_streak = 0
        crit_feet = (
            handled
            and not still_attached
            and mem.feet_release_streak >= cfg.feet_release_frames
        )

        # 4) grounded put-down while wrist geometry still looks like a carry.
        # Requires a prior carry ABOVE the ground-plane band so a low
        # stationary hold never false-releases (carry-only → NO_RELEASE).
        bag_left_behind = info.person_moving and not info.moves_with_person
        stopped_after_walk_with_resting_bag = (
            mem.ever_person_moved_while_carried
            and (not info.person_moving)
            and info.stationary
        )
        bag_unsynced = bag_left_behind or stopped_after_walk_with_resting_bag
        crit_ground = (
            handled
            and not still_attached
            and mem.ever_off_ground_while_carried
            and info.near_ground_plane
            and info.stationary
            and bag_unsynced
            and mem.carried_frames >= cfg.min_carried_frames
        )
        # Standing put-down after a real above-band carry: person never needs
        # to walk. Covers wrist-near grounded bags (standing over a drop).
        crit_standing_ground = (
            handled
            and not still_attached
            and mem.ever_off_ground_while_carried
            and info.near_ground_plane
            and info.stationary
            and (not info.person_moving)
            and mem.carried_stationary_streak
            >= max(2, int(cfg.feet_release_frames))
            and mem.carried_frames >= cfg.min_carried_frames
        )

        # 5) motion-desync put-down (streaked). Real tracks often never satisfy
        # the strict stationary gate (centroid jitter) — so also accept
        # "person still walking, bag no longer motion-synced" as leave-behind.
        person_stopped_with_static_bag = (
            mem.ever_person_moved_while_carried
            and (not info.person_moving)
            and info.stationary
        )
        desync_tick = (
            handled
            and not still_attached
            and mem.ever_person_moved_while_carried
            and mem.carried_frames >= cfg.min_carried_frames
            and (
                info.near_ground_plane
                or info.norm_distance >= mem.release_distance_threshold * 0.5
            )
            and (person_stopped_with_static_bag or bag_left_behind)
        )
        if desync_tick:
            mem.desync_release_streak += 1
        else:
            mem.desync_release_streak = 0
        crit_desync = mem.desync_release_streak >= max(2, cfg.feet_release_frames)

        # 6) stationary-after-walk put-down: after the person walked while
        # carrying, the bag goes stationary for a sustained streak while the
        # FSM is still wrist-latched. Real iPhone put-downs often never enter
        # near_ground_plane (bag projects above the feet line). Require either
        # ground-plane evidence, a modest separation, or the person stopping.
        # Stationary streak while still latched as carried — arms standing
        # put-down / held-static release after a real above-band or walking carry.
        if info.stationary and (
            mem.ever_person_moved_while_carried or mem.ever_off_ground_while_carried
        ):
            mem.carried_stationary_streak += 1
        else:
            mem.carried_stationary_streak = 0
        crit_held_static = (
            handled
            and not still_attached
            and mem.ever_person_moved_while_carried
            and mem.carried_stationary_streak
            >= max(2, int(cfg.feet_release_frames))
            and mem.carried_frames >= cfg.min_carried_frames
            and (
                info.near_ground_plane
                or info.norm_distance >= mem.release_distance_threshold * 0.35
                or (
                    (not info.person_moving)
                    and info.stationary
                    and not info.wrist_near
                )
            )
        )

        return bool(
            crit_static
            or crit_feet
            or crit_ground
            or crit_standing_ground
            or crit_desync
            or crit_held_static
        )

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
        # REPAIR-P0-02: color is admitted by _is_semantic_waste again, so
        # fallback_frames increments and this discount is LIVE.
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
        # P2-1 wiring (design principle #4: never let CSRT/color coasting
        # confirm a violation): a pair whose object has been tracked ONLY by
        # the fallback tracker for more than max_fallback_tracker_gap_frames
        # has not been re-confirmed by YOLO within the allowed window. The
        # learning store (LEARNING_STEPS["BAG_NOT_DETECTED"]) already assumes
        # this key drives BAG_NOT_DETECTED relaxations.
        if mem.fallback_gap_frames > cfg.max_fallback_tracker_gap_frames:
            return RejectionReason.BAG_NOT_DETECTED.value
        if mem.mean_confidence() < cfg.detection_low_conf:
            return RejectionReason.LOW_BAG_CONFIDENCE.value
        if mem.person_seen_frames < cfg.min_carried_frames:
            return RejectionReason.PERSON_NOT_DETECTED.value
        # Fast-drop: a clear physical separation may confirm with slightly
        # fewer carried ticks than the full min_carried_frames bar.
        min_carry_for_confirm = max(2, int(cfg.min_carried_frames) - 2)
        sep_floor_early = max(cfg.release_distance_floor, 0.08)
        fast_drop_ok = (
            mem.carried_frames >= min_carry_for_confirm
            and mem.release_frame is not None
            and (
                mem.max_post_release_norm_distance >= sep_floor_early
                or mem.separated_frames >= max(2, int(cfg.feet_release_frames))
            )
        )
        if mem.carried_frames < cfg.min_carried_frames and not fast_drop_ok:
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
        # BIN VS GROUND gate: a candidate whose resting location was never
        # observed on the actor's ground plane is a bin/container deposit, not
        # ground littering -> report BIN_DISPOSAL or operator bin-zone deposit.
        if cfg.require_ground_confirmation and mem.ground_evidence_frames <= 0:
            if mem.bin_zone_frames > 0:
                return RejectionReason.BIN_ZONE_DEPOSIT.value
            return RejectionReason.BIN_DISPOSAL.value
        # Physical separation gate: an object that was contained in the person
        # box (clothing / hip latch) but never left the body after "release"
        # is not ground littering. Also reject static color/clutter blobs that
        # never moved while the person walked away (inflated norm_distance).
        sep_floor = max(cfg.release_distance_floor, 0.08)
        physically_separated = (
            mem.max_post_release_norm_distance >= sep_floor
            and mem.separated_frames >= max(2, int(cfg.feet_release_frames))
        )
        bag_actually_moved = mem.max_bag_displacement_px >= (
            max(0.12, cfg.bag_move_norm * 2.0) * max(mem.release_person_height, 80.0)
        )
        if mem.ever_contained and (not physically_separated or not bag_actually_moved):
            return RejectionReason.NO_PHYSICAL_SEPARATION.value
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
        if forced_reason is None:
            reason = self._rejection_reason(mem, evidence)
            if reason is None:
                # End-of-stream with a fully satisfied evidence gate: confirm.
                # Without this, standing put-downs that never trip the
                # abandonment/departure transition were rejected as
                # EVENT_CONFIDENCE_TOO_LOW despite confidence >> threshold.
                return self._evaluate_confirmation(mem)
            # Regrab protection (explicit reclassification): if the person reclaimed
            # a released/grounded bag and the pair ends as a NON-violation (the bag
            # was never re-abandoned), reclassify it as PICKED_BACK_UP (NOT_ABANDONED)
            # instead of a generic NO_RELEASE_TRANSITION rejection. This records the
            # avoided false positive in the audit trail.
            if mem.reclaimed and mem.release_frame is None:
                mem.state = EventState.PICKED_BACK_UP
                reason = RejectionReason.PICKED_BACK_UP.value
        else:
            reason = forced_reason
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
            bag_track_id=mem.carry_object_track_id if mem.carry_object_track_id is not None else mem.bag_id,
            bag_uid=mem.bag_uid,
            event_actor_person_track_id=mem.carry_person_track_id if mem.carry_person_track_id is not None else mem.person_id,
            event_actor_person_uid=mem.carry_person_uid if mem.carry_person_uid is not None else mem.person_uid,
            event_object_track_id=mem.carry_object_track_id if mem.carry_object_track_id is not None else mem.bag_id,
            event_object_uid=mem.bag_uid,
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
                # BIN VS GROUND: always reported, even when the gate is off.
                "location_status": (
                    "GROUND_CONFIRMED" if mem.ground_evidence_frames > 0
                    else "BIN_ZONE" if mem.bin_zone_frames > 0
                    else "LOCATION_AMBIGUOUS"
                ),
                "ground_evidence_frames": mem.ground_evidence_frames,
                "bin_zone_frames": mem.bin_zone_frames,
                "max_post_release_norm_distance": round(
                    mem.max_post_release_norm_distance, 4
                ),
                "separated_frames": mem.separated_frames,
                "ever_contained": mem.ever_contained,
                "max_bag_displacement_px": round(mem.max_bag_displacement_px, 2),
                # P1-8: disclose the EFFECTIVE runtime thresholds that produced
                # this decision (base YAML + any adaptive learned overrides).
                "active_thresholds": {
                    k: getattr(self.config, k) for k in TUNABLE_THRESHOLD_KEYS
                },
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
        # P2-2: the legacy "csrt_fallback" branch is retired; proposals
        # (color/novelty) render in orange, semantic yolo in yellow.
        is_proposal = str(getattr(b, "source", "yolo") or "yolo").lower() != "yolo"
        color = (0, 165, 255) if is_proposal else (0, 255, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"B{b.track_id} {b.confidence:.2f}"
        if is_proposal:
            label += " PROPOSAL"
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


