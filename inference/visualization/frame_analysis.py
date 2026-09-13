"""Structured frame-level analysis contract for visualization.

Every field in FrameAnalysis must come from real production objects:

* bbox / confidence -> detector output
* track_id          -> ByteTrack / color tracker namespaced IDs
* keypoints         -> MoveNet output
* trail             -> BytetrackTracker TrackStore history
* association/state -> LitteringEventDetector pair memory
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from inference.association.person_object_assoc import Track


@dataclass
class PersonAnalysis:
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    keypoints: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    head_visible: bool = False
    head_bbox: Optional[Tuple[float, float, float, float]] = None
    trail: List[Tuple[float, float]] = field(default_factory=list)
    state: Optional[str] = None
    associated_object_id: Optional[int] = None
    person_uid: Optional[int] = None  # stable logical id across track-id churn (Phase C)


@dataclass
class ObjectAnalysis:
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    source: str = "yolo"
    trail: List[Tuple[float, float]] = field(default_factory=list)
    state: Optional[str] = None
    associated_person_id: Optional[int] = None
    object_uid: Optional[int] = None  # stable logical id across track-id churn (Phase B)


@dataclass
class AssociationAnalysis:
    person_track_id: int
    object_track_id: int
    state: str
    score: float
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass
class FrameAnalysis:
    timestamp: float
    frame_number: int
    persons: List[PersonAnalysis] = field(default_factory=list)
    objects: List[ObjectAnalysis] = field(default_factory=list)
    associations: List[AssociationAnalysis] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    event: Optional[Dict[str, Any]] = None
    hud: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_height(bbox: Tuple[float, float, float, float]) -> float:
    return max(1.0, float(bbox[3]) - float(bbox[1]))


def _head_bbox(track: Track) -> Tuple[bool, Optional[Tuple[float, float, float, float]]]:
    kp = getattr(track, "keypoints", None)
    nose = getattr(kp, "nose", None) if kp is not None else None
    if nose is None:
        return False, None
    x, y = float(nose[0]), float(nose[1])
    ls = getattr(kp, "left_shoulder", None)
    rs = getattr(kp, "right_shoulder", None)
    if ls is not None and rs is not None:
        shoulder_dist = math.hypot(float(ls[0]) - float(rs[0]), float(ls[1]) - float(rs[1]))
        radius = max(12.0, shoulder_dist * 0.55)
    else:
        radius = max(12.0, _box_height(track.bbox) * 0.12)
    return True, (x - radius, y - radius, x + radius, y + radius)


def _trail_for(tracker: Any, track_id: int, max_points: int = 24) -> List[Tuple[float, float]]:
    if tracker is None:
        return []
    try:
        hist = tracker.get_history(int(track_id))
    except Exception:
        return []
    if hist is None:
        return []
    pts = list(hist.centroids)
    if len(pts) > max_points:
        pts = pts[-max_points:]
    return [(float(x), float(y)) for x, y in pts]


def _association_start(person: Track) -> Tuple[float, float]:
    kp = getattr(person, "keypoints", None)
    if kp is not None:
        lw = getattr(kp, "left_wrist", None)
        rw = getattr(kp, "right_wrist", None)
        if lw is not None and rw is not None:
            # choose the wrist closer to the object? Caller can refine; use midpoint.
            return ((float(lw[0]) + float(rw[0])) / 2.0, (float(lw[1]) + float(rw[1])) / 2.0)
        if lw is not None:
            return (float(lw[0]), float(lw[1]))
        if rw is not None:
            return (float(rw[0]), float(rw[1]))
        tc = getattr(kp, "torso_center", None)
        if tc is not None:
            return (float(tc[0]), float(tc[1]))
    return _centroid(person.bbox)


def build_frame_analysis(
    persons: List[Track],
    objects: List[Track],
    detector: Any,
    tracker: Any,
    timestamp: float,
    frame_number: int,
    *,
    source_fps: Optional[float] = None,
    analysis_fps: Optional[float] = None,
    video_name: Optional[str] = None,
    event: Optional[Dict[str, Any]] = None,
) -> FrameAnalysis:
    """Build a visualization-only snapshot from real production objects."""
    pair_by_person: Dict[int, Any] = {}
    pair_by_bag: Dict[int, Any] = {}
    states: List[str] = []
    associations: List[AssociationAnalysis] = []

    pairs = getattr(detector, "_pairs", {}) if detector is not None else {}
    for (person_id, bag_id), mem in pairs.items():
        pair_by_person[int(person_id)] = mem
        pair_by_bag[int(bag_id)] = mem
        state_value = getattr(mem.state, "value", str(mem.state))
        if state_value not in states:
            states.append(state_value)

    person_map = {int(p.track_id): p for p in persons}
    object_map = {int(o.track_id): o for o in objects}

    # Association lines are drawn whenever the production detector actually
    # binds a person and an object into a pair — i.e. for the WHOLE real
    # association lifetime (from first near-contact through carry/release/ground/
    # departure), not only during the near/carried window. This makes the
    # association visibly persist across states once the link genuinely exists.
    for (person_id, bag_id), mem in pairs.items():
        person = person_map.get(int(person_id))
        bag = object_map.get(int(bag_id))
        if person is None or bag is None:
            continue
        state_value = getattr(mem.state, "value", str(mem.state))
        associations.append(
            AssociationAnalysis(
                person_track_id=int(person_id),
                object_track_id=int(bag_id),
                state=state_value,
                score=float(getattr(mem, "mean_association_score", lambda: 0.0)()),
                start=_association_start(person),
                end=_centroid(bag.bbox),
            )
        )

    person_analyses: List[PersonAnalysis] = []
    for p in persons:
        kp = getattr(p, "keypoints", None)
        keypoints: Dict[str, Optional[Tuple[float, float]]] = {}
        if kp is not None:
            keypoints = {
                "left_wrist": getattr(kp, "left_wrist", None),
                "right_wrist": getattr(kp, "right_wrist", None),
                "left_shoulder": getattr(kp, "left_shoulder", None),
                "right_shoulder": getattr(kp, "right_shoulder", None),
                "torso_center": getattr(kp, "torso_center", None),
                "nose": getattr(kp, "nose", None),
            }
        head_visible, head_bbox = _head_bbox(p)
        mem = pair_by_person.get(int(p.track_id))
        state_value = getattr(mem.state, "value", str(mem.state)) if mem is not None else None
        associated_object_id = int(mem.bag_id) if mem is not None else None
        p_uid_getter = getattr(detector, "_person_uid_of", None) if detector is not None else None
        person_uid = p_uid_getter(int(p.track_id)) if callable(p_uid_getter) else None
        person_analyses.append(
            PersonAnalysis(
                track_id=int(p.track_id),
                class_name=str(p.class_name),
                bbox=tuple(float(v) for v in p.bbox),  # type: ignore[arg-type]
                confidence=float(getattr(p, "confidence", 1.0) or 1.0),
                keypoints=keypoints,
                head_visible=head_visible,
                head_bbox=head_bbox,
                trail=_trail_for(tracker, p.track_id),
                state=state_value,
                associated_object_id=associated_object_id,
                person_uid=person_uid,
            )
        )

    object_analyses: List[ObjectAnalysis] = []
    for o in objects:
        mem = pair_by_bag.get(int(o.track_id))
        state_value = getattr(mem.state, "value", str(mem.state)) if mem is not None else None
        associated_person_id = int(mem.person_id) if mem is not None else None
        object_analyses.append(
            ObjectAnalysis(
                track_id=int(o.track_id),
                class_name=str(o.class_name),
                bbox=tuple(float(v) for v in o.bbox),  # type: ignore[arg-type]
                confidence=float(getattr(o, "confidence", 0.0) or 0.0),
                source=str(getattr(o, "source", "yolo") or "yolo"),
                trail=_trail_for(tracker, o.track_id),
                state=state_value,
                associated_person_id=associated_person_id,
                object_uid=getattr(o, "object_uid", None),
            )
        )

    hud = {
        "video": video_name or "video",
        "source_fps": round(float(source_fps), 1) if source_fps else None,
        "analysis_fps": round(float(analysis_fps), 1) if analysis_fps else None,
        "persons": len(person_analyses),
        "objects": len(object_analyses),
        "active_pairs": len(pairs),
    }

    return FrameAnalysis(
        timestamp=float(timestamp),
        frame_number=int(frame_number),
        persons=person_analyses,
        objects=object_analyses,
        associations=associations,
        states=states,
        event=event,
        hud=hud,
    )
