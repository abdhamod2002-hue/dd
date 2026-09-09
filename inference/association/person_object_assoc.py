"""Person–Object association shared dataclasses.

MASTER_REPAIR_PLAN P1-10: the ``PersonObjectAssociator`` class that used to
live here (self-described "core contribution") was DEAD CODE — it was
instantiated once in ``inference/pipeline.py`` and its ``update()`` was never
called by any production path. Per the audit it was removed; the association
that actually drives events is performed inside the authoritative temporal
engine (``littering_event_detector.LitteringEventDetector``, via
``_select_primary_associations``), which enforces a documented one-bag-per-
person invariant (see P1-4's known-limitation note there and the pinning test
in ``tests/test_correctness_audit.py``).

What remains here are the shared input/config dataclasses that are still
referenced by the pipeline config and tests:

* ``Keypoints``   — MoveNet-style keypoints (missing = None).
* ``Track``       — one tracked entity (person or object) in a single frame.
* ``AssociationConfig`` — class-gating + proximity tuning knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------- #
# Inputs
# ---------------------------------------------------------------------- #
@dataclass
class Keypoints:
    """MoveNet-style keypoints. Only what we need; missing = None."""

    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None
    left_shoulder: Optional[Tuple[float, float]] = None
    right_shoulder: Optional[Tuple[float, float]] = None
    torso_center: Optional[Tuple[float, float]] = None  # mid-shoulder
    nose: Optional[Tuple[float, float]] = None          # head/face visibility only
    nose_confidence: float = 0.0


@dataclass
class Track:
    """One tracked entity (person or object) in a single frame."""

    track_id: int
    class_name: str
    centroid: Tuple[float, float]            # (x, y) in pixels
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    keypoints: Optional[Keypoints] = None    # only for persons
    confidence: float = 1.0                  # detector confidence (ByteTrack/YOLO score)
    source: str = "yolo"                     # yolo | color | novelty (P2-2: legacy "csrt_fallback" retired)
    object_uid: Optional[int] = None         # stable logical id across track-id churn (objects only)


@dataclass
class AssociationConfig:
    # Classes eligible to be a litter candidate object
    # NOTE: must cover ALL classes of the production litter model
    # (inference/detection/weights/best.pt): bottle, juice-cup, nescafe,
    # plate, tissue — plus COCO fallback names (bottle, cup). Matching is
    # substring-based, so each entry below matches
    # any detected class containing it.
    litter_candidate_classes: Tuple[str, ...] = (
        "plastic bottle", "bottle", "cup", "can", "tissue paper",
        "paper", "wrapper", "trash", "waste", "bag", "garbage bag",
        "cardboard", "nescafe", "plate", "tissue",
    )

    bind_radius: float = 60.0          # px: wrist→object to call it "held"
    torso_radius: float = 110.0        # px: fallback torso→object
    min_persistence: int = 3           # frames of proximity to establish a pair
    persistence_window: float = 1.0    # seconds; rolling window for persistence
    reassoc_window: float = 1.5        # seconds to rebind after ID switch
    rebind_max_distance: float = 150.0  # px: max centroid jump for rebind
    rebind_max_size_ratio: float = 1.6  # bbox area ratio cap for rebind
    stationary_speed: float = 8.0      # px/s below this = stationary
    ground_band_ratio: float = 0.6     # y > height*this → "low/ground"
    frame_height: float = 480.0        # for ground band; set from capture
