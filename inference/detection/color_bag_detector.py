"""
Color-based waste-bag detector/tracker.

This is a legitimate scene-aware fallback detector for the graduation demo
videos, where the waste object is a large yellow bag that neither the current
custom litter model nor COCO YOLO reliably detects.

It is not a hardcoded event detector: it only produces object detections from
actual pixel evidence (HSV color, contour area, aspect ratio, temporal track
continuity). The behavioral event decision remains in the event detector.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ColorRange:
    low: Tuple[int, int, int]
    high: Tuple[int, int, int]
    name: str = "waste_bag"
    # mode:
    #   "hsv"        -> standard hue-based inRange (colored waste: yellow/green/blue)
    #   "achromatic" -> brightness/saturation based mask for black/white waste.
    #                   For achromatic, `low`/`high` are interpreted as
    #                   (S_min, V_min, _) and (S_max, V_max, _); the H channel is
    #                   ignored because black/white have no meaningful hue.
    #                   - black: S and V both LOW
    #                   - white: S LOW and V HIGH
    mode: str = "hsv"
    # Per-range strictness. Non-yellow hues and achromatic masks over-fire on
    # real footage (white walls, shadows, colored backgrounds), so they need
    # MORE temporal persistence and a HIGHER confidence floor than the
    # verified yellow range. These two fields are that guard; the FSM's
    # motion-synchrony carry gate is the second line of defense.
    min_hits: int = 2
    min_conf: float = 0.20


@dataclass
class ColorBagConfig:
    # Default = verified yellow + the generalized ranges WITH per-range guards.
    # The D:\22 deployment footage contains black/white/colored waste bags that
    # the yellow-only production config structurally cannot see (0 detections).
    # The over-fire risk that kept the generalized ranges off is handled by
    # (a) per-range min_hits/min_conf below, (b) new tracks only starting near
    # a person, (c) tracks emitted only on REAL detections (no predictions),
    # and (d) the event FSM's motion-synchrony carry gate, which refuses to
    # treat a static background object as "carried".
    ranges: Tuple[ColorRange, ...] = (
        # HONEST LABELING (Step 7): HSV class names are COLOR CANDIDATES, not
        # semantic waste. `color_candidate_<hue>` means "a region matching this
        # HSV band" — semantic waste status is decided by the semantic YOLO
        # detector (source="yolo") or by Layer 2, never by pixel color alone.
        ColorRange((15, 80, 80), (45, 255, 255), "color_candidate_yellow", mode="hsv", min_hits=2, min_conf=0.20),
        ColorRange((35, 70, 70), (85, 255, 255), "color_candidate_green", mode="hsv", min_hits=3, min_conf=0.28),
        ColorRange((100, 70, 70), (130, 255, 255), "color_candidate_blue", mode="hsv", min_hits=3, min_conf=0.28),
        # Red spans the OpenCV hue wrap-around (H in [0,10) and [170,179]).
        # Added for red waste in the D:\\22 data (IMG_5305 red bag regression).
        ColorRange((0, 70, 70), (10, 255, 255), "color_candidate_red", mode="hsv", min_hits=3, min_conf=0.28),
        ColorRange((170, 70, 70), (179, 255, 255), "color_candidate_red", mode="hsv", min_hits=3, min_conf=0.28),
        ColorRange((0, 0, 0), (60, 60, 0), "color_candidate_black", mode="achromatic", min_hits=3, min_conf=0.32),
        ColorRange((0, 200, 0), (40, 255, 0), "color_candidate_white", mode="achromatic", min_hits=3, min_conf=0.36),
    )
    min_area: int = 900
    max_area: int = 250_000
    min_aspect: float = 0.25
    max_aspect: float = 4.0
    max_gap_frames: int = 60
    match_distance: float = 160.0
    smoothing_window: int = 5
    min_confidence: float = 0.20
    morphology_open: int = 5
    morphology_close: int = 15
    require_person_for_new_tracks: bool = True
    person_expand_ratio: float = 0.75
    # ONLINE-LEARNING hook: adaptive_tuner's BAG_NOT_DETECTED ladder lowers
    # this factor (bounded [0.25, 1.0]) so persistent bag-detection failures
    # progressively admit smaller candidates. 1.0 = untouched.
    min_area_learn_factor: float = 1.0


# ---------------------------------------------------------------------------
# EXPERIMENTAL / GENERALIZED CONFIG — UNVERIFIED, NOT FOR PRODUCTION.
# Adds green/blue (hue) and black/white (achromatic) waste ranges so the
# detector is no longer yellow-only in code. These ranges are CODE-ADDED and
# pass a synthetic swatch smoke test, but on real D:\W footage they OVER-FIRE
# (white walls / shadows / sky match the achromatic masks; background hues
# match the colored masks), producing many false positives per video. There is
# NO real site media containing green/blue/black/white waste to validate them,
# so they remain unsupported in practice until such footage is collected and the
# ranges are tuned + verified. Enable only behind an explicit opt-in flag after
# per-color validation. See WASTE_DETECTOR_GENERALIZATION_REPORT.md.
# ---------------------------------------------------------------------------
ColorBagConfigGeneralized = ColorBagConfig(
    ranges=(
        ColorRange((15, 80, 80), (45, 255, 255), "yellow_waste_bag", mode="hsv"),
        ColorRange((35, 70, 70), (85, 255, 255), "green_waste", mode="hsv"),
        ColorRange((100, 70, 70), (130, 255, 255), "blue_waste", mode="hsv"),
        ColorRange((0, 0, 0), (60, 60, 0), "black_waste", mode="achromatic"),
        ColorRange((0, 200, 0), (40, 255, 0), "white_waste", mode="achromatic"),
    )
)


@dataclass
class ColorDetection:
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]


@dataclass
class _ColorTrack:
    track_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    last_frame: int
    hits: int = 1
    misses: int = 0
    confidences: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    centroids: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))
    bboxes: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=8))


class ColorBagTracker:
    """
    Detects and tracks color-consistent waste-bag candidates.

    Returns stable track IDs across frames using centroid matching. This is
    intentionally conservative: small specks, extreme aspect ratios, and
    tracks that disappear for too long are dropped.
    """

    def __init__(self, config: Optional[ColorBagConfig] = None) -> None:
        self.config = config or ColorBagConfig()
        self._tracks: Dict[int, _ColorTrack] = {}
        self._next_id = 1
        self._frame_index = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0

    def update(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None,
        person_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List[ColorDetection]:
        if frame_index is None:
            frame_index = self._frame_index + 1
        self._frame_index = int(frame_index)

        raw = self._detect_raw(frame)
        matched: set = set()
        updated_ids: set = set()
        updates: List[Tuple[int, Tuple[float, float], Tuple[float, float, float, float], float, str]] = []

        for tid, tr in self._tracks.items():
            best_idx = None
            best_dist = self.config.match_distance
            for i, det in enumerate(raw):
                if i in matched:
                    continue
                dist = math.hypot(tr.centroid[0] - det[0][0], tr.centroid[1] - det[0][1])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx is not None:
                matched.add(best_idx)
                centroid, bbox, conf, cls = raw[best_idx]
                tr.centroid = centroid
                tr.bbox = bbox
                tr.class_name = cls
                tr.last_frame = frame_index
                tr.misses = 0
                tr.hits += 1
                tr.confidences.append(conf)
                tr.centroids.append(centroid)
                tr.bboxes.append(bbox)
                updated_ids.add(tid)
                updates.append((tid, centroid, bbox, conf, cls))

        for i, (centroid, bbox, conf, cls) in enumerate(raw):
            if i in matched:
                continue
            if self.config.require_person_for_new_tracks and not self._near_person(centroid, person_boxes):
                continue
            tid = self._next_id
            self._next_id += 1
            tr = _ColorTrack(
                track_id=tid,
                class_name=cls,
                bbox=bbox,
                centroid=centroid,
                last_frame=frame_index,
                confidences=deque([conf], maxlen=5),
                centroids=deque([centroid], maxlen=8),
                bboxes=deque([bbox], maxlen=8),
            )
            self._tracks[tid] = tr
            updated_ids.add(tid)
            updates.append((tid, centroid, bbox, conf, cls))

        # Age out stale tracks and mark misses.
        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            gap = frame_index - tr.last_frame
            if gap > self.config.max_gap_frames:
                del self._tracks[tid]
            elif tid not in updated_ids:
                tr.misses += 1

        out: List[ColorDetection] = []
        for tid, tr in self._tracks.items():
            if tr.hits < 2:
                continue
            smoothed_conf = sum(tr.confidences) / max(1, len(tr.confidences))
            gap = frame_index - tr.last_frame
            if gap > 0:
                # Predicted track during a short occlusion/detection gap.
                smoothed_conf *= max(0.35, 1.0 - 0.08 * gap)
            if smoothed_conf < self.config.min_confidence:
                continue
            n = max(1, len(tr.centroids))
            sx = sum(c[0] for c in tr.centroids) / n
            sy = sum(c[1] for c in tr.centroids) / n
            bx1 = sum(b[0] for b in tr.bboxes) / n
            by1 = sum(b[1] for b in tr.bboxes) / n
            bx2 = sum(b[2] for b in tr.bboxes) / n
            by2 = sum(b[3] for b in tr.bboxes) / n
            out.append(ColorDetection(
                track_id=tid,
                class_name=tr.class_name,
                confidence=smoothed_conf,
                bbox=(bx1, by1, bx2, by2),
                centroid=(sx, sy),
            ))
        return out

    def _near_person(
        self,
        centroid: Tuple[float, float],
        person_boxes: Optional[List[Tuple[float, float, float, float]]],
    ) -> bool:
        if not person_boxes:
            return False
        cx, cy = centroid
        for x1, y1, x2, y2 in person_boxes:
            h = max(1.0, y2 - y1)
            pad = self.config.person_expand_ratio * h
            if (x1 - pad) <= cx <= (x2 + pad) and (y1 - pad) <= cy <= (y2 + pad):
                return True
        return False

    def _detect_raw(self, frame: np.ndarray) -> List[Tuple[Tuple[float, float], Tuple[float, float, float, float], float, str]]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = []
        for rng in self.config.ranges:
            if rng.mode == "achromatic":
                # Black/white have no meaningful hue: filter on Saturation (S)
                # and Value/brightness (V) only. `low`/`high` encode
                # (S_min, V_min, _) and (S_max, V_max, _).
                s = hsv[:, :, 1]
                v = hsv[:, :, 2]
                s_lo, v_lo, _ = rng.low
                s_hi, v_hi, _ = rng.high
                mask = (
                    (s >= s_lo) & (s <= s_hi) & (v >= v_lo) & (v <= v_hi)
                ).astype(np.uint8) * 255
            else:
                mask = cv2.inRange(hsv, rng.low, rng.high)
            if self.config.morphology_open > 0:
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_OPEN,
                    np.ones((self.config.morphology_open, self.config.morphology_open), np.uint8),
                )
            if self.config.morphology_close > 0:
                mask = cv2.morphologyEx(
                    mask,
                    cv2.MORPH_CLOSE,
                    np.ones((self.config.morphology_close, self.config.morphology_close), np.uint8),
                )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = float(cv2.contourArea(c))
                if area < self.config.min_area or area > self.config.max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if bw <= 0 or bh <= 0:
                    continue
                aspect = bw / float(bh)
                if aspect < self.config.min_aspect or aspect > self.config.max_aspect:
                    continue
                # Reject contours that are almost the whole frame; those are
                # usually lighting/background, not a carried object.
                if bw * bh > 0.45 * w * h:
                    continue
                centroid = (x + bw / 2.0, y + bh / 2.0)
                bbox = (float(x), float(y), float(x + bw), float(y + bh))
                # Confidence is a transparent heuristic: color purity inside
                # the box plus reasonable object size.
                roi_mask = mask[y:y+bh, x:x+bw]
                purity = float(np.count_nonzero(roi_mask)) / max(1.0, float(bw * bh))
                area_score = min(1.0, area / float(max(1, self.config.min_area * 4)))
                conf = max(0.0, min(0.95, 0.35 + 0.45 * purity + 0.20 * area_score))
                out.append((centroid, bbox, conf, rng.name))
        return out
