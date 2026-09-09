"""
Class-agnostic NOVELTY (scene-change) detection.

WHY THIS MODULE EXISTS
----------------------
Every prior waste detector in this project needed training data that matched
each waste *type* / *colour*:

  * HSV yellow-bag detector  -> only yellow, verified on D:\\W footage.
  * TACO / COCO boxes        -> only the classes in those datasets.
  * 8-colour Roboflow set    -> never materialised as a real 8-colour set.

All of them FAIL on "a new object appears that isn't yellow and isn't in the
training set" (a pen, a tissue, a bottle of a colour we never labelled). That
is a genuine, recurring blind spot.

This module takes the opposite stance: it does NOT try to *recognise* waste.
It recognises SCENE CHANGE. On a STATIC camera (verified for the D:\\W videos
in the camera-stability pre-check), if a region of the frame changes and then
stays static long enough, that region is a newly-appeared object. Whether it is
a yellow bag, a pen, or a tissue is irrelevant to *detection* — it is flagged
as a "novel object" candidate and can be handed to the same temporal littering
state machine that already decides carry -> release -> ground -> depart.

It is therefore a COMPLEMENT to the HSV yellow detector, never a replacement.
It is feature-flagged (NOVELTY_DETECTION_ENABLED) and SEPARATE: the HSV path
(inference/detection/color_bag_detector.py) is NEVER touched by this code.

HONESTY RULE (explicit)
-----------------------
The box label is "detected_object" with a confidence, NOT a waste class. We
detected a *change*; we did not *classify* rubbish. Any report that uses this
source must state that clearly — claiming "found a tissue" would be a
fabrication; "found a new static object here" is the true, supportable claim.

MECHANISM (per the agreed design)
---------------------------------
1. background model   : median of the first N frames (warmup), frozen as a
                        scene anchor. (A rolling median variant exists too, but
                        it would eventually absorb a static object, so warmup is
                        the default for reliable "appeared and stayed" detection.)
2. change detection    : |frame - background| thresholded; person regions are
                        ERASED from the change mask using the YOLO person boxes
                        so a walking/standing person is never flagged as novelty.
3. size / shape filter : drop tiny noise and drop regions that cover too much of
                        the frame (global lighting shifts, whole-scene change).
4. temporal persistence: change regions are tracked by centroid; a region must
                        persist for MIN_STATIONARY_FRAMES (8) before it is
                        emitted as a "detected_object" (reuses the same
                        stationary-gate idea as the event detector).
5. person association  : the track is linked to the nearest person (for face
                        evidence) but is NOT required to be near a person to be
                        emitted — a litter object is often dropped *away* from
                        the body.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

from inference.detection.yolo_detector import TrackedDetection


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class NoveltyConfig:
    # Master switch. When False the detector returns [] every frame and does no
    # work (the live pipeline can leave it enabled in config without effect).
    enabled: bool = False

    # --- background model -------------------------------------------------- #
    # Number of warmup frames whose median becomes the frozen scene anchor.
    bg_frames: int = 25
    # Internal processing width. Change detection runs at this resolution for
    # speed/noise-robustness; boxes are scaled back to full-res on output.
    proc_width: int = 640
    # "warmup"  -> median of first bg_frames frames, then frozen (default,
    #              reliable for "appeared and stayed").
    # "rolling" -> median of last bg_frames frames, refreshed each tick
    #              (absorbs static objects over time; good only for catching the
    #              *appearance* event, not for holding it).
    bg_mode: str = "warmup"

    # --- change detection -------------------------------------------------- #
    change_thresh: int = 25          # absdiff threshold (0-255)
    morph_open: int = 3
    morph_close: int = 7

    # --- person masking ---------------------------------------------------- #
    # Expand each YOLO person box by this fraction of its HEIGHT before erasing
    # its change. Keeps a carried/adjacent object from being classed as novelty
    # while the person is still there. Small (0.2) so a bag dropped at the feet
    # is NOT erased and can be detected once the person steps away.
    person_mask_pad: float = 0.2

    # --- size / shape filter (all in PROC-space pixels) -------------------- #
    min_area: int = 600              # below this = noise
    max_area_ratio: float = 0.30     # above this fraction of the frame = global
    min_aspect: float = 0.15         # reject thin slivers (edges/compression)
    max_aspect: float = 7.0

    # --- temporal persistence --------------------------------------------- #
    # Reuses the event detector's MIN_STATIONARY_FRAMES idea: a region must be
    # seen this many ticks (roughly 1 s at 8 fps) before it is emitted.
    stationary_frames: int = 8
    match_distance: float = 70.0     # proc-px centroid match for track continuity
    decay_frames: int = 12           # missed ticks before a track dies
    min_confidence: float = 0.30

    # --- person association (for face evidence only) ----------------------- #
    assoc_max_distance: float = 250.0  # full-res px: nearest person within this

    # --- output branding --------------------------------------------------- #
    class_name: str = "detected_object"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_yaml(cls, path: str = os.path.join("config", "events.yaml")) -> "NoveltyConfig":
        """Read NOVELTY_* keys from config/events.yaml (flat style)."""
        if not os.path.exists(path):
            return cls()
        data: Dict[str, object] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if not line or ":" not in line or line.startswith(("-", " ", "\t")):
                    continue
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if not key.startswith("NOVELTY_"):
                    continue
                name = key[len("NOVELTY_"):].lower()
                if val == "":
                    data[name] = None
                elif val.lower() in {"true", "false"}:
                    data[name] = val.lower() == "true"
                elif val.lower() in {"null", "none", "~"}:
                    data[name] = None
                else:
                    try:
                        data[name] = int(val) if val.isdigit() or (val.startswith("-") and val[1:].isdigit()) else float(val)
                    except ValueError:
                        data[name] = val
        allowed = {f.name for f in cls.__dataclass_fields__.values()} if hasattr(cls, "__dataclass_fields__") else set()
        # build kwargs only for fields that exist
        kwargs = {k: v for k, v in data.items() if k in cls.__annotations__}
        try:
            return cls(**kwargs)
        except TypeError:
            return cls()


# --------------------------------------------------------------------------- #
# Internal track
# --------------------------------------------------------------------------- #
@dataclass
class _NovelTrack:
    track_id: int
    bbox: Tuple[float, float, float, float]   # full-res
    centroid: Tuple[float, float]             # full-res
    first_frame: int
    last_frame: int
    hits: int = 1
    misses: int = 0
    max_area: float = 0.0
    person_track_id: Optional[int] = None
    emitted: bool = False


class NoveltyDetector:
    """
    Background-subtraction novelty detector.

    update(frame, frame_index, person_boxes) -> List[TrackedDetection]

    Each returned TrackedDetection has source="novelty" and class_name
    "detected_object" (configurable). Only tracks that have persisted for
    ``stationary_frames`` are returned — transient blips are suppressed.
    """

    def __init__(self, config: Optional[NoveltyConfig] = None) -> None:
        self.config = config or NoveltyConfig()
        self._buf: Deque[np.ndarray] = deque(maxlen=max(2, int(self.config.bg_frames)))
        self._bg: Optional[np.ndarray] = None
        self._tracks: Dict[int, _NovelTrack] = {}
        self._next_id = 1
        self._frame_index = 0
        self._scale: float = 1.0          # proc_width / frame_width
        self._proc_h: int = 0
        self._proc_w: int = 0
        self._anchored = False
        # diagnostics for evaluation harnesses
        self.last_raw_region_count: int = 0
        self.last_persistent_count: int = 0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self._buf.clear()
        self._bg = None
        self._tracks.clear()
        self._next_id = 1
        self._frame_index = 0
        self._anchored = False

    @property
    def warmup_ready(self) -> bool:
        return self._anchored

    # ------------------------------------------------------------------ #
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        self._scale = float(self.config.proc_width) / max(1, w)
        self._proc_w = int(round(w * self._scale))
        self._proc_h = int(round(h * self._scale))
        small = cv2.resize(frame, (self._proc_w, self._proc_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return gray

    def _scale_box_to_proc(self, bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (int(x1 * self._scale), int(y1 * self._scale),
                int(x2 * self._scale), int(y2 * self._scale))

    def _scale_box_to_full(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        inv = 1.0 / max(1e-6, self._scale)
        x1, y1, x2, y2 = bbox
        return (x1 * inv, y1 * inv, x2 * inv, y2 * inv)

    # ------------------------------------------------------------------ #
    def raw_region_count(
        self,
        frame: np.ndarray,
        person_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> int:
        """Count current change regions WITHOUT touching tracker state.

        Used by evaluation harnesses to sample the raw detection rate on a
        fixed frame cadence (e.g. every 15th frame) independent of the
        analysis-throttle that drives update(). Does not update the
        background or any track, so it is safe to call arbitrarily.
        """
        if not self.config.enabled or self._bg is None:
            return 0
        gray = self._preprocess(frame)
        return len(self._change_regions(gray, person_boxes))

    # ------------------------------------------------------------------ #
    def _update_background(self, gray: np.ndarray) -> None:
        if self.config.bg_mode == "rolling":
            self._buf.append(gray)
            if len(self._buf) >= 2:
                self._bg = np.median(np.stack(list(self._buf), axis=0), axis=0).astype(np.uint8)
                self._anchored = True
            return
        # warmup (default): collect first bg_frames, then freeze the median.
        if not self._anchored:
            self._buf.append(gray)
            if len(self._buf) >= max(2, int(self.config.bg_frames)):
                self._bg = np.median(np.stack(list(self._buf), axis=0), axis=0).astype(np.uint8)
                self._anchored = True

    # ------------------------------------------------------------------ #
    def _change_regions(
        self,
        gray: np.ndarray,
        person_boxes: Optional[List[Tuple[float, float, float, float]]],
    ) -> List[Tuple[Tuple[float, float, float, float], float]]:
        """Return list of (proc_bbox, area) change regions after person masking."""
        if self._bg is None:
            return []
        diff = cv2.absdiff(gray, self._bg)
        _, mask = cv2.threshold(diff, self.config.change_thresh, 255, cv2.THRESH_BINARY)
        if self.config.morph_open > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                   np.ones((self.config.morph_open, self.config.morph_open), np.uint8))
        if self.config.morph_close > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((self.config.morph_close, self.config.morph_close), np.uint8))

        # --- erase person regions from the change mask -------------------- #
        if person_boxes:
            for pb in person_boxes:
                x1, y1, x2, y2 = self._scale_box_to_proc(pb)
                ph = max(1.0, y2 - y1)
                pad = self.config.person_mask_pad * ph
                ex1 = int(max(0, x1 - pad))
                ey1 = int(max(0, y1 - pad))
                ex2 = int(min(self._proc_w, x2 + pad))
                ey2 = int(min(self._proc_h, y2 + pad))
                if ex2 > ex1 and ey2 > ey1:
                    cv2.rectangle(mask, (ex1, ey1), (ex2, ey2), 0, -1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        frame_area = float(self._proc_w * self._proc_h)
        out: List[Tuple[Tuple[float, float, float, float], float]] = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < self.config.min_area:
                continue
            if area > self.config.max_area_ratio * frame_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / float(bh)
            if aspect < self.config.min_aspect or aspect > self.config.max_aspect:
                continue
            out.append(((float(x), float(y), float(x + bw), float(y + bh)), area))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _inside_person(
        centroid_proc: Tuple[float, float],
        person_map: Dict[int, Tuple[float, float, float, float]],
        scale: float = 1.0,
    ) -> bool:
        """True if the (proc-space) region centroid lies inside a person box."""
        cx = centroid_proc[0] / max(1e-6, scale)
        cy = centroid_proc[1] / max(1e-6, scale)
        for pb in person_map.values():
            if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
                return True
        return False

    # ------------------------------------------------------------------ #
    def _associate_person(
        self,
        region_full: Tuple[float, float, float, float],
        person_boxes_full: Optional[Dict[int, Tuple[float, float, float, float]]],
    ) -> Optional[int]:
        if not person_boxes_full:
            return None
        cx = (region_full[0] + region_full[2]) / 2.0
        cy = (region_full[1] + region_full[3]) / 2.0
        best_pid = None
        best_d = self.config.assoc_max_distance
        for pid, pb in person_boxes_full.items():
            px, py = (pb[0] + pb[2]) / 2.0, (pb[1] + pb[3]) / 2.0
            d = float(np.hypot(cx - px, cy - py))
            if d < best_d:
                best_d = d
                best_pid = pid
        # Also reject if the region centroid is still INSIDE a person box
        # (carried object, not yet a scene change).
        for pb in person_boxes_full.values():
            if pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]:
                return None
        return best_pid

    # ------------------------------------------------------------------ #
    def update(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None,
        person_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
        person_map: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
    ) -> List[TrackedDetection]:
        """
        One analysis tick.

        Args:
            frame         : BGR numpy frame (any resolution).
            frame_index   : monotonic tick index.
            person_boxes  : full-res YOLO person bboxes (list) — used to MASK
                            person change. Optional but strongly recommended.
            person_map    : {person_track_id: bbox} — used to ASSOCIATE a novel
                            object with the nearest person for face evidence.

        Returns:
            List[TrackedDetection] with source="novelty" for every track that
            has persisted >= stationary_frames. Empty until warmup completes
            and a region stabilises.
        """
        if not self.config.enabled:
            return []
        if frame_index is None:
            frame_index = self._frame_index + 1
        self._frame_index = int(frame_index)

        gray = self._preprocess(frame)
        self._update_background(gray)
        regions = self._change_regions(gray, person_boxes)
        self.last_raw_region_count = len(regions)

        # match regions to existing tracks by proc-space centroid distance
        region_centroids = [((r[0][0] + r[0][2]) / 2.0, (r[0][1] + r[0][3]) / 2.0) for r in regions]
        matched: set = set()
        updated_ids: set = set()
        for tid, tr in self._tracks.items():
            best_i = None
            best_d = self.config.match_distance
            # tr.centroid is already in PROC space (matching region_centroids).
            tcx, tcy = tr.centroid[0], tr.centroid[1]
            for i, rc in enumerate(region_centroids):
                if i in matched:
                    continue
                d = float(np.hypot(rc[0] - tcx, rc[1] - tcy))
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i is not None:
                matched.add(best_i)
                rb = regions[best_i][0]
                area = regions[best_i][1]
                tr.bbox = self._scale_box_to_full(rb)
                tr.centroid = ((rb[0] + rb[2]) / 2.0, (rb[1] + rb[3]) / 2.0)
                tr.last_frame = frame_index
                tr.hits += 1
                tr.misses = 0
                tr.max_area = max(tr.max_area, area)
                if tr.person_track_id is None and person_map:
                    tr.person_track_id = self._associate_person(tr.bbox, person_map)
                updated_ids.add(tid)

        for i, rb in enumerate(regions):
            if i in matched:
                continue
            area = regions[i][1]
            bbox_full = self._scale_box_to_full(rb[0])
            centroid = ((rb[0][0] + rb[0][2]) / 2.0, (rb[0][1] + rb[0][3]) / 2.0)
            # Do not start a track for a region whose centroid is still INSIDE a
            # current person box. That is the carrier's own body or a bag still
            # at their feet — not yet a scene change. Once the person steps away
            # the same static region reappears outside any person box and is
            # tracked normally.
            if person_map and self._inside_person(centroid, person_map, self._scale):
                continue
            pid = self._associate_person(bbox_full, person_map) if person_map else None
            tr = _NovelTrack(
                track_id=self._next_id,
                bbox=bbox_full,
                centroid=centroid,
                first_frame=frame_index,
                last_frame=frame_index,
                hits=1,
                misses=0,
                max_area=area,
                person_track_id=pid,
            )
            self._tracks[self._next_id] = tr
            self._next_id += 1
            updated_ids.add(self._next_id - 1)

        # age out stale tracks
        for tid in list(self._tracks.keys()):
            tr = self._tracks[tid]
            if tid not in updated_ids:
                tr.misses += 1
                if tr.misses > self.config.decay_frames:
                    del self._tracks[tid]

        # emit only persistent tracks
        out: List[TrackedDetection] = []
        for tid, tr in self._tracks.items():
            if tr.hits < self.config.stationary_frames:
                continue
            area_norm = min(1.0, tr.max_area / max(1.0, self.config.min_area * 4.0))
            persistence_norm = min(1.0, tr.hits / max(1, self.config.stationary_frames * 2))
            conf = float(max(self.config.min_confidence,
                             min(0.95, 0.35 + 0.30 * persistence_norm + 0.35 * area_norm)))
            out.append(TrackedDetection(
                track_id=100000 + tid,  # separate raw namespace from YOLO/color
                class_name=self.config.class_name,
                confidence=conf,
                bbox=tuple(float(v) for v in tr.bbox),  # type: ignore[arg-type]
                centroid=(float(tr.centroid[0]) / self._scale,
                          float(tr.centroid[1]) / self._scale),  # full-res
                is_person=False,
                source="novelty",
            ))
        self.last_persistent_count = len(out)
        return out
