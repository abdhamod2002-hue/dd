"""
YOLO Detector — 🟢 wrapper around ultralytics.

Loads ``best.pt`` (the garbage-detection weights from the reference
project) and runs detection per frame. Returns a list of ``Detection``
dataclasses that the pipeline converts into ``Track`` objects after
ByteTrack assigns IDs.

The reference model's classes (plastic bottle, juice cup, tissue paper,
...) are surfaced via :attr:`classes`. Person detection uses the COCO
``person`` class — we load yolov8n.pt (or a configured COCO model) for
people, and the custom ``best.pt`` for litter objects. Two models is
fine on CPU for a demo; we can also fine-tune a single model later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np  # type: ignore


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    centroid: Tuple[float, float]

    @property
    def is_person(self) -> bool:
        return self.class_name.lower() == "person"


# --------------------------------------------------------------------------- #
# Detector-output deduplication (Phase 3).
#
# ONE physical object → ONE semantic detection → ONE track → ONE current box.
#
# Rule (deterministic, detector-level only — no filenames, frame numbers,
# identity history, or event state):
#   1. Group by is_person. Persons and objects NEVER suppress each other
#      (a person overlapping a bag, or two overlapping people, stay distinct).
#   2. Priority order: (source_rank, -confidence, area, bbox).
#      source_rank: yolo=0, color=1, novelty=2 — semantic YOLO detections are
#      authoritative; HSV/novelty proposals NEVER override semantic YOLO waste.
#   3. Greedy keep: a candidate is suppressed when, against any kept box of
#      the SAME group, IoU >= 0.5 OR max(containment either way) >= 0.7.
#      The symmetric max-containment fixes the observed failure where a
#      smaller high-confidence box is kept first and the bigger duplicate
#      (only ~22% of its own area inside the small one) survives the old
#      asymmetric test (IMG_5306 src=4816: two Garbage-Bag boxes, IoU=0.21).
# Applied to the POOLED output of all models in detect() and track(), i.e.
# cross-model duplicates (best.pt vs garbage_bag_v2.pt) are compared — the
# old _dedup_contained() only ever saw one model's output at a time.
# --------------------------------------------------------------------------- #

DEDUP_IOU_THRESH = 0.5
DEDUP_CONTAIN_THRESH = 0.7

_SOURCE_RANK = {"yolo": 0, "color": 1, "novelty": 2}


def _dedup_source_rank(source: str) -> int:
    return _SOURCE_RANK.get(str(source or "yolo").lower(), 0)


def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _box_containments(a, b):
    """Return (fraction of a's area inside b, fraction of b's area inside a)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    aa = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / aa, inter / ab


def boxes_duplicate(a, b,
                    iou_thresh: float = DEDUP_IOU_THRESH,
                    contain_thresh: float = DEDUP_CONTAIN_THRESH) -> bool:
    """Symmetric duplicate predicate for two bboxes (same-group only)."""
    if _box_iou(a, b) >= iou_thresh:
        return True
    ca, cb = _box_containments(a, b)
    return max(ca, cb) >= contain_thresh


def _pooled_dedup_keep_indices(boxes, is_person_flags, confidences,
                               source_ranks) -> list:
    """Greedy keep-list over pooled same-frame detections. Returns kept indices
    in ORIGINAL order (stable output for downstream trackers)."""
    n = len(boxes)
    areas = [max(1e-6, (b[2] - b[0]) * (b[3] - b[1])) for b in boxes]
    order = sorted(range(n),
                   key=lambda i: (source_ranks[i], -confidences[i],
                                  areas[i], tuple(boxes[i])))
    suppressed = [False] * n
    for i in order:
        if suppressed[i]:
            continue
        for j in order:
            if j == i or suppressed[j]:
                continue
            if bool(is_person_flags[j]) != bool(is_person_flags[i]):
                continue
            if boxes_duplicate(tuple(boxes[i]), tuple(boxes[j])):
                suppressed[j] = True
    return [i for i in range(n) if not suppressed[i]]


def deduplicate_detections(dets: List["Detection"]) -> List["Detection"]:
    """Cross-model dedup for Detection lists (used by detect())."""
    if len(dets) < 2:
        return list(dets)
    keep = _pooled_dedup_keep_indices(
        [tuple(d.bbox) for d in dets],
        [bool(d.is_person) for d in dets],
        [float(d.confidence) for d in dets],
        [_dedup_source_rank("yolo") for _ in dets],
    )
    return [dets[i] for i in keep]


def deduplicate_tracked(dets: List["TrackedDetection"]) -> List["TrackedDetection"]:
    """Cross-model dedup for TrackedDetection lists (used by track()).

    Honors each item's source rank: semantic YOLO wins over color/novelty
    proposals describing the same physical object.
    """
    if len(dets) < 2:
        return list(dets)
    keep = _pooled_dedup_keep_indices(
        [tuple(d.bbox) for d in dets],
        [bool(d.is_person) for d in dets],
        [float(d.confidence) for d in dets],
        [_dedup_source_rank(getattr(d, "source", "yolo")) for d in dets],
    )
    return [dets[i] for i in keep]


class YoloDetector:
    """
    Wraps two ultralytics YOLO models:
      * person_model : COCO person (yolov8n.pt by default)
      * litter_model : custom best.pt for trash classes

    Both are loaded lazily so importing this module does not require
    ultralytics/torch at import time (the pipeline imports this only on
    the laptop, not in the test sandbox).
    """

    def __init__(
        self,
        litter_weights: str = "inference/detection/weights/best.pt",
        person_weights: str = "yolov8n.pt",
        person_conf: float = 0.4,
        litter_conf: float = 0.35,
        device: str = "cpu",
        fallback_coco_classes: bool = True,
        color_fallback: bool = True,
        imgsz: int = 640,
        # Dedicated waste-bag model slot. The file ``waste_bag_real_v1.pt``
        # is PERMANENTLY REJECTED (stop-audit 2026-09-01: 0-0.5% agreement
        # with the independent TACO ground-truth detector on held-out videos;
        # trained with zero negative frames). Using, fixing, or retraining
        # that model is prohibited by the user's plan. The default here is
        # ``None``: resolution happens in ``load()`` via
        # ``_resolve_bag_weights()`` — env ``WASTE_BAG_WEIGHTS`` first, then
        # the approved human-annotated ``garbage_bag_v2.pt`` (Roboflow CC BY 4.0
        # dataset, human labels, 1 class "Garbage Bag") if present.
        bag_weights: Optional[str] = None,
        bag_conf: float = 0.25,
    ) -> None:
        self.litter_weights = litter_weights
        self.person_weights = person_weights
        self.person_conf = person_conf
        self.litter_conf = litter_conf
        self.device = device
        self.imgsz = int(imgsz)
        # When the litter model (best.pt) is absent, fall back to emitting
        # non-person COCO classes (bottle, cup, ...) from the person model so
        # the pipeline can still track litter-likely objects. This keeps the
        # demo honest when best.pt is not yet installed.
        self._fallback_coco_classes = fallback_coco_classes
        # Real D:\W demo videos contain a yellow waste bag that neither
        # best.pt nor COCO reliably detects. This fallback produces actual
        # pixel-based object tracks; it does not fabricate events.
        self._color_fallback_enabled = color_fallback
        self.bag_weights = bag_weights
        self.bag_conf = bag_conf
        self._color_tracker = None
        self._color_frame_index = 0
        self._person_model = None
        self._litter_model = None
        self._bag_model = None
        self._litter_classes: Optional[List[str]] = None
        self._bag_classes: Optional[List[str]] = None

    @staticmethod
    def _resolve_bag_weights() -> Optional[str]:
        """Resolve the semantic waste-bag model path (once, at load time).

        Order:
        1. explicit ``WASTE_BAG_WEIGHTS`` env var (if the file exists)
        2. approved human-annotated ``inference/detection/weights/garbage_bag_v2.pt``
        3. ``None`` (slot disabled — no hidden fallback model)

        NOTE: ``waste_bag_real_v1.pt`` is PERMANENTLY REJECTED and is never
        resolved here under any name/path.
        """
        env_path = os.environ.get("WASTE_BAG_WEIGHTS")
        if env_path and os.path.exists(env_path):
            return env_path
        default_path = "inference/detection/weights/garbage_bag_v2.pt"
        if os.path.exists(default_path):
            return default_path
        return None

    def load(self) -> None:
        from ultralytics import YOLO  # type: ignore

        self._person_model = YOLO(self.person_weights)
        if os.path.exists(self.litter_weights):
            self._litter_model = YOLO(self.litter_weights)
            self._litter_classes = list(self._litter_model.names.values())
        else:
            # allow running person-only if litter weights absent
            self._litter_model = None
            self._litter_classes = []
        # Optional dedicated waste-bag model (runs in addition to litter model).
        # Resolved once here (MODEL LOAD ONCE rule): env override first, then
        # the approved human-annotated semantic model if it exists on disk.
        bag_weights = self._resolve_bag_weights()
        self.bag_weights = bag_weights
        if bag_weights and os.path.exists(bag_weights):
            self._bag_model = YOLO(self.bag_weights)
            self._bag_classes = list(self._bag_model.names.values())
        else:
            self._bag_model = None
            self._bag_classes = []

    @property
    def litter_classes(self) -> List[str]:
        base = list(self._litter_classes or [])
        base += list(self._bag_classes or [])
        return base

    def reset_tracking(self) -> None:
        """Reset stateful fallback trackers between videos/camera sessions."""
        self._color_frame_index = 0
        if self._color_tracker is not None:
            self._color_tracker.reset()

    def detect(self, frame) -> List[Detection]:
        """Single-frame detection (no tracking). Used for eval/inspection."""
        if self._person_model is None:
            raise RuntimeError("YoloDetector.load() must be called first")
        out: List[Detection] = []

        # people (+ fallback non-person COCO classes when the litter model is absent)
        for r in self._person_model(frame, conf=self.person_conf, device=self.device, imgsz=self.imgsz, verbose=False):
            for box in r.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = self._person_model.names[cls_id]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                if name.lower() == "person":
                    out.append(Detection("person", conf, (x1, y1, x2, y2), (cx, cy)))
                elif self._litter_model is None and self._fallback_coco_classes:
                    # litter model absent → use non-person COCO classes (bottle, cup,
                    # etc.) as a fallback so the pipeline can still track litter-likely
                    # objects. This is the honest path when best.pt is not installed.
                    out.append(Detection(name, conf, (x1, y1, x2, y2), (cx, cy)))

        # litter (custom model)
        if self._litter_model is not None:
            for r in self._litter_model(frame, conf=self.litter_conf, device=self.device, imgsz=self.imgsz, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self._litter_model.names[cls_id]
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    out.append(Detection(name, conf, (x1, y1, x2, y2), (cx, cy)))

        # dedicated waste-bag model (added in addition to the litter model)
        if self._bag_model is not None:
            for r in self._bag_model(frame, conf=self.bag_conf, device=self.device, imgsz=self.imgsz, verbose=False):
                for box in r.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self._bag_model.names[cls_id]
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    out.append(Detection(name, conf, (x1, y1, x2, y2), (cx, cy)))

        # Pooled cross-model dedup BEFORE any tracking consumer (Phase 3).
        # Kill-switch (default ON): PHASE3_DISABLE_DEDUP=1 restores legacy output.
        if os.environ.get("PHASE3_DISABLE_DEDUP") == "1":
            return out
        return deduplicate_detections(out)

    def track(self, frame, persist: bool = True) -> List["TrackedDetection"]:
        """
        Real ByteTrack tracking via ultralytics model.track(...).

        Returns TrackedDetection objects that carry a STABLE track id
        assigned by ByteTrack across frames (persist=True keeps the
        tracker state between calls). This is the production path —
        the Association/FSM/Voting layers REQUIRE stable ids.

        Persons come from the person model; litter objects come from the
        litter model. Both are tracked independently with their own
        model.track(...) call so their ID spaces never collide.
        """
        if self._person_model is None:
            raise RuntimeError("YoloDetector.load() must be called first")
        out: List[TrackedDetection] = []

        # people (ByteTrack, persist keeps IDs stable across calls)
        for r in self._person_model.track(
            frame, conf=self.person_conf, device=self.device, imgsz=self.imgsz,
            tracker="bytetrack.yaml", persist=persist, verbose=False,
        ):
            out.extend(self._parse_tracked(r, is_person=True, naming_model=self._person_model,
                                           allow_fallback=self._litter_model is None and self._fallback_coco_classes))

        # litter (separate tracker instance via separate model.track calls)
        if self._litter_model is not None:
            for r in self._litter_model.track(
                frame, conf=self.litter_conf, device=self.device, imgsz=self.imgsz,
                tracker="bytetrack.yaml", persist=persist, verbose=False,
            ):
                out.extend(self._parse_tracked(r, is_person=False, naming_model=self._litter_model))

        # dedicated waste-bag model (added in addition to the litter model;
        # detections carry class "waste_bag" and source="yolo" so the event
        # detector's bag logic treats them as real confirmed objects)
        if self._bag_model is not None:
            for r in self._bag_model.track(
                frame, conf=self.bag_conf, device=self.device, imgsz=self.imgsz,
                tracker="bytetrack.yaml", persist=persist, verbose=False,
            ):
                out.extend(self._parse_tracked(r, is_person=False, naming_model=self._bag_model))

        # Pooled cross-model dedup BEFORE tracker input (Phase 3): the
        # per-model ByteTrack instances above each saw only their own model's
        # boxes, so best.pt-vs-garbage_bag_v2.pt (and same-model partial
        # overlap) duplicates are collapsed here, deterministically, with
        # semantic YOLO priority. This list IS the downstream tracker input.
        # Kill-switch (default ON): PHASE3_DISABLE_DEDUP=1 restores legacy output.
        if os.environ.get("PHASE3_DISABLE_DEDUP") != "1":
            out = deduplicate_tracked(out)

        # Color fallback for the real demo videos: if neither the custom litter
        # model nor COCO fallback produced an object, try a conservative
        # HSV/contour waste-bag detector. This is still real inference output;
        # it does not inject events or synthetic tracks.
        has_object = any(not d.is_person for d in out)
        if self._color_fallback_enabled and not has_object:
            person_boxes = [d.bbox for d in out if d.is_person]
            out.extend(self._color_fallback_track(frame, persist=persist, person_boxes=person_boxes))
            # The color tracker emits EVERY established track each frame
            # (yellow/green/blue/black/white fragments can overlap on the same
            # pile). Collapse same-frame duplicates here too — idempotent for
            # the already-deduped YOLO part above. Persons unaffected (groups).
            if os.environ.get("PHASE3_DISABLE_DEDUP") != "1":
                out = deduplicate_tracked(out)

        return out

    def _color_fallback_track(
        self,
        frame,
        persist: bool = True,
        person_boxes: Optional[List[Tuple[float, float, float, float]]] = None,
    ) -> List["TrackedDetection"]:
        try:
            from inference.detection.color_bag_detector import ColorBagTracker
        except Exception:
            return []
        if self._color_tracker is None:
            self._color_tracker = ColorBagTracker()
        if not persist:
            self._color_tracker.reset()
        self._color_frame_index += 1
        try:
            dets = self._color_tracker.update(
                frame,
                frame_index=self._color_frame_index,
                person_boxes=person_boxes or [],
            )
        except Exception:
            return []
        out: List[TrackedDetection] = []
        for d in dets:
            # Keep color IDs in a separate raw namespace so they cannot collide
            # with ByteTrack litter IDs before the +10000 object offset.
            out.append(TrackedDetection(
                track_id=50000 + int(d.track_id),
                class_name=d.class_name,
                confidence=float(d.confidence),
                bbox=d.bbox,
                centroid=d.centroid,
                is_person=False,
                source="color",
            ))
        return out

    @staticmethod
    def _dedup_contained(dets: List["TrackedDetection"], contain_thresh: float = 0.7) -> List["TrackedDetection"]:
        """Suppress partial duplicate detections of the SAME physical object.

        Standard NMS only removes boxes with IoU > ~0.55, so a small box
        fully contained inside a bigger one (e.g. a bottle's cap/neck
        detected separately from the whole bottle) survives and competes in
        tracking. Real-video probing showed these phantoms create ghost
        pairs in the association layer. Rule: process by confidence desc;
        drop any detection whose overlap with an already-kept detection
        covers >= ``contain_thresh`` of ITS OWN area (same person/object
        group). This is standard containment-NMS practice.
        """
        kept: List[TrackedDetection] = []
        for d in sorted(dets, key=lambda t: t.confidence, reverse=True):
            x1, y1, x2, y2 = d.bbox
            area = max(1e-6, (x2 - x1) * (y2 - y1))
            contained = False
            for k in kept:
                if k.is_person != d.is_person:
                    continue
                kx1, ky1, kx2, ky2 = k.bbox
                ix = max(0.0, min(x2, kx2) - max(x1, kx1))
                iy = max(0.0, min(y2, ky2) - max(y1, ky1))
                if (ix * iy) / area >= contain_thresh:
                    contained = True
                    break
            if not contained:
                kept.append(d)
        return kept

    def _parse_tracked(self, r, is_person: bool, naming_model=None, allow_fallback: bool = False) -> List["TrackedDetection"]:
        out: List[TrackedDetection] = []
        boxes = r.boxes
        if boxes.id is None:
            return out
        if naming_model is None:
            naming_model = self._person_model if is_person else self._litter_model
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(float, boxes.xyxy[i].tolist())
            conf = float(boxes.conf[i])
            tid = int(boxes.id[i])
            cls_id = int(boxes.cls[i])
            raw_name = naming_model.names[cls_id]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if is_person and raw_name.lower() == "person":
                out.append(TrackedDetection(
                    track_id=tid, class_name="person", confidence=conf,
                    bbox=(x1, y1, x2, y2), centroid=(cx, cy), is_person=True,
                ))
            elif is_person and allow_fallback:
                # litter model absent → emit non-person COCO classes (bottle, cup, ...)
                # as fallback litter-likely objects, flagged is_person=False so they
                # get namespaced into the object ID space (10000+).
                out.append(TrackedDetection(
                    track_id=tid, class_name=raw_name, confidence=conf,
                    bbox=(x1, y1, x2, y2), centroid=(cx, cy), is_person=False,
                ))
            elif not is_person:
                out.append(TrackedDetection(
                    track_id=tid, class_name=raw_name, confidence=conf,
                    bbox=(x1, y1, x2, y2), centroid=(cx, cy), is_person=False,
                ))
        return self._dedup_contained(out)

    def _model_name_for(self, is_person: bool, cls_id: int) -> str:
        if is_person:
            return "person"
        if self._litter_model is None:
            return f"object_{cls_id}"
        return str(self._litter_model.names.get(cls_id, f"object_{cls_id}"))


@dataclass
class TrackedDetection:
    """A detection with a stable ByteTrack-assigned id."""
    track_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    is_person: bool
    source: str = "yolo"
