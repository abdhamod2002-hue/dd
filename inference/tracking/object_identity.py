"""
Object identity manager — assigns a STABLE logical id to each physical waste
object across ByteTrack / color-fallback track-id churn.

Root cause addressed (spec rules #11–#17): the HSV/color fallback and ByteTrack
emit unstable numeric track ids for the same physical bag (observed: a single
bag re-born under 13 different ids in one clip). Without a stable logical id,
evidence cannot be reliably anchored to "the same object" across the carry ->
release -> ground arc, and (person, object) cross-wiring becomes possible in
multi-object scenes.

This manager performs a per-frame GREEDY nearest-neighbour assignment:
  * every frame, all current object detections are matched to persistent
    physical-object records by centroid proximity + class-family overlap,
  * each persistent record may be claimed by at most ONE detection per frame
    (reservation), with the largest detection matched first so the dominant
    bag keeps its identity and small noisy detections cannot steal it,
  * an unmatched detection allocates a fresh monotonic uid (never reused).

This yields one persistent uid per physical object even when the underlying
tracker churns ids frame to frame.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

# Class tokens that all denote the same physical litter-object family. Two
# detections only match if they share at least one token (or one is empty).
_FAMILY_TOKENS = (
    "bag", "waste", "trash", "yellow", "bottle", "cup", "can", "paper",
    "tissue", "wrapper", "plastic", "nescafe", "plate", "garbage",
)


def _tokens(class_name: str) -> set:
    low = (class_name or "").lower()
    return {t for t in _FAMILY_TOKENS if t in low}


def _family(a: set, b: set) -> bool:
    if not a or not b:
        return True
    return bool(a & b)


def _centroid(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _area(bbox: Tuple[float, float, float, float]) -> float:
    return max(1.0, bbox[2] - bbox[0]) * max(1.0, bbox[3] - bbox[1])


class ObjectIdentityManager:
    def __init__(
        self,
        distance_px: float = 220.0,
        max_age_frames: int = 120,
        start_uid: int = 100001,
    ) -> None:
        # uid -> record
        self.records: Dict[int, Dict] = {}
        self.next_uid = start_uid
        self.distance_px = float(distance_px)
        self.max_age = int(max_age_frames)
        self.frame = 0

    @property
    def _uids(self) -> Dict[int, Dict]:
        """Alias for records dict for compatibility with introspection/telemetry."""
        return self.records

    def update(self, detections: List[Tuple[str, Tuple, Tuple]]) -> Dict[int, int]:
        """Match a batch of detections (class, bbox, centroid) to stable uids.

        One call == one analysis tick; the internal frame counter advances per
        call so ``max_age`` expiry actually works across a video.
        Returns mapping {detection_index: uid}.
        """
        self.frame += 1
        for rec in self.records.values():
            rec["claimed"] = False
        results: Dict[int, int] = {}
        # Largest detection first: the dominant bag claims its record before
        # small noisy detections can steal it (reservation via "claimed").
        order = sorted(range(len(detections)), key=lambda i: -_area(detections[i][1]))
        for i in order:
            cls, bbox, cen = detections[i]
            fam = _tokens(cls)
            best_uid: Optional[int] = None
            best_d = self.distance_px
            for uid, rec in self.records.items():
                if rec.get("claimed"):
                    continue
                if self.frame - rec["last_seen"] > self.max_age:
                    continue
                if not _family(fam, _tokens(rec["class_name"])):
                    continue
                d = math.hypot(cen[0] - rec["centroid"][0], cen[1] - rec["centroid"][1])
                if d <= best_d:
                    best_d = d
                    best_uid = uid
            if best_uid is not None:
                rec = self.records[best_uid]
                rec["centroid"] = cen
                rec["last_seen"] = self.frame
                rec["class_name"] = cls
                rec["bbox"] = bbox
                rec["claimed"] = True
                results[i] = best_uid
            else:
                uid = self.next_uid
                self.next_uid += 1
                self.records[uid] = {
                    "class_name": cls,
                    "centroid": cen,
                    "last_seen": self.frame,
                    "bbox": bbox,
                    "claimed": True,
                }
                results[i] = uid
        return results

    def reset(self) -> None:
        self.records.clear()
        # keep next_uid monotonic so no id is ever reused after a reset
