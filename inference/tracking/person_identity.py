"""
Stable person-identity manager for the temporal littering event detector.

Resolves raw ByteTrack person track IDs to STABLE logical person UIDs, so
that an ID switch in a crowd scene (ByteTrack re-assigns P2 -> P7) does NOT
orphan the (person, object) pair's temporal history or hand the event to a
bystander. Raw tracker IDs are NOT authoritative event identity (spec rule).

Re-association evidence (priority order):
  1. Direct mapping: this raw track id was seen before -> same uid.
  2. Spatial + temporal continuity: a recently-lost uid reappears under a new
     raw id at a nearby centroid within the re-association window.
  3. Pose/geometry continuity (when keypoints available): bbox size and
     wrist/torso geometry similarity.
  4. Otherwise: mint a fresh uid (genuinely new person).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class _PersonSnap:
    uid: int
    centroid: Tuple[float, float]
    bbox: Tuple[float, float, float, float]
    keypoints: Optional[object] = None
    last_timestamp: float = 0.0
    last_frame: int = -1
    first_timestamp: float = 0.0
    first_frame: int = -1
    raw_ids: list = field(default_factory=list)


class PersonIdentityManager:
    def __init__(self, max_age_frames: int = 45, distance_threshold: float = 200.0,
                 pose_assist: bool = True) -> None:
        self._next_uid: int = 1
        self._raw_to_uid: Dict[int, int] = {}
        self._uids: Dict[int, _PersonSnap] = {}
        self._max_age = int(max_age_frames)
        self._dist_thresh = float(distance_threshold)
        self._pose_assist = bool(pose_assist)
        self._frame: int = 0
        # uids already resolved this frame. A physical person can only occupy
        # ONE detection per frame, so a uid claimed by another raw id in the
        # same frame must never be re-associated to a second raw id.
        self._claimed_this_frame: set = set()

    def reset(self) -> None:
        self._next_uid = 1
        self._raw_to_uid.clear()
        self._uids.clear()
        self._frame = 0

    def _touch(self, uid: int, centroid, bbox, keypoints, timestamp, raw_id: int) -> None:
        snap = self._uids[uid]
        snap.centroid = centroid
        snap.bbox = bbox
        snap.keypoints = keypoints
        snap.last_timestamp = timestamp
        snap.last_frame = int(self._frame)
        if raw_id not in snap.raw_ids:
            snap.raw_ids.append(raw_id)

    def _try_reassociate(self, centroid, bbox, keypoints) -> Optional[int]:
        best_uid: Optional[int] = None
        best_score: float = -1.0
        for uid, snap in self._uids.items():
            gap = self._frame - snap.last_frame
            # gap == 0 means the uid was ALREADY seen this frame under another
            # raw id: two simultaneous detections are two different physical
            # people (a same-frame merge would fuse bystanders into one actor).
            # Re-association is only for DISAPPEARED uids (gap >= 1).
            if gap < 1 or gap > self._max_age:
                continue
            # One uid per frame: never let two fresh raw ids claim the same uid.
            if uid in self._claimed_this_frame:
                continue
            d = math.hypot(centroid[0] - snap.centroid[0], centroid[1] - snap.centroid[1])
            if d > self._dist_thresh:
                continue
            spatial = max(0.0, 1.0 - d / max(self._dist_thresh, 1e-6))
            temporal = max(0.0, 1.0 - gap / max(self._max_age, 1e-6))
            score = 0.6 * spatial + 0.4 * temporal
            if self._pose_assist:
                score += 0.2 * self._geometry_match(snap.bbox, bbox, snap.keypoints, keypoints)
            if score > best_score:
                best_score = score
                best_uid = uid
        if best_uid is not None and best_score >= 0.35:
            return best_uid
        return None

    @staticmethod
    def _geometry_match(bbox_a, bbox_b, kp_a, kp_b) -> float:
        try:
            ha = max(1e-6, bbox_a[3] - bbox_a[1])
            hb = max(1e-6, bbox_b[3] - bbox_b[1])
            ratio = min(ha, hb) / max(ha, hb)
        except Exception:
            ratio = 0.0
        pose_sim = 0.0
        try:
            if kp_a is not None and kp_b is not None:
                pts_a = [kp_a.left_wrist, kp_a.right_wrist, kp_a.torso_center]
                pts_b = [kp_b.left_wrist, kp_b.right_wrist, kp_b.torso_center]
                dims = 0
                agree = 0
                for a, b in zip(pts_a, pts_b):
                    if a is not None and b is not None:
                        dims += 1
                        if math.hypot(a[0] - b[0], a[1] - b[1]) < 0.3 * max(ha, hb):
                            agree += 1
                if dims:
                    pose_sim = agree / dims
        except Exception:
            pose_sim = 0.0
        return 0.6 * ratio + 0.4 * pose_sim

    def resolve(
        self,
        raw_id: int,
        centroid: Tuple[float, float],
        bbox: Tuple[float, float, float, float],
        keypoints: Optional[object],
        frame_index: int,
        timestamp: float,
    ) -> int:
        """
        Map a raw person tracker id observed THIS frame to a stable person uid.

        Priority order (see module docstring):
          1. The raw id was seen before  -> same uid (normal path).
          2. An unmatched raw id that re-appears near a recently-lost uid's
             last position (within max_age_frames / distance_threshold, aided
             by bbox/pose geometry) -> the lost uid (track-id switch recovery).
          3. Otherwise -> a fresh uid is minted (genuinely new person).
        """
        if int(frame_index) != self._frame:
            self._claimed_this_frame.clear()
        self._frame = int(frame_index)
        raw_id = int(raw_id)

        known_uid = self._raw_to_uid.get(raw_id)
        if known_uid is not None:
            self._touch(known_uid, centroid, bbox, keypoints, timestamp, raw_id)
            return known_uid

        uid = self._try_reassociate(centroid, bbox, keypoints)
        if uid is None:
            uid = self._next_uid
            self._next_uid += 1
            self._uids[uid] = _PersonSnap(
                uid=uid,
                centroid=centroid,
                bbox=bbox,
                keypoints=keypoints,
                last_timestamp=float(timestamp),
                last_frame=int(self._frame),
                first_timestamp=float(timestamp),
                first_frame=int(self._frame),
                raw_ids=[raw_id],
            )
        else:
            # Track-id switch recovered: bind the new raw id to the existing uid.
            self._claimed_this_frame.add(uid)
            self._touch(uid, centroid, bbox, keypoints, timestamp, raw_id)
        self._raw_to_uid[raw_id] = uid
        return uid

    def last_centroid(self, uid: int) -> Optional[Tuple[float, float]]:
        snap = self._uids.get(uid)
        return snap.centroid if snap else None

