"""
Adaptive self-tuning layer for the temporal littering event detector.

WHY THIS EXISTS
---------------
Measured on the 19 real videos in D:\\22 (phase1 production runs), the base
event-detector config confirms events on only 6 of 19 videos. The dominant
rejection reasons are NOT_ENOUGH_CARRIED_FRAMES (31), PERSON_NOT_DETECTED (17),
NO_RELEASE_TRANSITION (14) and PERSON_DID_NOT_DEPART (9) — all
threshold/temporal-brittleness failures, not detection failures.

This module makes the system adapt WITHOUT any manual per-video tuning:

1. TIERED CONCURRENT DETECTION (per-video auto-tuning, zero extra latency)
   The same analysis-tick stream is fed into up to 3 LitteringEventDetector
   instances running in parallel: tier 0 = the production config, tier 1 =
   a moderately relaxed config, tier 2 = the maximally (but still safely)
   relaxed config. Tier relaxations are hard-clamped so they can never
   loosen the *evidence* requirements (carry -> release -> ground/abandon),
   only their numeric sensitivity. The strictest tier that confirms an
   event wins; events confirmed by tier 0 suppress any duplicate from
   higher tiers. Pure-Python cost: ~3 state machines on the same ticks,
   microseconds per analysis tick (<0.1% of the ~125 ms YOLO budget).

2. ONLINE LEARNING ACROSS VIDEOS (learning/learning.json)
   After each analyzed video, the tier-0 (base) rejection reasons are
   recorded into a persistent learning store. Each recorded occurrence of
   a rejection reason advances that reason's relaxation by ONE bounded
   step (e.g. NOT_ENOUGH_CARRIED_FRAMES lowers min_carried_frames 6 -> 4
   -> 3). Steps saturate at safe floors, so repeated learning can never
   push thresholds into nonsense. The learned overrides are applied to
   the tier-0 config of the NEXT video automatically.

3. NO MANUAL EDITS REQUIRED
   Neither config/events.yaml nor littering_event_detector.py thresholds
   are edited. The base config is loaded from the YAML as usual and every
   adjustment lives in this layer (and is fully auditable in
   learning/learning.json).

USAGE
-----
    from adaptive_tuner import AdaptiveEventDetector, load_event_config

    detector = AdaptiveEventDetector(load_event_config())
    detector.learning_video = "IMG_5290.MOV"   # enables learning on finalize()
    for tick in ...:
        events = detector.update(persons, bags, timestamp, frame_index)
    ...
    detector.finalize()   # flushes + records learning

It is a drop-in wrapper for LitteringEventDetector: InferencePipeline,
run_pipeline.py and the backend analysis job all keep calling the same
attributes (.update / .finalize / .reset / .summary / ._pairs /
.rejected_events / .confirmed_events).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from littering_event_detector import (
    EventDetectorConfig,
    LitteringEventDetector,
)

# --------------------------------------------------------------------------- #
# Tier specifications (bounded relaxations of the production config)          #
# --------------------------------------------------------------------------- #
# Every value is a HARD CLAMP: a tier may never go past these numbers, no
# matter what the online learning store asks for. Direction rules (enforced
# by test_adaptive_tuner.py):
#   * "min_*" keys and distance/motion ratios may only DECREASE
#   * "max_*" keys and window/grace keys may only INCREASE
TIER_SPECS: List[Dict[str, Any]] = [
    # tier 0 = production config, no overrides.
    {},
    # tier 1 — moderate relaxation.
    {
        "min_carried_frames": 4,
        "release_distance_ratio": 0.22,
        "release_distance_floor": 0.10,
        "release_window_frames": 6,
        "feet_release_frames": 2,
        "departure_motion_ratio": 0.45,
        "departure_distance_ratio": 0.90,
        "min_stationary_frames": 6,
        "stationary_max_pixel_step": 18.0,
        "stationary_grace_frames": 4,
        "min_abandonment_frames": 5,
        "max_pair_age_frames": 60,
        "max_fallback_tracker_gap_frames": 30,
        "min_event_confidence": 0.70,
    },
    # tier 2 — floor relaxation (still requires carry->release->ground).
    {
        "min_carried_frames": 3,
        "smoothing_window": 3,
        "release_distance_ratio": 0.12,
        "release_distance_floor": 0.06,
        "release_window_frames": 8,
        "feet_release_frames": 2,
        "departure_motion_ratio": 0.30,
        "departure_distance_ratio": 0.60,
        "min_departed_frames": 1,
        "min_stationary_frames": 4,
        "stationary_max_pixel_step": 30.0,
        "stationary_grace_frames": 5,
        "min_abandonment_frames": 3,
        "max_pair_age_frames": 90,
        "max_fallback_tracker_gap_frames": 45,
        "min_event_confidence": 0.62,
        "confirmation_grace_frames": 12,
    },
]


# --------------------------------------------------------------------------- #
# Online-learning steps: reason -> ordered bounded relaxations                #
# --------------------------------------------------------------------------- #
# Each analyzed video whose tier-0 summary contains >=1 occurrence of a
# reason advances that reason's step index by 1 (saturating at the last
# step). Steps are cumulative overrides applied to the tier-0 config of
# subsequent videos.
LEARNING_STEPS: Dict[str, List[Dict[str, Any]]] = {
    "NOT_ENOUGH_CARRIED_FRAMES": [
        {"min_carried_frames": 4},
        {"min_carried_frames": 3},
        {"smoothing_window": 3},
    ],
    "NO_RELEASE_TRANSITION": [
        {"release_distance_ratio": 0.22, "release_distance_floor": 0.10},
        {"release_window_frames": 6, "feet_release_frames": 2},
        {"release_distance_ratio": 0.12, "release_window_frames": 8},
    ],
    "PERSON_DID_NOT_DEPART": [
        {"departure_motion_ratio": 0.45},
        {"departure_distance_ratio": 0.85},
        {"min_abandonment_frames": 5},
    ],
    "BAG_NOT_STATIONARY": [
        {"stationary_max_pixel_step": 18.0},
        {"min_stationary_frames": 6},
        {"min_stationary_frames": 4, "stationary_max_pixel_step": 30.0},
    ],
    "PERSON_NOT_DETECTED": [
        {"max_pair_age_frames": 60},
        {"max_pair_age_frames": 90},
    ],
    "BAG_NOT_DETECTED": [
        {"max_fallback_tracker_gap_frames": 30},
        {"max_fallback_tracker_gap_frames": 45},
    ],
}

LEARNING_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "learning", "learning.json"
)
# P0-A: pinned snapshot used for inference reads. Online learning still writes
# to learning.json; promote_learning_pin() copies live → pin explicitly.
INFERENCE_PIN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "learning", "inference_pin.json"
)


def _pin_snapshot_hash(pin_path: str = INFERENCE_PIN_PATH) -> Optional[str]:
    """SHA-256 of the pinned inference snapshot (None when no pin)."""
    import hashlib

    try:
        with open(pin_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


def _learning_source_mode(*, deterministic: bool) -> str:
    """Return ``pin`` | ``none`` | ``live``.

    * deterministic (upload / Motared default): pin or YAML-only — NEVER
      live learning.json. An explicit ``MOTARED_LEARNING_SOURCE=live`` is
      IGNORED here so a stray env cannot reopen the §26.2 read leak
      (editing live step_index must not change a pinned run).
    * explicit ``none`` | ``off`` | ``yaml`` always mean YAML-only.
    * non-deterministic online path: live store (legacy adaptation) unless
      overridden to pin/none.
    """
    raw = (os.environ.get("MOTARED_LEARNING_SOURCE") or "").strip().lower()
    if deterministic:
        if raw in {"none", "off", "yaml"}:
            return "none"
        # "live" is deliberately NOT honored under determinism (P0-A).
        return "pin" if os.path.exists(INFERENCE_PIN_PATH) else "none"
    if raw in {"pin", "none", "live", "off", "yaml"}:
        if raw in {"off", "yaml"}:
            return "none"
        return raw
    # Non-deterministic online path: live store (legacy adaptation).
    return "live"


def resolve_inference_overrides(
    store: Optional["LearningStore"],
    *,
    deterministic: bool = False,
) -> Optional[Dict[str, Any]]:
    """Overrides applied to tier-0 for this detector construction (P0-A)."""
    mode = _learning_source_mode(deterministic=deterministic)
    if mode == "none":
        return None
    if mode == "pin":
        if not os.path.exists(INFERENCE_PIN_PATH):
            return None
        return LearningStore(INFERENCE_PIN_PATH).learned_overrides()
    # live
    if store is None:
        return None
    return store.learned_overrides()


def promote_learning_pin(
    live_path: str = LEARNING_PATH,
    pin_path: str = INFERENCE_PIN_PATH,
) -> Dict[str, Any]:
    """Copy live learning.json step_index into the inference pin (offline gate)."""
    live = LearningStore(live_path)
    pin_doc = {
        "version": 1,
        "pin_schema": 1,
        "description": "Pinned inference overrides. Updated only via promote_learning_pin().",
        "source_videos_analyzed": live.videos_analyzed,
        "step_index": live.step_index(),
        "adjusted_thresholds": live.learned_overrides(),
        "rejection_reasons": {},
        "history": {},
        "promoted_at": time.time(),
    }
    os.makedirs(os.path.dirname(pin_path) or ".", exist_ok=True)
    tmp = pin_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pin_doc, f, indent=2, ensure_ascii=False)
    os.replace(tmp, pin_path)
    return pin_doc


class LearningStore:
    """Persistent online-learning state across analyzed videos (see module
    docstring for the learning/learning.json schema)."""

    def __init__(self, path: str = LEARNING_PATH) -> None:
        self.path = path
        self._doc: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass  # corrupt store -> start fresh (learning is additive)
        return {
            "version": 1,
            "videos_analyzed": 0,
            "rejection_reasons": {},
            "step_index": {},
            "adjusted_thresholds": {},
            "history": {},
        }

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._doc, f, indent=2, ensure_ascii=False)
        os.replace(tmp, self.path)

    @property
    def videos_analyzed(self) -> int:
        return int(self._doc.get("videos_analyzed", 0))

    def rejection_reasons(self) -> Dict[str, int]:
        return dict(self._doc.get("rejection_reasons", {}))

    def step_index(self) -> Dict[str, int]:
        return dict(self._doc.get("step_index", {}))

    def learned_overrides(self) -> Dict[str, Any]:
        """Merged threshold overrides implied by the current step indices."""
        overrides: Dict[str, Any] = {}
        for reason, idx in (self._doc.get("step_index") or {}).items():
            steps = LEARNING_STEPS.get(reason)
            if not steps:
                continue
            for step in steps[: max(0, int(idx))]:
                overrides.update(step)
        return overrides

    def record(
        self,
        video_name: str,
        summary: Dict[str, Any],
        tier_confirmed: Optional[Dict[str, int]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record one analyzed video's tier-0 outcome and advance learning.

        Returns the new adjusted_thresholds map (also persisted).
        """
        counts = (
            summary.get("rejection_reason_counts")
            if isinstance(summary, dict)
            else None
        ) or {}
        reasons = self._doc.setdefault("rejection_reasons", {})
        steps = self._doc.setdefault("step_index", {})
        for reason, count in counts.items():
            try:
                reasons[reason] = int(reasons.get(reason, 0)) + int(count)
            except (TypeError, ValueError):
                continue
            if reason in LEARNING_STEPS and int(count) > 0:
                cap = len(LEARNING_STEPS[reason])
                steps[reason] = min(cap, int(steps.get(reason, 0)) + 1)

        self._doc["videos_analyzed"] = int(self._doc.get("videos_analyzed", 0)) + 1
        adjusted = self.learned_overrides()
        self._doc["adjusted_thresholds"] = adjusted

        history = self._doc.setdefault("history", {})
        entry: Dict[str, Any] = {
            "summary": {
                "total_candidates": summary.get("total_candidates"),
                "confirmed_violations": summary.get("confirmed_violations"),
                "rejected_candidates": summary.get("rejected_candidates"),
                "rejection_reason_counts": counts,
            },
            "recorded_at": time.time(),
        }
        if tier_confirmed:
            entry["tiers_confirmed"] = tier_confirmed
        if extra:
            entry.update(extra)
        history[video_name] = entry

        self.save()
        return adjusted

    def reset(self) -> None:
        self._doc = {
            "version": 1,
            "videos_analyzed": 0,
            "rejection_reasons": {},
            "step_index": {},
            "adjusted_thresholds": {},
            "history": {},
        }
        self.save()


# --------------------------------------------------------------------------- #
# Tier config construction                                                     #
# --------------------------------------------------------------------------- #
def build_tier_configs(
    base: EventDetectorConfig,
    learned: Optional[Dict[str, Any]] = None,
    max_tiers: Optional[int] = None,
) -> List[EventDetectorConfig]:
    """Build the per-tier configs: tier0 = base + learned overrides,
    tierN = tier0 + TIER_SPECS[N]. Duplicate configs (a learned override may
    already equal a tier's values) are removed while preserving order.
    """
    tier0 = replace(base, **(learned or {})) if learned else base
    specs = TIER_SPECS if max_tiers is None else TIER_SPECS[: max_tiers + 1]
    configs: List[EventDetectorConfig] = [tier0]
    seen = {json.dumps(tier0.to_dict(), sort_keys=True, default=str)}
    for spec in specs[1:]:
        cfg = replace(tier0, **spec)
        key = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        configs.append(cfg)
    return configs


# --------------------------------------------------------------------------- #
# Adaptive wrapper (drop-in for LitteringEventDetector)                        #
# --------------------------------------------------------------------------- #
class AdaptiveEventDetector:
    """Runs N tier detectors concurrently over the same tick stream.

    The strictest tier that confirms an event wins; tier-0 confirmations
    suppress duplicates from relaxed tiers (same person within a short time
    window). Tier-0 rejections drive the online learning store, so the base
    config adapts across videos while the tier ladder handles per-video
    variance immediately.
    """

    DEDUP_WINDOW_SEC = 6.0
    # Relaxed-tier confirmations are held back for this many seconds before
    # being emitted. The relaxed tiers confirm EARLIER than the strict tier
    # (lower thresholds), so emitting immediately would beat the strict tier
    # to the punch. Within the grace window a tier-0 confirmation of the same
    # physical event suppresses the buffered relaxed event — the strictest
    # tier that confirms wins.
    RELAXED_EMIT_GRACE_SEC = 3.0

    def __init__(
        self,
        base_config: EventDetectorConfig,
        store: Optional[LearningStore] = None,
        max_tiers: Optional[int] = None,
        enable_learning: bool = True,
        camera_id: Optional[str] = None,
        *,
        deterministic: bool = False,
        freeze_learning_writes: bool = False,
    ) -> None:
        self.deterministic = bool(deterministic)
        # P0-A: freeze writes AND isolate reads from the live learning.json
        # mutation path. Deterministic inference uses inference_pin.json (or
        # YAML-only if no pin) — never the live store that other jobs mutate.
        self.freeze_learning_writes = bool(
            freeze_learning_writes or self.deterministic
        )
        self.store = store if store is not None else (
            LearningStore() if enable_learning else None
        )
        learned = resolve_inference_overrides(
            self.store, deterministic=self.deterministic
        )
        self.learning_source = _learning_source_mode(
            deterministic=self.deterministic
        )
        self.tier_configs = build_tier_configs(base_config, learned, max_tiers)
        self._detectors: List[LitteringEventDetector] = [
            LitteringEventDetector(cfg, camera_id=camera_id)
            for cfg in self.tier_configs
        ]
        # Assigned by the caller (run_pipeline.py / backend analysis job) so
        # finalize() can record the video outcome into the learning store.
        self.learning_video: Optional[str] = None
        self.learning_recorded = False
        # (event_id, person_track_id, confirmed_ts) of every confirmation we
        # have emitted so far — used to suppress duplicate confirmations from
        # the more relaxed tiers (the strictest tier that confirms wins). The
        # event_id is kept so an event is never treated as a duplicate of
        # ITSELF on a later re-check (e.g. from _dedup_confirmed() re-reading
        # a relaxed tier's own confirmed_events list after _accept()).
        self._emitted_confirmed: List[tuple] = []
        # (tier, event, emit_at) — relaxed-tier confirmations waiting out
        # their grace window (see RELAXED_EMIT_GRACE_SEC).
        self._pending_relaxed: List[tuple] = []

    # ------------------------------------------------------------------ #
    # Introspection expected by the pipeline / dashboard                  #
    # ------------------------------------------------------------------ #
    @property
    def primary(self) -> LitteringEventDetector:
        return self._detectors[0]

    @property
    def config(self) -> EventDetectorConfig:
        return self.tier_configs[0]

    @property
    def _pairs(self):
        """Pair memory of the PRIMARY (strictest) tier.

        Dashboard/status code iterates this to show the live AI state; the
        conservative tier-0 view is the honest one to display.
        """
        return self.primary._pairs

    @property
    def confirmed_events(self) -> List[Any]:
        return self._dedup_confirmed()

    @property
    def rejected_events(self) -> List[Any]:
        return self.primary.rejected_events

    @property
    def _person_identity(self) -> Any:
        return self.primary._person_identity

    @property
    def _object_identity(self) -> Any:
        return self.primary._object_identity

    @property
    def last_person_uid_map(self) -> Dict[int, int]:
        return self.primary.last_person_uid_map

    @property
    def last_object_uid_map(self) -> Dict[int, Optional[int]]:
        return self.primary.last_object_uid_map

    # ------------------------------------------------------------------ #
    def update(self, persons, bags, timestamp, frame_index=None, frame_size=None) -> List[Any]:
        emitted: List[Any] = []
        for tier, det in enumerate(self._detectors):
            tier_events = det.update(persons, bags, timestamp, frame_index, frame_size)
            for ev in tier_events:
                if not ev.confirmed:
                    # Only the primary tier's rejections surface and drive
                    # learning; relaxed-tier rejections are noise.
                    if tier == 0:
                        emitted.append(ev)
                    continue
                if self._is_duplicate(ev, self._event_ts(ev)):
                    continue
                if tier == 0:
                    self._accept(ev)
                    emitted.append(ev)
                else:
                    # Hold relaxed confirmations for the grace window so a
                    # tier-0 confirmation of the same physical event can
                    # still win (it confirms later but is stricter).
                    ev.evidence = dict(ev.evidence or {})
                    ev.evidence["adaptive_tier"] = tier
                    ts = self._event_ts(ev)
                    emit_at = (float(ts) if ts is not None else float(timestamp)) \
                        + self.RELAXED_EMIT_GRACE_SEC
                    self._pending_relaxed.append((tier, ev, emit_at))
        emitted.extend(self._flush_pending(timestamp, final=False))
        return emitted

    def finalize(self) -> List[Any]:
        emitted: List[Any] = []
        for tier, det in enumerate(self._detectors):
            for ev in det.finalize():
                if not ev.confirmed:
                    if tier == 0:
                        emitted.append(ev)
                    continue
                if self._is_duplicate(ev, self._event_ts(ev)):
                    continue
                if tier == 0:
                    self._accept(ev)
                    emitted.append(ev)
                else:
                    ev.evidence = dict(ev.evidence or {})
                    ev.evidence["adaptive_tier"] = tier
                    self._pending_relaxed.append((tier, ev, None))
        # End of stream: no further tier-0 confirmation can arrive, so every
        # still-unique buffered relaxed confirmation is emitted now.
        emitted.extend(self._flush_pending(None, final=True))
        if (
            self.learning_video
            and self.store is not None
            and not self.learning_recorded
            and not self.freeze_learning_writes
        ):
            self.record_learning()
            self.learning_recorded = True
        return emitted

    def _flush_pending(self, now: Optional[float], final: bool) -> List[Any]:
        """Emit buffered relaxed confirmations whose grace window elapsed
        (or all of them when ``final``), dropping any that a stricter tier
        confirmed in the meantime.

        Also drops relaxed confirmations that lack physical separation
        evidence (clothing / hip color latch) — tier 1/2 must not promote
        a body-attached blob that tier-0 correctly refused.
        """
        out: List[Any] = []
        remaining: List[tuple] = []
        for tier, ev, emit_at in sorted(self._pending_relaxed, key=lambda x: x[0]):
            ts = self._event_ts(ev)
            if not final and not self._is_duplicate(ev, ts) and now is not None \
                    and emit_at is not None and float(now) < float(emit_at):
                remaining.append((tier, ev, emit_at))
                continue
            if self._is_duplicate(ev, ts):
                continue  # a stricter tier confirmed the same physical event
            if not self._event_has_physical_separation(ev):
                continue  # clothing latch / no real discard
            if self._primary_rejected_weak_carry(ev, ts):
                continue
            self._accept(ev)
            out.append(ev)
        self._pending_relaxed = remaining
        return out

    @staticmethod
    def _event_has_physical_separation(ev: Any) -> bool:
        """True when the confirmed event shows the object left the actor."""
        details = getattr(ev, "details", None) or {}
        try:
            max_sep = float(details.get("max_post_release_norm_distance") or 0.0)
        except (TypeError, ValueError):
            max_sep = 0.0
        try:
            sep_frames = int(details.get("separated_frames") or 0)
        except (TypeError, ValueError):
            sep_frames = 0
        try:
            bag_disp = float(details.get("max_bag_displacement_px") or 0.0)
        except (TypeError, ValueError):
            bag_disp = 0.0
        # Match littering_event_detector: need sustained separation AND the
        # bag itself must have moved (not a static clothing/clutter latch).
        return max_sep >= 0.08 and sep_frames >= 2 and bag_disp >= 50.0

    def _primary_rejected_weak_carry(
        self, ev: Any, ts: Optional[float]
    ) -> bool:
        """Suppress relaxed confirmations when tier-0 already rejected the
        same actor for missing carry / release / separation in-window."""
        pid = self._actor_key(ev)
        if pid is None:
            return False
        if ts is None:
            return False
        weak = {
            "NOT_ENOUGH_CARRIED_FRAMES",
            "NO_RELEASE_TRANSITION",
            "NO_PHYSICAL_SEPARATION",
            # Tier-0 bin / dumpster-lip refusals must not be overturned by
            # relaxed tiers (IMG_5305 hard negative vs IMG_5290 street drop).
            "BIN_DISPOSAL",
            "BIN_ZONE_DEPOSIT",
            "NO_CONFIDENT_EVENT",
            # RCM-12 (forensic corrective plan): a relaxed tier confirming
            # a physical incident that tier-0 already resolved as an
            # explicit RECLAIM, an unresolved ACTOR ambiguity, or a closer
            # bystander is not "detector brittleness" the ladder should
            # rescue — it is definitive contrary evidence about WHAT
            # happened or WHO did it. These three are always vetoed below,
            # unconditionally (unlike the carry/release-shortfall reasons,
            # which the physical-separation check may still rescue).
            "PICKED_BACK_UP",
            "ASSOCIATION_AMBIGUOUS",
            "OTHER_PERSON_CLOSER",
        }
        always_block = {
            "NO_PHYSICAL_SEPARATION",
            "BIN_DISPOSAL",
            "BIN_ZONE_DEPOSIT",
            "NO_CONFIDENT_EVENT",
            "PICKED_BACK_UP",
            "ASSOCIATION_AMBIGUOUS",
            "OTHER_PERSON_CLOSER",
        }
        for rej in self.primary.rejected_events:
            rej_pid = self._actor_key(rej)
            if rej_pid is None or rej_pid != pid:
                continue
            reason = getattr(rej, "reason", None)
            if reason not in weak:
                continue
            rej_ts = None
            try:
                rej_ts = (
                    (rej.timestamps or {}).get("confirmed")
                    or (rej.timestamps or {}).get("departure")
                    or (rej.timestamps or {}).get("release")
                    or (rej.timestamps or {}).get("carry_start")
                )
            except Exception:
                rej_ts = None
            if rej_ts is None:
                continue
            if abs(float(ts) - float(rej_ts)) <= self.DEDUP_WINDOW_SEC * 2.0:
                if reason in always_block:
                    return True
                # Carry/release shortfalls: still allow a relaxed rescue when
                # the event shows real bag motion + separation (brittle but
                # genuine litter). Block clothing/static latch rescues.
                if not self._event_has_physical_separation(ev):
                    return True
        return False

    def reset(self) -> None:
        for det in self._detectors:
            det.reset()
        self.learning_recorded = False
        self._emitted_confirmed = []
        self._pending_relaxed = []

    def summary(self) -> Dict[str, Any]:
        out = dict(self.primary.summary())
        out["adaptive"] = {
            "tiers": len(self._detectors),
            "tier_summaries": [det.summary() for det in self._detectors],
            "learned_overrides": (
                self.store.learned_overrides() if self.store else {}
            ),
            "videos_analyzed": (
                self.store.videos_analyzed if self.store else 0
            ),
            # P0-A: which snapshot produced THIS run's tier-0 thresholds.
            "learning_source": self.learning_source,
            "inference_pin_hash": (
                _pin_snapshot_hash() if self.learning_source == "pin" else None
            ),
        }
        return out

    def record_learning(self, video_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Persist the tier-0 outcome into the learning store. Returns the
        new adjusted thresholds."""
        name = video_name or self.learning_video
        if not name or self.store is None:
            return None
        tier_confirmed = {
            f"tier{i}": det.summary().get("confirmed_violations", 0)
            for i, det in enumerate(self._detectors)
        }
        return self.store.record(
            name, self.primary.summary(), tier_confirmed=tier_confirmed
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _actor_key(ev) -> Optional[int]:
        """Stable actor identity for dedup/arbitration purposes.

        RCM-09/RC-7 (forensic corrective plan): this MUST prefer the
        stable ``event_actor_person_uid`` (frozen at carry time, survives
        a raw tracker-id switch) over the raw ``person_track_id``. Keying
        deduplication on the raw id let a person-track-id churn mid-arc
        read as a "different person", producing more than one persisted
        event for a single physical incident. Falls back to the raw id
        only when no stable uid was populated (legacy/unit-test events).
        """
        uid = getattr(ev, "event_actor_person_uid", None)
        if uid is not None:
            try:
                return int(uid)
            except (TypeError, ValueError):
                pass
        try:
            return int(ev.person_track_id)
        except Exception:
            return None

    def _is_duplicate(self, ev, ts: Optional[float]) -> bool:
        """True when this person already has a confirmation within the
        dedup window — either from the strict tier or from an earlier,
        stricter tier emit. Events with incomparable timestamps are assumed
        to be DIFFERENT events (never suppressed).

        An event's own previously-recorded acceptance (same ``event_id``) is
        always excluded from this check, so an event is never treated as a
        duplicate of ITSELF — e.g. when ``_dedup_confirmed()`` re-reads a
        relaxed tier's ``confirmed_events`` list after that same event was
        already `_accept()`-ed (see module bug history: this previously
        caused every non-tier-0-confirmed event to be discarded on its very
        next read)."""
        pid = self._actor_key(ev)
        if pid is None:
            return False
        if ts is None:
            return False
        self_event_id = getattr(ev, "event_id", None)
        for done_id, done_pid, done_ts in self._emitted_confirmed:
            if self_event_id is not None and done_id == self_event_id:
                continue
            if done_pid != pid or done_ts is None:
                continue
            if abs(float(ts) - float(done_ts)) <= self.DEDUP_WINDOW_SEC:
                return True
        return False

    def _suppressed_by_primary(self, ev, fallback_ts: Optional[float]) -> bool:
        ts = None
        try:
            ts = ev.timestamps.get("confirmed") or ev.timestamps.get("departure")
        except Exception:
            ts = None
        if ts is None:
            ts = fallback_ts
        return self._is_duplicate(ev, ts)

    def _accept(self, ev) -> float:
        ts = None
        try:
            ts = ev.timestamps.get("confirmed") or ev.timestamps.get("departure")
        except Exception:
            ts = None
        pid = self._actor_key(ev)
        if pid is None:
            pid = -1
        self._emitted_confirmed.append((getattr(ev, "event_id", None), pid, ts))
        return ts

    def _dedup_confirmed(self) -> List[Any]:
        """Confirmed events that were actually accepted for emit.

        Higher-tier detectors may locally confirm clothing latches; those must
        NOT appear in the dashboard/report unless ``_flush_pending`` accepted
        them (physical-separation + primary-rejection gates).
        """
        accepted_ids = {
            eid for eid, _, _ in self._emitted_confirmed if eid is not None
        }
        out: List[Any] = list(self.primary.confirmed_events)
        for tier, det in enumerate(self._detectors[1:], start=1):
            for ev in det.confirmed_events:
                eid = getattr(ev, "event_id", None)
                if eid is None or eid not in accepted_ids:
                    continue
                if self._is_duplicate(ev, self._event_ts(ev)):
                    continue
                ev.evidence = dict(ev.evidence or {})
                ev.evidence["adaptive_tier"] = tier
                out.append(ev)
        return out

    @staticmethod
    def _event_ts(ev) -> Optional[float]:
        try:
            return ev.timestamps.get("confirmed") or ev.timestamps.get("departure")
        except Exception:
            return None


def tuned_config(base: Optional[EventDetectorConfig] = None,
                 store: Optional[LearningStore] = None) -> EventDetectorConfig:
    """Base config with the current learned overrides applied (no tiers)."""
    if base is None:
        from littering_event_detector import load_event_config
        base = load_event_config()
    st = store if store is not None else LearningStore()
    learned = st.learned_overrides()
    return replace(base, **learned) if learned else base
