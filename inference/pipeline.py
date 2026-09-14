"""
Inference Pipeline — orchestrates the full per-frame flow.

This is the integration layer that wires:
    capture → buffer → YOLO → ByteTrack → MoveNet → association
    → state machine → voting → evidence manager → (optional) backend POST

It is the only module that knows about all the pieces. Each piece is
independently unit-tested; this module's job is choreography.

Design choices made here (matches the architecture doc):
  * Capture FPS != Analysis FPS. The buffer ingests every captured frame
    at full FPS so evidence clips are smooth, but the heavy pipeline
    (YOLO + pose) runs at a configurable analysis_fps to stay real-time
    on CPU. Frames between analysis ticks are still buffered.
  * Pose is lazy: we only run MoveNet on persons the event engine is
    currently tracking or about to evaluate, not on every detected person.
  * On LITTERING_CONFIRMED, we (a) take the snapshot+pre segment
    immediately, (b) wait ``post_seconds`` of real time, (c) finalize the
    video. The dashboard reflects this ~3s latency by design.
  * Backend reporting is optional and async (HTTP POST on a thread) so a
    slow/unreachable backend never stalls the live pipeline.
"""

from __future__ import annotations

import threading
import time
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from inference.association.person_object_assoc import (
    AssociationConfig,
    Track,
)
from inference.capture.circular_buffer import CircularFrameBuffer
from inference.evidence.evidence_manager import EvidenceArtifact, EvidenceManager, EvidenceRequest
from littering_event_detector import (
    DetectorBag,
    DetectorKeypoints,
    DetectorPerson,
    EventDetectorConfig,
    LitteringEventDetector,
    load_event_config,
)


@dataclass
class PipelineConfig:
    buffer_seconds: float = 6.0
    analysis_fps: float = 10.0
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    camera_id: str = "cam-01"
    assoc_config: AssociationConfig = field(default_factory=AssociationConfig)
    event_detector_config: EventDetectorConfig = field(default_factory=load_event_config)
    post_backend_url: Optional[str] = None  # if set, POST events to FastAPI
    # Adaptive self-tuning: concurrent tier ladder + online learning store
    # (adaptive_tuner.py). P1-7 (audit): the dataclass default is False only
    # so unit tests can exercise the bare detector in isolation — BOTH
    # production entry points (backend/routers/analysis.py and
    # scripts/run_pipeline.py) hardcode auto_tune=True, so in production the
    # adaptive wrapper is ALWAYS ON. Do not "fix" the default without
    # re-auditing every test that relies on the legacy path.
    auto_tune: bool = False
    # Online-learning identity for this source (video filename / camera id).
    # Feeds adaptive_tuner's learning store history and names the
    # active-learning image folder. None -> camera_id is used.
    learning_tag: Optional[str] = None
    # Collect carry/confirmation crops under datasets/active_learning/ for
    # future trash-model training. Cheap (a few imwrites per video, capped).
    active_learning: bool = True
    # Process-wide determinism (seeds, cudnn, no learning.json writes).
    # Upload analysis sets this True so the same video yields the same FSM
    # decision across re-runs. Live cameras may keep learning writes.
    deterministic: bool = False


@dataclass
class PipelineEvent:
    """A confirmed littering event surfaced to the UI/backend."""

    event_id: str
    camera_id: str
    person_track_id: int
    object_track_id: int
    object_type: str
    confidence: float
    event_timestamp: float
    state_history: list
    # Phase C — authoritative event-actor/object identity, frozen at carry
    # time by the event detector. These (not the legacy person_track_id /
    # object_track_id above, which reflect raw/rebound tracker ids) are what
    # must reach the backend so the DB's stable-identity columns are ever
    # populated for events reported via the live-camera HTTP path.
    event_actor_person_track_id: Optional[int] = None
    event_actor_person_uid: Optional[int] = None
    event_object_track_id: Optional[int] = None
    event_object_uid: Optional[int] = None


class InferencePipeline:
    """
    The pipeline is driven by :meth:`process_frame`, which the capture
    loop calls. Internally it throttles to ``analysis_fps``.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.buffer = CircularFrameBuffer(window_seconds=self.config.buffer_seconds)
        self.evidence_mgr = EvidenceManager()
        # Authoritative temporal event decision engine. The older FSM/voting
        # modules remain available for regression tests, but production
        # decisions come from this single detector.
        #
        # Adaptive self-tuning (enabled by the production entry points via
        # PipelineConfig.auto_tune): wraps the detector in a
        # concurrent tier ladder (tier 0 = this config, tier 1/2 = bounded
        # relaxations) plus the persistent online-learning store. It is a
        # drop-in wrapper exposing the same interface, so nothing downstream
        # changes. auto_tune=False keeps the exact legacy single-detector
        # behaviour (used by unit tests that assert on it).
        if getattr(self.config, "auto_tune", False):
            from adaptive_tuner import AdaptiveEventDetector
            self.event_detector = AdaptiveEventDetector(
                self.config.event_detector_config,
                camera_id=self.config.camera_id,
                deterministic=bool(getattr(self.config, "deterministic", False)),
            )
            if self.config.learning_tag:
                # attribute exists only on the adaptive wrapper; setting it on
                # the plain detector would shadow nothing harmful, but guard
                # anyway to keep the legacy path pristine.
                try:
                    self.event_detector.learning_video = self.config.learning_tag
                except Exception:
                    pass
        else:
            self.event_detector = LitteringEventDetector(self.config.event_detector_config)
        # pending evidence awaiting post-window finalize, keyed by event_id
        self._pending: Dict[str, EvidenceArtifact] = {}
        # evidence artifacts that were finalized AND verified (non-empty files),
        # keyed by event_id — lets in-process callers (video-upload analysis)
        # persist them without an HTTP self-call.
        self.finalized_artifacts: Dict[str, EvidenceArtifact] = {}
        self._last_analysis_ts: float = 0.0
        self._frame_count = 0
        self.events: List[PipelineEvent] = []
        self.rejected_events = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    def _detector_person(self, track: Track) -> DetectorPerson:
        kp = getattr(track, "keypoints", None)
        detector_kp = None
        if kp is not None:
            detector_kp = DetectorKeypoints(
                left_wrist=getattr(kp, "left_wrist", None),
                right_wrist=getattr(kp, "right_wrist", None),
                torso_center=getattr(kp, "torso_center", None),
                left_shoulder=getattr(kp, "left_shoulder", None),
                right_shoulder=getattr(kp, "right_shoulder", None),
            )
        return DetectorPerson(
            track_id=int(track.track_id),
            bbox=tuple(float(v) for v in track.bbox),
            confidence=float(getattr(track, "confidence", 1.0) or 1.0),
            keypoints=detector_kp,
        )

    def _detector_bag(self, track: Track) -> DetectorBag:
        source = str(getattr(track, "source", "yolo") or "yolo")
        return DetectorBag(
            track_id=int(track.track_id),
            bbox=tuple(float(v) for v in track.bbox),
            confidence=float(getattr(track, "confidence", 0.0) or 0.0),
            class_name=str(track.class_name),
            source=source,
            yolo_confirmed=(source == "yolo"),
            object_uid=getattr(track, "object_uid", None),
        )

    def should_analyze(self, timestamp: float) -> bool:
        """Return True when this timestamp should run the throttled AI path."""
        return self._last_analysis_ts <= 0 or (timestamp - self._last_analysis_ts) >= (1.0 / self.config.analysis_fps)

    def process_frame(
        self,
        frame,
        timestamp: float,
        persons: List[Track],
        objects: List[Track],
    ) -> List[PipelineEvent]:
        """
        Called by the capture loop. ``persons``/``objects`` are already
        tracker-output ``Track`` objects (the pipeline caller is
        responsible for running YOLO + ByteTrack + MoveNet and assembling
        these). This method handles buffering + the authoritative temporal
        event detector + evidence.

        Returns any new confirmed events emitted this frame.
        """
        # 1) always buffer the raw frame (full capture FPS)
        self.buffer.push(frame, timestamp=timestamp, frame_index=self._frame_count)
        self._frame_count += 1

        # 2) throttle the heavy logic to analysis_fps
        if self._last_analysis_ts > 0 and (timestamp - self._last_analysis_ts) < (1.0 / self.config.analysis_fps):
            return []

        self._last_analysis_ts = timestamp
        new_events: List[PipelineEvent] = []

        # 3) authoritative temporal littering detector.
        det_persons = [self._detector_person(p) for p in persons]
        det_bags = [self._detector_bag(o) for o in objects]
        state_before = {
            key: mem.state for key, mem in self.event_detector._pairs.items()
        }
        detector_events = self.event_detector.update(
            det_persons, det_bags, timestamp, self._frame_count,
            frame_size=(frame.shape[1], frame.shape[0]),
        )
        self.rejected_events = self.event_detector.rejected_events

        # 3b) Anchor a STABLE logical object identity onto the object Tracks so
        # the FrameAnalysis record and downstream evidence anchor to the SAME
        # physical bag across tracker/colour id churn (spec rules #11-#17). The
        # uid comes from the FSM pair memory, which already persists the bag
        # through churn via its rebind logic.
        for mem in self.event_detector._pairs.values():
            if mem.bag_uid is None:
                continue
            for o in objects:
                if int(o.track_id) == int(mem.bag_id):
                    o.object_uid = mem.bag_uid
                    break

        # 4) on confirmation, start evidence assembly.
        for dev in detector_events:
            if not dev.confirmed:
                continue
            ev = self._start_evidence_for_detector_event(dev, timestamp)
            if ev is not None:
                new_events.append(ev)

        # 4b) active-learning crops: carry establishment + confirmations only
        # (rare), so the cost is a couple of imwrites per video at most.
        if self.config.active_learning:
            try:
                self._collect_active_learning(frame, persons, objects, state_before, detector_events)
            except Exception:
                pass

        # 5) finalize pending evidence whose post-window has elapsed,
        #    THEN upload to backend (correct order: finalize -> verify -> upload)
        self._finalize_pending_evidence(timestamp)

        return new_events

    def _collect_active_learning(self, frame, persons, objects, state_before, detector_events) -> None:
        """Dump person/waste crops when the FSM establishes a carry or confirms
        an event (adaptive.py). These crops are the training set for a future
        dedicated trash-bag model — collected with zero manual effort."""
        from adaptive import dump_pair_crops
        from littering_event_detector import EventState

        tag = self.config.learning_tag or self.config.camera_id
        track_by_id = {}
        for t in list(persons) + list(objects):
            track_by_id.setdefault(int(t.track_id), t)
        for key, mem in self.event_detector._pairs.items():
            before = state_before.get(key)
            established_carry = (
                mem.state == EventState.BAG_CARRIED
                and before is not None
                and before != EventState.BAG_CARRIED
                and mem.carry_start_frame is not None
                and mem.carry_start_frame >= self._frame_count - 2
            )
            if not established_carry:
                continue
            person = track_by_id.get(int(mem.person_id))
            bag = track_by_id.get(int(mem.bag_id))
            dump_pair_crops(
                tag, frame,
                person.bbox if person is not None else None,
                bag.bbox if bag is not None else None,
                "carry", self._frame_count, mem.person_id, mem.bag_id,
            )
        for dev in detector_events:
            if not dev.confirmed:
                continue
            person = track_by_id.get(int(dev.event_actor_person_track_id or dev.person_track_id))
            bag = track_by_id.get(int(dev.event_object_track_id or dev.bag_track_id))
            dump_pair_crops(
                tag, frame,
                person.bbox if person is not None else None,
                bag.bbox if bag is not None else None,
                "confirmed", self._frame_count,
                int(dev.event_actor_person_track_id or dev.person_track_id),
                int(dev.event_object_track_id or dev.bag_track_id),
            )

    def _start_evidence_for_detector_event(self, dev, timestamp: float) -> Optional[PipelineEvent]:
        if dev.event_id in self._pending:
            return None
        event_ts = float(dev.timestamps.get("confirmed") or dev.timestamps.get("departure") or timestamp)
        req = EvidenceRequest(
            camera_id=self.config.camera_id,
            person_track_id=int(dev.person_track_id),
            object_track_id=int(dev.bag_track_id),
            object_type=str(dev.bag_class),
            confidence=float(dev.confidence),
            event_timestamp=event_ts,
            pre_seconds=self.config.pre_seconds,
            post_seconds=self.config.post_seconds,
            extra={
                "detector_event_id": dev.event_id,
                "reason": dev.reason,
                "evidence": dev.evidence,
                "frames": dev.frames,
                "timestamps": dev.timestamps,
            },
        )
        art = self.evidence_mgr.assemble_snapshot(dev.event_id, self.buffer, req)
        if art is None:
            import logging
            logging.getLogger("ai_littering").error(
                "Confirmed detector event %s has no buffer frames for evidence snapshot",
                dev.event_id,
            )
            return None
        self._pending[dev.event_id] = art
        ev = PipelineEvent(
            event_id=dev.event_id,
            camera_id=req.camera_id,
            person_track_id=req.person_track_id,
            object_track_id=req.object_track_id,
            object_type=req.object_type,
            confidence=req.confidence,
            event_timestamp=req.event_timestamp,
            state_history=[
                {"state": dev.state.value, "timestamp": event_ts, "reason": dev.reason},
            ],
            event_actor_person_track_id=getattr(dev, "event_actor_person_track_id", None),
            event_actor_person_uid=getattr(dev, "event_actor_person_uid", None),
            event_object_track_id=getattr(dev, "event_object_track_id", None),
            event_object_uid=getattr(dev, "event_object_uid", None),
        )
        self.events.append(ev)
        return ev

    def _finalize_pending_evidence(self, timestamp: float, force: bool = False) -> None:
        for event_id in list(self._pending.keys()):
            art = self._pending[event_id]
            if not force and timestamp - art.request.event_timestamp < self.config.post_seconds:
                continue
            # 5a) finalize the evidence (writes the MP4)
            try:
                self.evidence_mgr.finalize(art, self.buffer)
            except Exception as e:
                import logging
                logging.getLogger("ai_littering").error(
                    "Evidence finalize failed for event %s: %s", art.event_id, e
                )
                del self._pending[event_id]
                continue
            # 5b) verify the files exist and are valid (non-empty)
            snap_ok = art.snapshot_path and os.path.exists(art.snapshot_path) and os.path.getsize(art.snapshot_path) > 0
            vid_ok = art.video_path and os.path.exists(art.video_path) and os.path.getsize(art.video_path) > 0
            if not snap_ok:
                import logging
                logging.getLogger("ai_littering").error(
                    "Evidence snapshot missing or empty for event %s", art.event_id
                )
            if not vid_ok:
                import logging
                logging.getLogger("ai_littering").warning(
                    "Evidence video missing or empty for event %s", art.event_id
                )
            # record the verified artifact so embedders of the pipeline
            # (e.g. the video-upload analysis job) can persist it without
            # an HTTP self-call.
            if snap_ok:
                self.finalized_artifacts[art.event_id] = art
            # 5c) find the PipelineEvent and upload to backend
            for ev in self.events:
                if ev.event_id == art.event_id:
                    self._maybe_post_backend(ev, art, snap_ok, vid_ok)
                    break
            del self._pending[event_id]

    def finalize(self, timestamp: Optional[float] = None) -> List[PipelineEvent]:
        """Flush detector state at end-of-stream and finalize pending evidence.

        Returns any confirmed events that were only emitted by the final
        detector flush.
        """
        ts = float(timestamp) if timestamp is not None else float(self._last_analysis_ts)
        new_events: List[PipelineEvent] = []
        for dev in self.event_detector.finalize():
            if not dev.confirmed:
                continue
            ev = self._start_evidence_for_detector_event(dev, ts)
            if ev is not None:
                new_events.append(ev)
        self.rejected_events = self.event_detector.rejected_events
        self._finalize_pending_evidence(ts, force=True)
        return new_events

    def _maybe_post_backend(self, event: PipelineEvent, artifact: EvidenceArtifact,
                             snap_ok: bool = True, vid_ok: bool = True) -> None:
        """Upload event + evidence to the backend AFTER finalize.

        Correct order: CONFIRMED -> create pending -> wait post_seconds -> finalize MP4
        -> verify files exist -> POST event -> UPLOAD evidence -> dashboard

        Failures are LOGGED, not silently swallowed.
        """
        url = self.config.post_backend_url
        if not url:
            return
        def _post():
            try:
                import requests  # type: ignore
                import logging
                log = logging.getLogger("ai_littering")
                payload = {
                    "camera_id": event.camera_id,
                    "person_track_id": str(event.person_track_id),
                    "object_track_id": str(event.object_track_id),
                    "object_type": event.object_type,
                    "confidence": event.confidence,
                    "timestamp": event.event_timestamp,
                    "status": "confirmed",
                    # Phase C — authoritative actor/object identity, frozen at
                    # carry time. Without these the DB's stable-identity
                    # columns stay NULL for every event reported via this
                    # live-camera HTTP path (see MASTER_REPAIR_PLAN.md P0-1).
                    "event_actor_person_track_id": event.event_actor_person_track_id,
                    "event_actor_person_uid": event.event_actor_person_uid,
                    "event_object_track_id": event.event_object_track_id,
                    "event_object_uid": event.event_object_uid,
                }
                base = url.rsplit("/events", 1)[0]
                # P0-3 (MASTER_REPAIR_PLAN): POST /api/events now requires the
                # shared internal ingest token. Keep the default in sync with
                # backend/routers/events.py; override with EVENT_INGEST_TOKEN.
                ingest_token = os.environ.get(
                    "EVENT_INGEST_TOKEN", "littering-internal-ingest-v1"
                )
                resp = requests.post(
                    f"{base}/events",
                    json=payload,
                    headers={"X-Internal-Token": ingest_token},
                    timeout=3,
                )
                if resp.status_code != 201:
                    log.error("Backend event creation failed: HTTP %s", resp.status_code)
                    return
                event_db_id = resp.json().get("id")
                if event_db_id is None:
                    log.error("Backend event creation returned no id")
                    return
                snap_path = getattr(artifact, "snapshot_path", None)
                vid_path = getattr(artifact, "video_path", None)
                if snap_ok and snap_path and os.path.exists(snap_path):
                    if vid_ok and vid_path and os.path.exists(vid_path):
                        with open(snap_path, "rb") as sf, open(vid_path, "rb") as vf:
                            files = {
                                "snapshot": ("snapshot.jpg", sf, "image/jpeg"),
                                "video": ("evidence.mp4", vf, "video/mp4"),
                            }
                            data = {"duration_sec": str(artifact.duration_seconds)}
                            r = requests.post(
                                f"{base}/evidence/{event_db_id}/upload",
                                files=files, data=data, timeout=10,
                            )
                            if r.status_code != 201:
                                log.error("Evidence upload failed: HTTP %s", r.status_code)
                            else:
                                log.info("Evidence uploaded for event %s (snapshot+video)", event_db_id)
                    else:
                        with open(snap_path, "rb") as sf:
                            files = {"snapshot": ("snapshot.jpg", sf, "image/jpeg")}
                            data = {"duration_sec": str(artifact.duration_seconds)}
                            r = requests.post(
                                f"{base}/evidence/{event_db_id}/upload",
                                files=files, data=data, timeout=10,
                            )
                            if r.status_code != 201:
                                log.error("Evidence upload (snapshot only) failed: HTTP %s", r.status_code)
                            else:
                                log.warning("Evidence uploaded for event %s (snapshot only, video missing)", event_db_id)
                else:
                    log.error("No snapshot to upload for event %s", event_db_id)
            except Exception as e:
                import logging
                logging.getLogger("ai_littering").error(
                    "Backend upload failed for event %s: %s", event.event_id, e
                )
        threading.Thread(target=_post, daemon=True).start()

    def push_status(self, timestamp: float, capture_fps: Optional[float] = None,
                     analysis_fps_actual: Optional[float] = None,
                     inference_latency_ms: Optional[float] = None,
                     source_type: str = "camo",
                     live_entities: Optional[List[dict]] = None) -> None:
        """Push live pipeline metrics to the backend /api/status singleton.

        Called from run_pipeline.py each stats tick so the dashboard's status
        bar reflects the real AI engine + camera + processing state. If the
        backend status router is not importable (e.g. backend not installed),
        this is a no-op - the live pipeline must not depend on the backend.

        Metrics are SEPARATED honestly:
          - capture_fps: how fast the camera delivers frames (NOT AI speed)
          - analysis_fps_actual: how fast the AI pipeline actually processes
          - inference_latency_ms: measured end-to-end per-frame latency
          - source_type: actual source (camo / file / webcam)
          - live_entities: bounding boxes and track state for live dashboard view
        """
        try:
            from backend.routers.status import set_status
        except Exception:
            return
        st = self.stats()
        # Find highest priority current AI state among active detector pairs.
        current_ai_state = "UNKNOWN"
        for mem in self.event_detector._pairs.values():
            if mem.state.value != "NO_BAG":
                current_ai_state = mem.state.value
                if mem.state.value == "VIOLATION_CONFIRMED":
                    break

        proc_fps = analysis_fps_actual if analysis_fps_actual is not None else None
        set_status(
            ai_engine={"status": "online", "model_loaded": True, "classes": []},
            camera={"status": "online", "fps": capture_fps, "resolution": None, "source": source_type},
            processing={
                "fps": proc_fps,  # actual measured AI FPS, NOT capture FPS
                "latency_ms": inference_latency_ms,  # measured, NOT 1000/analysis_fps
                "analysis_fps": self.config.analysis_fps,  # configured target
            },
            buffer={
                "window_seconds": self.config.buffer_seconds,
                "frames_buffered": st["frames_buffered"],
                "buffer_duration": st["buffer_duration"],
            },
            live_state={
                "ai_state": current_ai_state,
                "active_pairs": len(self.event_detector._pairs),
                "entities": live_entities or [],
            },
        )

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "frames_buffered": len(self.buffer),
            "buffer_duration": self.buffer.current_duration(),
            "active_pairs": len(self.event_detector._pairs),
            "pending_evidence": len(self._pending),
            "confirmed_events": len(self.events),
            "rejected_candidates": len(self.event_detector.rejected_events),
        }
