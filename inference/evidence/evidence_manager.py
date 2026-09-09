"""
Evidence Manager — 🔴 core contribution.

When the state machine + voting confirm a littering event, the evidence
manager assembles a defensible artifact:

    snapshot          : the frame at the confirmed event timestamp
    evidence video    : pre_event (T-3s) + event + post_event (T+3s)

It pulls the pre/post frames from the CircularFrameBuffer, so we never
"start recording late" — the seconds before the throw are always
available because the buffer retains them continuously.

Outputs are written to disk under ``store_root/{event_id}/``:
    snapshot.jpg
    evidence.mp4
    metadata.json     (event id, timestamps, track ids, confidence)

OpenCV / FFmpeg are used lazily inside the write methods so the module
imports cleanly without CV libs (tests stub the writer).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional

from inference.capture.circular_buffer import BufferedFrame, BufferSnapshot, CircularFrameBuffer


class EvidenceWriteError(RuntimeError):
    """Raised when evidence snapshot/video cannot be written.

    Replaces the old silent placeholder-byte fallbacks: a failed write
    must surface as an error, not produce a fake file that hides the
    problem from the operator.
    """


@dataclass
class EvidenceRequest:
    """Inputs describing a confirmed littering event."""

    camera_id: str
    person_track_id: int
    object_track_id: int
    object_type: str
    confidence: float
    event_timestamp: float
    pre_seconds: float = 3.0
    post_seconds: float = 3.0
    extra: dict = field(default_factory=dict)


@dataclass
class EvidenceArtifact:
    """Record of a saved evidence bundle."""

    event_id: str
    camera_id: str
    snapshot_path: str
    video_path: str
    metadata_path: str
    duration_seconds: float
    frame_count: int
    created_at: float
    request: EvidenceRequest


class EvidenceManager:
    """
    Coordinates snapshot + video assembly from the circular buffer.

    Note on the post-event window: the post-event frames do not exist yet
    at the moment of confirmation (the event *just* happened). The
    pipeline calls :meth:`finalize` after waiting ``post_seconds`` of real
    time so the buffer has accumulated the post-event segment. This is
    why the live dashboard shows "LITTERING CONFIRMED" with a ~3s delay —
    by design, to assemble complete evidence.
    """

    def __init__(self, store_root: str = "evidence_store", face_capture=None) -> None:
        self.store_root = store_root
        os.makedirs(self.store_root, exist_ok=True)
        # Optional face-evidence capture hook. When None (default) the live
        # evidence path is UNCHANGED. The video-analysis evidence package
        # (inference/visualization/evidence_package.py) performs the actual
        # face_evidence.jpg capture. This hook lets the live pipeline opt in
        # later by passing a FaceEvidenceCapture + feeding per-frame person
        # boxes. Left inert by default to avoid touching the confirmed path.
        self.face_capture = face_capture

    def new_event_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def assemble_snapshot(
        self,
        event_id: str,
        buffer: CircularFrameBuffer,
        request: EvidenceRequest,
    ) -> Optional[EvidenceArtifact]:
        """
        Take the snapshot + pre-event segment immediately at confirmation
        time. The post-event segment is filled later by :meth:`finalize`.
        """
        event_dir = os.path.join(self.store_root, event_id)
        os.makedirs(event_dir, exist_ok=True)

        # find the frame closest to the event timestamp
        frames = buffer.get_around(
            request.event_timestamp,
            pre_seconds=request.pre_seconds,
            post_seconds=0.0,  # at confirmation, post is empty
        )
        if not frames:
            return None

        snapshot_frame = self._closest_frame(frames, request.event_timestamp)
        snapshot_path = os.path.join(event_dir, "snapshot.jpg")
        self._write_snapshot(snapshot_frame, snapshot_path)

        # write metadata stub (updated on finalize)
        metadata_path = os.path.join(event_dir, "metadata.json")
        self._write_metadata(metadata_path, event_id, request, frames_written=len(frames), finalized=False)

        video_path = os.path.join(event_dir, "evidence.mp4")
        return EvidenceArtifact(
            event_id=event_id,
            camera_id=request.camera_id,
            snapshot_path=snapshot_path,
            video_path=video_path,
            metadata_path=metadata_path,
            duration_seconds=0.0,
            frame_count=0,
            created_at=time.time(),
            request=request,
        )

    def finalize(
        self,
        artifact: EvidenceArtifact,
        buffer: CircularFrameBuffer,
    ) -> EvidenceArtifact:
        """
        Called after ``post_seconds`` of real time has elapsed. Pulls the
        full pre+event+post window from the buffer and writes the mp4.
        """
        req = artifact.request
        frames = buffer.get_around(
            req.event_timestamp,
            pre_seconds=req.pre_seconds,
            post_seconds=req.post_seconds,
        )
        duration = 0.0
        if frames:
            duration = frames[-1].timestamp - frames[0].timestamp
        self._write_video(frames, artifact.video_path, req)
        # PHASE 2: browsers cannot decode OpenCV's mp4v output (the empty
        # evidence-video panel root cause). Re-encode to H.264; on failure
        # the mp4v original is kept (honest degradation, never a stub).
        try:
            from inference.visualization.h264 import transcode_to_h264
            transcode_to_h264(artifact.video_path)
        except Exception:
            import logging
            logging.getLogger(__name__).exception(
                "H.264 transcode failed for %s — keeping mp4v original", artifact.video_path)
        # update metadata
        self._write_metadata(
            artifact.metadata_path, artifact.event_id, req,
            frames_written=len(frames), finalized=True, duration=duration,
        )
        artifact.duration_seconds = duration
        artifact.frame_count = len(frames)
        return artifact

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _closest_frame(frames: List[BufferedFrame], target_ts: float) -> BufferedFrame:
        return min(frames, key=lambda f: abs(f.timestamp - target_ts))

    def _write_snapshot(self, frame: BufferedFrame, path: str) -> None:
        """Write a single frame as JPEG. Lazily imports cv2.

        Raises EvidenceWriteError on any failure — we do NOT silently
        write placeholder bytes. A fake snapshot file would hide a real
        pipeline failure and is exactly what the audit flagged.
        """
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        img = frame.frame
        if frame.encoded:
            # already JPEG bytes — write directly
            with open(path, "wb") as f:
                f.write(img)
            return
        try:
            arr = np.frombuffer(img, dtype=np.uint8) if isinstance(img, (bytes, bytearray)) else img
            ok = cv2.imwrite(path, arr)
            if not ok:
                raise EvidenceWriteError(f"cv2.imwrite returned False for {path}")
        except EvidenceWriteError:
            raise
        except Exception as e:
            raise EvidenceWriteError(f"snapshot write failed: {type(e).__name__}: {e}") from e

    def _write_video(self, frames: List[BufferedFrame], path: str, req: EvidenceRequest) -> None:
        """Write frames as an mp4 using OpenCV VideoWriter. Lazy import.

        Raises EvidenceWriteError on any failure — no placeholder bytes.
        An empty `frames` list is a real error (the buffer had no
        evidence window), not a silent stub.
        """
        if not frames:
            raise EvidenceWriteError("cannot write evidence video: no frames in window")
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        try:
            # determine frame size from the first frame
            first = frames[0].frame
            if frames[0].encoded:
                arr = np.frombuffer(first, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    raise EvidenceWriteError("could not decode first encoded frame")
            else:
                img = first
            h, w = img.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30.0
            if len(frames) > 1:
                span = frames[-1].timestamp - frames[0].timestamp
                if span > 0:
                    fps = max(10.0, len(frames) / span)
            writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if not writer.isOpened():
                raise EvidenceWriteError(f"VideoWriter could not open {path} (codec mp4v, {w}x{h})")
            for f in frames:
                if f.encoded:
                    a = np.frombuffer(f.frame, dtype=np.uint8)
                    decoded = cv2.imdecode(a, cv2.IMREAD_COLOR)
                    if decoded is None:
                        continue
                    writer.write(decoded)
                else:
                    writer.write(f.frame)
            writer.release()
        except EvidenceWriteError:
            raise
        except Exception as e:
            raise EvidenceWriteError(f"video write failed: {type(e).__name__}: {e}") from e

    def _write_metadata(
        self, path: str, event_id: str, req: EvidenceRequest,
        frames_written: int, finalized: bool, duration: float = 0.0,
    ) -> None:
        data = {
            "event_id": event_id,
            "camera_id": req.camera_id,
            "person_track_id": req.person_track_id,
            "object_track_id": req.object_track_id,
            "object_type": req.object_type,
            "confidence": req.confidence,
            "event_timestamp": req.event_timestamp,
            "pre_seconds": req.pre_seconds,
            "post_seconds": req.post_seconds,
            "frames_written": frames_written,
            "finalized": finalized,
            "duration_seconds": duration,
            "extra": req.extra,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
