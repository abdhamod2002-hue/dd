import json
import pytest
from backend import models
from backend.models import VideoAnalysisJob
from backend.routers.analysis import (
    PipelineStageTracker,
    request_job_cancellation,
    is_job_cancellation_requested,
    PIPELINE_STAGES,
    _extract_person_identity_info,
    _extract_object_identity_count,
)
from tests.test_backend_api import TestingSessionLocal
from fastapi.testclient import TestClient
from backend.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_pipeline_stages_list_exact_12():
    """Verify all 12 pipeline stages are explicitly defined in sequence."""
    assert len(PIPELINE_STAGES) == 12
    expected_stages = [
        "video_input",
        "person_detection",
        "waste_detection",
        "tracking",
        "person_identity",
        "object_identity",
        "pose_estimation",
        "ownership_association",
        "temporal_event_detection",
        "evidence_assembly",
        "database_persistence",
        "api_dashboard",
    ]
    actual_stages = [s["id"] for s in PIPELINE_STAGES]
    assert actual_stages == expected_stages


def test_pipeline_stage_tracker_lifecycle_and_telemetry():
    """Verify PipelineStageTracker manages stages, transitions, metrics and telemetry properly."""
    tracker = PipelineStageTracker()

    # Initial state
    assert tracker.current_stage_id == "video_input"
    assert tracker.stages["video_input"]["status"] == "PENDING"

    # Start stage 1
    tracker.start_stage("video_input", "Reading container", frame_idx=0)
    assert tracker.stages["video_input"]["status"] == "RUNNING"
    assert tracker.stages["video_input"]["started_at"] is not None

    # Complete stage 1
    tracker.complete_stage("video_input", "100 frames decoded", frame_idx=100)
    assert tracker.stages["video_input"]["status"] == "COMPLETED"
    assert tracker.last_successful_stage_id == "video_input"

    # Start stage 2
    tracker.start_stage("person_detection", "YOLO person detection", frame_idx=1)
    assert tracker.current_stage_id == "person_detection"
    assert tracker.stages["person_detection"]["status"] == "RUNNING"

    # Build telemetry payload with separated person metrics
    telemetry = tracker.build_telemetry(
        job_id=42,
        status="processing",
        processed_frames=50,
        total_frames=100,
        active_persons_count=2,
        peak_concurrent_people=3,
        unique_persons_count=1,
        total_person_track_ids=5,
        candidates_count=2,
        confirmed_events_count=1,
        rejected_events_count=1,
        current_fsm_state="LITTERING_CONFIRMED",
        elapsed_sec=12.5,
    )

    assert "Stage 2/12" in telemetry["current_stage"]
    assert telemetry["current_stage_status"] == "RUNNING"
    assert "Stage 1/12" in telemetry["last_successful_stage"]
    assert telemetry["person_metrics"]["active_people"] == 2
    assert telemetry["person_metrics"]["unique_people_seen"] == 1
    assert telemetry["person_metrics"]["total_track_ids"] == 5
    assert telemetry["event_metrics"]["candidates_count"] == 2
    assert telemetry["event_metrics"]["confirmed_events_count"] == 1
    assert telemetry["event_metrics"]["rejected_events_count"] == 1
    assert len(telemetry["stages"]) == 12


def test_pipeline_stage_tracker_failure_recording():
    """Verify stage failure records exact stage, reason, and error state."""
    tracker = PipelineStageTracker()
    tracker.start_stage("pose_estimation", "Running MoveNet SinglePose", frame_idx=50)
    tracker.fail_stage("pose_estimation", "MoveNet pose estimation out of memory", frame_idx=50)

    assert tracker.stages["pose_estimation"]["status"] == "FAILED"
    assert tracker.stages["pose_estimation"]["error"] == "MoveNet pose estimation out of memory"
    assert tracker.failed_stage_id == "pose_estimation"
    assert tracker.error_message == "MoveNet pose estimation out of memory"

    telemetry = tracker.build_telemetry(
        job_id=42,
        status="failed",
        processed_frames=50,
        total_frames=100,
    )
    assert telemetry["current_stage_status"] == "FAILED"
    assert telemetry["error_message"] == "MoveNet pose estimation out of memory"


def test_pipeline_stage_tracker_cancellation():
    """Verify cancellation marks running stage as cancelled without claiming it failed."""
    tracker = PipelineStageTracker()
    tracker.start_stage("ownership_association", "Spatial IoU proximity", frame_idx=60)
    tracker.cancel("User cancelled analysis", frame_idx=60)

    assert tracker.stages["ownership_association"]["status"] == "CANCELLED"
    assert tracker.stages["ownership_association"]["detail"] == "Cancelled: User cancelled analysis"

    telemetry = tracker.build_telemetry(
        job_id=42,
        status="cancelled",
        processed_frames=60,
        total_frames=100,
    )
    assert telemetry["current_stage_status"] == "CANCELLED"
    assert telemetry["status"] == "cancelled"


def test_cancellation_token_threadsafe():
    """Test the job cancellation token flag."""
    job_id = 99999
    assert is_job_cancellation_requested(job_id) is False

    res = request_job_cancellation(job_id)
    assert res is True
    assert is_job_cancellation_requested(job_id) is True


def test_stop_analysis_api_endpoint(client):
    """Test POST /api/analysis/jobs/{job_id}/stop on queued/processing and completed jobs."""
    db = TestingSessionLocal()
    test_job = VideoAnalysisJob(
        filename="test_stop.mp4",
        original_filename="test_stop.mp4",
        file_path="uploads/test_stop.mp4",
        status="processing",
        total_frames=500,
        processed_frames=120,
    )
    db.add(test_job)
    db.commit()
    db.refresh(test_job)
    job_id = test_job.id

    try:
        # Call stop analysis
        response = client.post(f"/api/analysis/jobs/{job_id}/stop")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == job_id
        assert is_job_cancellation_requested(job_id) is True

        # Calling stop on an already stopped/cancelled job returns the job idempotently
        test_job.status = "cancelled"
        db.commit()
        response_dup = client.post(f"/api/analysis/jobs/{job_id}/stop")
        assert response_dup.status_code == 200
        assert response_dup.json()["status"] == "cancelled"

    finally:
        db.delete(test_job)
        db.commit()
        db.close()


def test_delete_analysis_job_guards_and_confirmation(client):
    """Test DELETE /api/analysis/jobs/{job_id} requires confirmation and blocks running jobs."""
    db = TestingSessionLocal()
    running_job = VideoAnalysisJob(
        filename="running.mp4",
        original_filename="running.mp4",
        file_path="uploads/running.mp4",
        status="processing",
        total_frames=500,
        processed_frames=50,
    )
    db.add(running_job)
    db.commit()
    db.refresh(running_job)
    running_id = running_job.id

    completed_job = VideoAnalysisJob(
        filename="completed.mp4",
        original_filename="completed.mp4",
        file_path="uploads/completed.mp4",
        status="completed",
        total_frames=500,
        processed_frames=500,
    )
    db.add(completed_job)
    db.commit()
    db.refresh(completed_job)
    completed_id = completed_job.id

    try:
        # 1. Missing confirm=true should return 400
        res = client.delete(f"/api/analysis/jobs/{completed_id}")
        assert res.status_code == 400
        assert "confirm=true" in res.json()["detail"]

        # 2. Deleting actively running job should return 400
        res_running = client.delete(f"/api/analysis/jobs/{running_id}?confirm=true")
        assert res_running.status_code == 400
        assert "Cannot delete an actively running analysis job" in res_running.json()["detail"]

        # 3. Deleting completed job with confirm=true should succeed
        res_del = client.delete(f"/api/analysis/jobs/{completed_id}?confirm=true")
        assert res_del.status_code == 200
        assert res_del.json()["deleted"] is True

        # Verify it was removed from db
        check = db.query(VideoAnalysisJob).filter(VideoAnalysisJob.id == completed_id).first()
        assert check is None

    finally:
        # Cleanup
        db.query(VideoAnalysisJob).filter(
            VideoAnalysisJob.id.in_([running_id, completed_id])
        ).delete(synchronize_session=False)
        db.commit()
        db.close()


def test_person_count_semantics_in_model_and_api(client):
    """Verify distinct semantics: active_persons, unique_persons_count, total_person_track_ids."""
    db = TestingSessionLocal()
    telemetry_report = {
        "current_stage": "Stage 7/12 — POSE",
        "current_stage_status": "RUNNING",
        "last_successful_stage": "Stage 6/12 — OBJECT IDENTITY",
        "last_processed_frame": 340,
        "total_frames": 600,
        "last_update_time": "2026-09-12T17:30:00Z",
        "active_persons_count": 2,
        "unique_persons_count": 4,
        "total_person_track_ids": 11,
        "candidates_count": 2,
        "rejected_count": 1,
        "events_count": 1,
    }

    job = VideoAnalysisJob(
        filename="test_telemetry.mp4",
        original_filename="test_telemetry.mp4",
        file_path="uploads/test_telemetry.mp4",
        status="processing",
        total_frames=600,
        processed_frames=340,
        report_json=json.dumps(telemetry_report),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    job_id = job.id

    try:
        # Inspect model dynamic properties
        assert job.current_stage == "Stage 7/12 — POSE"
        assert job.current_stage_status == "RUNNING"
        assert job.active_persons_count == 2
        assert job.unique_persons_count == 4
        assert job.total_person_track_ids == 11
        assert job.candidates_count == 2
        assert job.rejected_count == 1
        assert job.last_processed_frame == 340

        # Query API endpoint to ensure FastAPI serialization includes all fields
        response = client.get(f"/api/analysis/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()

        assert data["current_stage"] == "Stage 7/12 — POSE"
        assert data["current_stage_status"] == "RUNNING"
        assert data["active_persons_count"] == 2
        assert data["unique_persons_count"] == 4
        assert data["total_person_track_ids"] == 11
        assert data["candidates_count"] == 2
        assert data["rejected_count"] == 1
        assert data["last_processed_frame"] == 340
        assert data["status"] == "processing"

    finally:
        db.delete(job)
        db.commit()
        db.close()


def test_adaptive_event_detector_identity_attributes():
    """Verify AdaptiveEventDetector forwards _person_identity, _object_identity, and maps."""
    from adaptive_tuner import AdaptiveEventDetector
    from littering_event_detector import EventDetectorConfig

    cfg = EventDetectorConfig(analysis_fps=8.0)
    adaptive = AdaptiveEventDetector(cfg)

    # Must have the forwarded attributes without raising AttributeError
    assert hasattr(adaptive, "_person_identity")
    assert hasattr(adaptive, "_object_identity")
    assert hasattr(adaptive, "last_person_uid_map")
    assert hasattr(adaptive, "last_object_uid_map")

    # They should match the primary detector
    assert adaptive._person_identity is adaptive.primary._person_identity
    assert adaptive._object_identity is adaptive.primary._object_identity


def test_safe_identity_extraction_helpers():
    """Verify _extract_person_identity_info and _extract_object_identity_count are defensive."""
    from adaptive_tuner import AdaptiveEventDetector
    from littering_event_detector import EventDetectorConfig, LitteringEventDetector

    cfg = EventDetectorConfig(analysis_fps=8.0)

    # 1. Standard detector
    det = LitteringEventDetector(cfg)
    unique_p, p_uids = _extract_person_identity_info(det, {1, 2})
    assert isinstance(unique_p, int)
    assert isinstance(p_uids, dict)
    obj_count = _extract_object_identity_count(det, {"1:cup", "2:bag"})
    assert isinstance(obj_count, int)

    # 2. Adaptive detector
    adaptive = AdaptiveEventDetector(cfg)
    unique_p_ad, p_uids_ad = _extract_person_identity_info(adaptive, {1, 2, 3})
    assert isinstance(unique_p_ad, int)
    assert isinstance(p_uids_ad, dict)
    obj_count_ad = _extract_object_identity_count(adaptive, {"1:cup"})
    assert isinstance(obj_count_ad, int)

    # 3. Dummy / None object fallback
    class EmptyDetector:
        pass

    empty = EmptyDetector()
    unique_p_empty, p_uids_empty = _extract_person_identity_info(empty, {10, 20})
    assert unique_p_empty == 1  # Fallback to 1 if detected_persons_set is non-empty
    assert p_uids_empty == {}
    obj_count_empty = _extract_object_identity_count(empty, {"10:box"})
    assert obj_count_empty == 1

    # 4. None detector fallback
    unique_p_none, p_uids_none = _extract_person_identity_info(None, set())
    assert unique_p_none == 0
    assert p_uids_none == {}
    obj_count_none = _extract_object_identity_count(None, set())
    assert obj_count_none == 0


def test_stage_tracker_failure_attribution_temporal_event_detection():
    """Verify that a failure during temporal_event_detection attributes to Stage 9, not Stage 1."""
    tracker = PipelineStageTracker()
    tracker.complete_stage("video_input", "Decoded 500 frames")
    tracker.current_stage_id = "temporal_event_detection"

    tracker.fail_stage("temporal_event_detection", "Simulated FSM anomaly", frame_idx=15)

    assert tracker.failed_stage_id == "temporal_event_detection"
    assert tracker.stages["temporal_event_detection"]["status"] == "FAILED"
    assert tracker.stages["video_input"]["status"] == "COMPLETED"
    assert tracker.stages["evidence_assembly"]["status"] == "SKIPPED"

    telemetry = tracker.build_telemetry(
        job_id=99,
        status="failed",
        processed_frames=15,
        total_frames=500,
        error_message="Simulated FSM anomaly",
    )

    assert telemetry["current_stage_id"] == "temporal_event_detection"
    assert telemetry["current_step"] == 9
    assert telemetry["last_successful_stage"] == "Stage 1/12 — VIDEO INPUT"
    assert telemetry["status"] == "failed"

