// Domain types matching the FastAPI backend schemas (backend/schemas.py)
// and the new status endpoint (backend/routers/status.py).

export interface Camera {
  id: number;
  name: string;
  location: string;
  status: string;
  created_at: string;
}

export interface Event {
  id: number;
  camera_id: number;
  person_track_id: string | null;
  object_track_id: string | null;
  object_type: string;
  timestamp: string;
  confidence: number;
  status: EventStatus;
  analysis_job_id: number | null;
  created_at: string;
  event_actor_person_track_id?: number | null;
  event_actor_person_uid?: number | null;
  event_object_track_id?: number | null;
  event_object_uid?: number | null;
}

export interface EventList {
  items: Event[];
  total: number;
  limit: number;
  offset: number;
}

export interface Evidence {
  id: number;
  event_id: number;
  image_path: string | null;
  video_path: string | null;
  person_image_path: string | null;
  waste_image_path: string | null;
  clip_path: string | null;
  face_image_path: string | null;
  carry_image_path: string | null;
  release_image_path: string | null;
  ground_image_path: string | null;
  duration_sec: number | null;
  created_at: string;
}

export interface Statistics {
  total_events: number;
  events_today: number;
  per_object_type: Record<string, number>;
  avg_confidence: number;
}

export interface AnalysisMarker {
  label: string;
  frame: number;
  timestamp: number;
  kind: "person" | "object" | "event";
  event_id?: string;
  person_track_id?: number | string;
  object_track_id?: number | string;
}

export interface EventReview {
  event: Event;
  evidence: Evidence[];
  job: VideoAnalysisJob | null;
  report: any | null;
}

export type JobStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";

export type PipelineStageStatus = "pending" | "running" | "completed" | "failed" | "skipped" | "cancelled";

export interface PipelineStageInfo {
  step: number;
  name: string;
  display_name: string;
  status: PipelineStageStatus;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  detail: Record<string, any> | null;
}

export interface PipelineTelemetryMetrics {
  active_persons: number;
  peak_concurrent_persons: number;
  unique_persons_seen: number;
  total_track_ids: number;
  objects_detected: number;
  candidates_count: number;
  confirmed_events_count: number;
  rejected_events_count: number;
  active_fsm_state: string;
  elapsed_sec: number;
}

export interface PipelineTelemetry {
  current_stage: string;
  current_stage_status: PipelineStageStatus;
  current_step: number;
  total_stages: number;
  last_successful_stage: string | null;
  last_processed_frame: number;
  total_frames: number;
  last_update_time: string;
  error: string | null;
  cancellation_requested: boolean;
  stages: Record<string, PipelineStageInfo>;
  metrics: PipelineTelemetryMetrics;
}

export interface VideoAnalysisJob {
  id: number;
  filename: string;
  original_filename: string;
  status: JobStatus;
  duration_sec: number | null;
  total_frames: number | null;
  processed_frames: number;
  fps: number | null;
  processing_fps: number | null;
  events_count: number;
  persons_detected: number;
  objects_detected: number;
  report_json: string | null;
  analyzed_video_path: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  original_video_path: string | null;
  manifest_json: string | null;
  analysis_id: number | null;
  // Live stage & observability telemetry
  current_stage?: string | null;
  current_stage_status?: PipelineStageStatus | null;
  stage_step?: number | null;
  total_stages?: number | null;
  stage_name_display?: string | null;
  last_successful_stage?: string | null;
  active_persons_count?: number | null;
  unique_persons_count?: number | null;
  total_person_track_ids?: number | null;
  candidates_count?: number | null;
  rejected_count?: number | null;
  confirmed_count?: number | null;
  last_processed_frame?: number | null;
  last_update_time?: string | null;
  cancellation_requested?: boolean | null;
  stages?: Record<string, PipelineStageInfo> | null;
}

/** Per-analysis artifact manifest (backend/mirrors _build_analysis_manifest). */
export interface AnalysisManifest {
  analysis_id: number;
  job_id: number;
  original_filename: string;
  original_video: string | null;
  analyzed_video: string | null;
  frames_jsonl: string | null;
  event_clips: Array<{
    event_id: string | null;
    evidence_dir: string | null;
    snapshot: string | null;
    person: string | null;
    waste: string | null;
    carry: string | null;
    release: string | null;
    ground: string | null;
    clip: string | null;
    face: string | null;
  }>;
  events: Array<{
    event_id: string | null;
    person_track_id: string | null;
    bag_track_id: string | null;
    confidence: number | null;
    state: string | null;
    reason: string | null;
    frames: Record<string, number> | null;
    timestamps: Record<string, number> | null;
  }>;
  timeline: Array<{ timestamp?: number; frame?: number; state?: string }>;
  markers: AnalysisMarker[];
  metadata: {
    duration_sec: number | null;
    source_fps: number | null;
    resolution: number[] | null;
    processed_frames: number | null;
    persons_count: number | null;
    objects_count: number | null;
    detector_summary: any | null;
    no_candidate_reason: string | null;
    status: string;
    error_message: string | null;
    created_at: string | null;
    started_at: string | null;
    completed_at: string | null;
  };
  sizes_bytes: {
    original: number | null;
    analyzed: number | null;
    frames_jsonl: number | null;
    events_count: number | null;
  };
  final_result: "LITTERING_EVENT_CANDIDATE" | "NO_EVENT";
}

export interface VideoAnalysisJobList {
  items: VideoAnalysisJob[];
  total: number;
  limit: number;
  offset: number;
}

export type EventStatus = "new" | "reviewing" | "confirmed" | "rejected";

// /api/status — the live system status bar payload
export interface SystemStatus {
  system_online: boolean;
  ai_engine: {
    status: "online" | "offline" | "degraded";
    model_loaded: boolean;
    classes: string[];
  };
  camera: {
    status: "online" | "offline" | "waiting";
    fps: number | null;
    resolution: string | null;
    source: string | null;
  };
  processing: {
    fps: number | null;
    latency_ms: number | null;
    analysis_fps: number | null;
  };
  buffer: {
    window_seconds: number;
    frames_buffered: number;
    buffer_duration: number;
  };
  live_state?: {
    ai_state: string;
    active_pairs: number;
    entities: Array<{
      trackId: number;
      label: string;
      bbox: { x: number; y: number; w: number; h: number };
      confidence: number;
      isPerson: boolean;
    }>;
  };
  events_today: number;
  active_cameras: number;
  updated_at: string;
}

// AI state shown on live monitoring — mirrors LitterState enum
export type LitterState =
  | "UNKNOWN"
  | "INTERACTING"
  | "HOLDING"
  | "RELEASE"
  | "OBJECT_ON_GROUND"
  | "PERSON_AWAY"
  | "SUSPICIOUS"
  | "LITTERING_CONFIRMED"
  | "NORMAL";

// The AI reasoning checklist for the event detail page
export interface AIReasoningStep {
  label: string;
  satisfied: boolean;
}
