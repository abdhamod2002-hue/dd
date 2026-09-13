import {
  AlertOctagon,
  Clock,
  Layers,
  Users,
  Activity,
} from "lucide-react";
import { cn } from "../lib/utils";
import type {
  PipelineStageStatus,
  VideoAnalysisJob,
} from "../types";

interface StageDefinition {
  key: string;
  aliases: string[];
  step: number;
  display_name: string;
  algorithm: string;
}

const DEFAULT_STAGES: StageDefinition[] = [
  { key: "video_input", aliases: ["video_input"], step: 1, display_name: "Video Input", algorithm: "Decord / OpenCV VideoCapture" },
  { key: "person_detection", aliases: ["person_detection"], step: 2, display_name: "Person Detection", algorithm: "YOLOv8 Detection" },
  { key: "waste_detection", aliases: ["waste_detection", "object_detection"], step: 3, display_name: "Object / Waste Detection", algorithm: "YOLO Waste / Trash Detector" },
  { key: "tracking", aliases: ["tracking"], step: 4, display_name: "Tracking", algorithm: "ByteTrack Multi-Object Tracker" },
  { key: "person_identity", aliases: ["person_identity"], step: 5, display_name: "Person Identity", algorithm: "PersonIdentityManager (Appearance Re-ID)" },
  { key: "object_identity", aliases: ["object_identity"], step: 6, display_name: "Object Identity", algorithm: "ObjectTrackManager & Handheld Association" },
  { key: "pose_estimation", aliases: ["pose_estimation", "pose"], step: 7, display_name: "Pose Estimation", algorithm: "MoveNet SinglePose (Wrists & Ankles)" },
  { key: "ownership_association", aliases: ["ownership_association"], step: 8, display_name: "Ownership / Association", algorithm: "Spatial & Temporal IoU Proximity" },
  { key: "temporal_event_detection", aliases: ["temporal_event_detection", "event_detection"], step: 9, display_name: "Temporal Event Detection", algorithm: "Littering FSM State Machine" },
  { key: "evidence_assembly", aliases: ["evidence_assembly"], step: 10, display_name: "Evidence Assembly", algorithm: "Cropping, Keyframes & Video Clipper" },
  { key: "database_persistence", aliases: ["database_persistence"], step: 11, display_name: "Database Persistence", algorithm: "SQLAlchemy Models & Event Records" },
  { key: "api_dashboard", aliases: ["api_dashboard"], step: 12, display_name: "API / Dashboard Manifest", algorithm: "FastAPI Analysis Manifest Engine" },
];

interface Props {
  job: VideoAnalysisJob;
  report?: any;
}

export function PipelineStageMonitor({ job, report }: Props) {
  // Extract live telemetry if present
  let telemetry: any = null;
  if (report?.pipeline_telemetry) {
    telemetry = report.pipeline_telemetry;
  } else if (report?.stages) {
    telemetry = report;
  } else if (job.report_json) {
    try {
      const parsed = typeof job.report_json === "string" ? JSON.parse(job.report_json) : job.report_json;
      if (parsed?.pipeline_telemetry) {
        telemetry = parsed.pipeline_telemetry;
      } else if (parsed?.stages) {
        telemetry = parsed;
      }
    } catch {}
  }

  // Normalize stages dictionary from Array or Object
  const rawStages = telemetry?.stages || job.stages || null;
  const normalizedStages: Record<
    string,
    { status: PipelineStageStatus; detail?: string; error?: string; step?: number; display_name?: string }
  > = {};

  const registerStage = (id: string, s: any) => {
    if (!id || !s) return;
    const rawStatus = String(s.status || "pending").toLowerCase() as PipelineStageStatus;
    const def = DEFAULT_STAGES.find((d) => d.key === id || d.aliases.includes(id));
    const entry = {
      status: rawStatus,
      detail: s.detail || undefined,
      error: s.error || undefined,
      step: s.order || s.step || def?.step,
      display_name: s.name?.replace(/^\d+\.\s*/, "") || s.display_name || def?.display_name || id,
    };
    normalizedStages[id] = entry;
    if (def) {
      normalizedStages[def.key] = entry;
      def.aliases.forEach((alias) => {
        normalizedStages[alias] = entry;
      });
    }
  };

  if (Array.isArray(rawStages)) {
    rawStages.forEach((s) => {
      const id = s.id || s.key || s.name;
      if (id) registerStage(id, s);
    });
  } else if (rawStages && typeof rawStages === "object") {
    Object.entries(rawStages).forEach(([key, val]) => {
      registerStage(key, val);
    });
  }

  const isFailed = job.status === "failed" || telemetry?.status === "failed";
  const isCancelled = job.status === "cancelled" || telemetry?.status === "cancelled";
  const isCompleted = job.status === "completed" || telemetry?.status === "completed";
  const isRunning = !isFailed && !isCancelled && !isCompleted && (job.status === "processing" || job.status === "queued");

  // Locate specifically failed or running stages from the stages map
  const failedStageDef = isFailed
    ? DEFAULT_STAGES.find((s) => {
        const info = normalizedStages[s.key];
        return info && info.status === "failed";
      })
    : null;

  const runningStageDef = isRunning
    ? DEFAULT_STAGES.find((s) => {
        const info = normalizedStages[s.key];
        return info && info.status === "running";
      })
    : null;

  // Determine active stage key and step
  let rawStageKey = telemetry?.current_stage_id || telemetry?.current_stage || job.current_stage || "";
  let currentStep = telemetry?.current_step || telemetry?.stage_step || job.stage_step || 1;

  // If stage key has format "Stage 9/12 — TEMPORAL EVENT DETECTION", extract step
  const parsedStepMatch = typeof rawStageKey === "string" ? rawStageKey.match(/Stage\s+(\d+)\/12/i) : null;
  if (parsedStepMatch) {
    currentStep = parseInt(parsedStepMatch[1], 10);
    const matchedDef = DEFAULT_STAGES.find((s) => s.step === currentStep);
    if (matchedDef) rawStageKey = matchedDef.key;
  }

  if (isFailed && failedStageDef) {
    currentStep = failedStageDef.step;
    rawStageKey = failedStageDef.key;
  } else if (isRunning && runningStageDef) {
    currentStep = runningStageDef.step;
    rawStageKey = runningStageDef.key;
  }

  const activeDef = DEFAULT_STAGES.find((s) => s.key === rawStageKey || s.aliases.includes(rawStageKey)) || DEFAULT_STAGES[currentStep - 1] || DEFAULT_STAGES[0];
  const currentStageDisplay = activeDef.display_name;
  const totalStages = 12;

  const currentStageStatus: PipelineStageStatus = isFailed
    ? "failed"
    : isCancelled
    ? "cancelled"
    : isCompleted
    ? "completed"
    : "running";

  const totalFrames = job.total_frames ?? telemetry?.total_frames ?? 0;
  const processedFrames = job.processed_frames ?? telemetry?.last_processed_frame ?? telemetry?.processed_frames ?? 0;
  const pct = totalFrames > 0 ? Math.min(100, Math.round((processedFrames / totalFrames) * 100)) : isCompleted ? 100 : 0;

  // Last successful stage
  let lastSuccessfulDisplay = "None";
  const explicitLastSuccessful = telemetry?.last_successful_stage || job.last_successful_stage;
  if (explicitLastSuccessful) {
    const def = DEFAULT_STAGES.find((s) => s.key === explicitLastSuccessful || s.aliases.includes(explicitLastSuccessful));
    lastSuccessfulDisplay = def ? `Stage ${def.step}/12 — ${def.display_name}` : explicitLastSuccessful;
  } else {
    // Infer the highest completed stage before failure
    const completedDefs = DEFAULT_STAGES.filter((s) => {
      const info = normalizedStages[s.key];
      return info && info.status === "completed";
    });
    if (completedDefs.length > 0) {
      const highestCompleted = completedDefs[completedDefs.length - 1];
      lastSuccessfulDisplay = `Stage ${highestCompleted.step}/12 — ${highestCompleted.display_name}`;
    }
  }

  // Person metrics
  const activePersons = job.active_persons_count ?? telemetry?.person_metrics?.active_people ?? telemetry?.active_persons_count ?? 0;
  const uniquePersons = job.unique_persons_count ?? telemetry?.person_metrics?.unique_people_seen ?? telemetry?.unique_persons_count ?? job.persons_detected ?? 0;
  const totalTrackIds = job.total_person_track_ids ?? telemetry?.person_metrics?.total_track_ids ?? telemetry?.total_person_track_ids ?? job.persons_detected ?? 0;

  // Event metrics
  const candidatesCount = job.candidates_count ?? telemetry?.event_metrics?.candidates_count ?? telemetry?.candidates_count ?? 0;
  const confirmedCount = job.confirmed_count ?? job.events_count ?? telemetry?.event_metrics?.confirmed_events_count ?? telemetry?.confirmed_events ?? 0;
  const rejectedCount = job.rejected_count ?? telemetry?.event_metrics?.rejected_events_count ?? telemetry?.rejected_count ?? 0;
  const activeFsmState = telemetry?.event_metrics?.current_fsm_state || telemetry?.current_fsm_state || "UNKNOWN";

  // Last update time
  const lastUpdateTime = job.last_update_time ?? telemetry?.last_update_time ?? null;

  return (
    <div className="space-y-4">
      {/* 1. Global Progress & Current Stage Header */}
      <div className="rounded-xl border border-slate-800 bg-slate-950/90 p-4 shadow-lg space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div
              className={cn(
                "flex h-10 w-10 items-center justify-center rounded-xl border font-bold mono text-sm",
                isCompleted
                  ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-400"
                  : isFailed
                  ? "bg-rose-500/15 border-rose-500/40 text-rose-400"
                  : isCancelled
                  ? "bg-amber-500/15 border-amber-500/40 text-amber-400"
                  : "bg-cyan-500/15 border-cyan-500/40 text-cyan-400 animate-pulse"
              )}
            >
              {isCompleted ? "100%" : `${pct}%`}
            </div>

            <div>
              <div className="flex items-center gap-2">
                <span className="mono text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Global Progress:
                </span>
                <span
                  className={cn(
                    "rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                    isCompleted
                      ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                      : isFailed
                      ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                      : isCancelled
                      ? "bg-amber-500/20 text-amber-300 border border-amber-500/30"
                      : "bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 animate-pulse"
                  )}
                >
                  {isCancelled ? "CANCELLED" : isFailed ? "FAILED" : isCompleted ? "COMPLETED" : "RUNNING"}
                </span>
              </div>
              <div className="text-sm font-bold text-slate-100 flex items-center gap-2 mt-0.5">
                <span>Stage {currentStep}/{totalStages} —</span>
                <span className={cn(isFailed ? "text-rose-400" : isCancelled ? "text-amber-400" : isCompleted ? "text-emerald-400" : "text-cyan-400")}>
                  {currentStageDisplay}
                </span>
              </div>
            </div>
          </div>

          <div className="text-right text-xs">
            <div className="mono text-slate-300">
              Frames: <span className="font-bold text-slate-100">{processedFrames}</span>
              {totalFrames > 0 && <span> / {totalFrames}</span>}
            </div>
            {lastUpdateTime && (
              <div className="mono text-[10px] text-slate-400 flex items-center justify-end gap-1 mt-0.5">
                <Clock className="h-3 w-3" />
                <span>Updated: {new Date(lastUpdateTime).toLocaleTimeString()}</span>
              </div>
            )}
          </div>
        </div>

        {/* Real Global Progress Bar */}
        <div className="h-2.5 w-full rounded-full bg-slate-900 border border-slate-800 overflow-hidden">
          <div
            className={cn(
              "h-full transition-all duration-300",
              isCompleted
                ? "bg-emerald-500"
                : isFailed
                ? "bg-rose-500"
                : isCancelled
                ? "bg-amber-500"
                : "bg-cyan-500"
            )}
            style={{ width: `${isCompleted ? 100 : Math.max(2, pct)}%` }}
          />
        </div>
      </div>

      {/* 2. Failure Diagnostic Alert */}
      {isFailed && (
        <div className="rounded-xl border border-rose-500/40 bg-rose-950/25 p-4 space-y-3">
          <div className="flex items-center gap-2 text-rose-300 font-bold text-sm">
            <AlertOctagon className="h-5 w-5 text-rose-400 shrink-0" />
            <span>Pipeline Execution Failed at Stage {currentStep}: {currentStageDisplay}</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs text-slate-300 pt-1">
            <div className="rounded-lg bg-slate-900/80 p-2 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">Failed Stage</span>
              <span className="font-semibold text-rose-300">{currentStageDisplay}</span>
            </div>
            <div className="rounded-lg bg-slate-900/80 p-2 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">Last Successful Stage</span>
              <span className="font-semibold text-emerald-400">{lastSuccessfulDisplay}</span>
            </div>
            <div className="rounded-lg bg-slate-900/80 p-2 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">Last Processed Frame</span>
              <span className="mono font-bold text-slate-200">#{processedFrames}</span>
            </div>
            <div className="rounded-lg bg-slate-900/80 p-2 border border-slate-800">
              <span className="text-[10px] uppercase text-slate-400 block">Processing State</span>
              <span className="font-semibold text-rose-400">HALTED</span>
            </div>
          </div>
          <div className="rounded-lg bg-slate-950/90 p-2.5 border border-rose-900/50 text-xs text-rose-300 mono whitespace-pre-wrap">
            Error: {job.error_message || telemetry?.error_message || telemetry?.error || "Unknown pipeline exception"}
          </div>
        </div>
      )}

      {/* 3. Real 12-Stage Pipeline Monitoring Matrix */}
      <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 shadow-xl space-y-3">
        <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <Layers className="h-4 w-4 text-cyan-400" />
            12-Stage Live AI Pipeline Execution State
          </h3>
          <span className="mono text-[10px] text-slate-400">
            Real Backend Driven · Zero Timer Mocking
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5">
          {DEFAULT_STAGES.map((stage) => {
            const recordedInfo = normalizedStages[stage.key];
            let status: PipelineStageStatus = "pending";

            if (recordedInfo) {
              status = recordedInfo.status;
            } else if (isCompleted) {
              status = "completed";
            } else if (isCancelled) {
              if (stage.step < currentStep) status = "completed";
              else if (stage.step === currentStep) status = "cancelled";
              else status = "pending";
            } else if (isRunning) {
              if (stage.step < currentStep) status = "completed";
              else if (stage.step === currentStep) status = currentStageStatus || "running";
              else status = "pending";
            } else if (isFailed) {
              if (stage.step < currentStep) status = "completed";
              else if (stage.step === currentStep) status = "failed";
              else status = "skipped";
            }

            return (
              <div
                key={stage.key}
                className={cn(
                  "rounded-lg border p-2.5 transition-all text-xs flex flex-col justify-between space-y-1.5",
                  status === "running"
                    ? "border-cyan-500/60 bg-cyan-950/30 shadow-md shadow-cyan-950/50"
                    : status === "completed"
                    ? "border-emerald-500/30 bg-emerald-950/15"
                    : status === "failed"
                    ? "border-rose-500/50 bg-rose-950/20"
                    : status === "cancelled"
                    ? "border-amber-500/30 bg-amber-950/15"
                    : "border-slate-800 bg-slate-950/60 opacity-60"
                )}
              >
                <div className="flex items-center justify-between gap-1.5">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span className="mono text-[10px] font-bold text-slate-400">
                      {String(stage.step).padStart(2, "0")}
                    </span>
                    <span className="font-semibold text-slate-200 truncate">
                      {stage.display_name}
                    </span>
                  </div>

                  {/* Visual Status Indicator */}
                  <div className="shrink-0 flex items-center gap-1">
                    {status === "completed" && (
                      <span className="flex items-center gap-1 mono text-[10px] font-bold text-emerald-400">
                        <span>✓</span> COMPLETED
                      </span>
                    )}
                    {status === "running" && (
                      <span className="flex items-center gap-1 mono text-[10px] font-bold text-cyan-400 animate-pulse">
                        <span className="inline-block h-2 w-2 rounded-full bg-cyan-400" /> RUNNING
                      </span>
                    )}
                    {status === "failed" && (
                      <span className="flex items-center gap-1 mono text-[10px] font-bold text-rose-400">
                        <span>✕</span> FAILED
                      </span>
                    )}
                    {status === "cancelled" && (
                      <span className="flex items-center gap-1 mono text-[10px] font-bold text-amber-400">
                        <span>⊘</span> CANCELLED
                      </span>
                    )}
                    {status === "skipped" && (
                      <span className="flex items-center gap-1 mono text-[10px] font-bold text-slate-400">
                        <span>⊝</span> SKIPPED
                      </span>
                    )}
                    {status === "pending" && (
                      <span className="flex items-center gap-1 mono text-[10px] text-slate-400">
                        <span>○</span> PENDING
                      </span>
                    )}
                  </div>
                </div>

                <div className="text-[11px] text-slate-400 truncate">
                  {stage.algorithm}
                </div>

                {recordedInfo?.detail && (
                  <div className="mono text-[10px] text-slate-400 truncate pt-1 border-t border-slate-800/60">
                    {recordedInfo.detail}
                  </div>
                )}
                {recordedInfo?.error && (
                  <div className="text-[10px] text-rose-400 truncate">
                    Err: {recordedInfo.error}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* 4. Correct Person Counting & Event State Distinction */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Person Counting Semantics */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 space-y-3 shadow-lg">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
              <Users className="h-4 w-4 text-emerald-400" />
              Person Counting Demarcation
            </h4>
            <span className="text-[9px] mono uppercase text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
              Identity-Aware
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <div className="mono text-lg font-extrabold text-cyan-400">{activePersons}</div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-300 mt-0.5">
                Active People
              </div>
              <div className="text-[9px] text-slate-400 mt-1">In current frame</div>
            </div>

            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <div className="mono text-lg font-extrabold text-emerald-400">{uniquePersons}</div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-300 mt-0.5">
                Unique People
              </div>
              <div className="text-[9px] text-slate-400 mt-1">Identity UID resolved</div>
            </div>

            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <div className="mono text-lg font-extrabold text-amber-400">{totalTrackIds}</div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-300 mt-0.5">
                Total Track IDs
              </div>
              <div className="text-[9px] text-slate-400 mt-1">Raw tracker churn</div>
            </div>
          </div>

          <p className="text-[10px] text-slate-400 leading-relaxed">
            <span className="font-semibold text-slate-300">Auditing Note:</span> Raw ByteTrack IDs can churn across occlusions and view changes. The honest unique human count is represented by{" "}
            <span className="font-semibold text-emerald-400">Unique People Seen</span>, maintained by{" "}
            <code className="text-emerald-300">PersonIdentityManager</code>.
          </p>
        </div>

        {/* Event State & Confirmation Lifecycle */}
        <div className="rounded-xl border border-slate-800 bg-slate-900/80 p-4 space-y-3 shadow-lg">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-1.5">
              <Activity className="h-4 w-4 text-rose-400" />
              Event Lifecycle & Confirmation Matrix
            </h4>
            <span className="text-[9px] mono uppercase text-rose-400 bg-rose-500/10 px-1.5 py-0.5 rounded border border-rose-500/20">
              Active FSM: {activeFsmState}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <div className="mono text-lg font-extrabold text-amber-400">{candidatesCount}</div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-300 mt-0.5">
                Candidates
              </div>
              <div className="text-[9px] text-slate-400 mt-1">Potential interactions</div>
            </div>

            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <div className="mono text-lg font-extrabold text-rose-400">{confirmedCount}</div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-300 mt-0.5">
                Confirmed
              </div>
              <div className="text-[9px] text-slate-400 mt-1">Full gates validated</div>
            </div>

            <div className="rounded-lg bg-slate-950/80 border border-slate-800 p-2.5">
              <div className="mono text-lg font-extrabold text-slate-400">{rejectedCount}</div>
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-300 mt-0.5">
                Rejected
              </div>
              <div className="text-[9px] text-slate-400 mt-1">Failed gate criteria</div>
            </div>
          </div>

          <p className="text-[10px] text-slate-400 leading-relaxed">
            A candidate object interaction is only promoted to a{" "}
            <span className="font-semibold text-rose-400">Confirmed Violation</span> after temporal continuity, release trajectory, ground placement, and person separation gates pass.
          </p>
        </div>
      </div>
    </div>
  );
}
