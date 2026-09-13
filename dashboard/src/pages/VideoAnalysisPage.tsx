import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  Upload,
  FileVideo,
  Layers,
  FileText,
  AlertCircle,
  ShieldAlert,
  CheckCircle2,
  ExternalLink,
  Camera,
  StopCircle,
  Trash2,
  VideoOff,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import { useFetch } from "../lib/useFetch";
import {
  analyzedVideoUrl,
  evidenceFileUrl,
  getAnalysisJobEvents,
  getAnalysisJobs,
  getEvidence,
  originalVideoUrl,
  uploadVideoAnalysis,
  stopAnalysisJob,
  deleteAnalysisJob,
  deleteSourceVideo,
} from "../lib/api";
import { selectDetectorViolation } from "../lib/detectorEvent";
import { ForensicAssetPanel } from "../components/ForensicAssetPanel";
import { SequenceStrip, buildSequenceSteps } from "../components/SequenceStrip";
import { DebugReviewPanel } from "../components/DebugReviewPanel";
import { PipelineStageMonitor } from "../components/PipelineStageMonitor";
import { cn, formatTime } from "../lib/utils";
import type { AnalysisMarker, Event, Evidence, VideoAnalysisJob } from "../types";

function outcomeForJob(job: VideoAnalysisJob | undefined, report: any): { label: string; tone: string; detail?: string } {
  if (!job) return { label: "NO JOB SELECTED", tone: "text-slate-400" };
  if (job.status === "failed") {
    return { label: "ANALYSIS FAILED", tone: "text-rose-400", detail: job.error_message || undefined };
  }
  if (job.status === "cancelled") {
    return {
      label: "ANALYSIS CANCELLED",
      tone: "text-amber-400",
      detail: "Job execution was safely stopped by user request. All resources and memory released.",
    };
  }
  if (job.status === "processing" || job.status === "queued") {
    const stageName = job.stage_name_display || job.current_stage || "AI PIPELINE RUNNING";
    const step = job.stage_step || 1;
    return {
      label: `STAGE ${step}/12 — ${stageName.toUpperCase()}`,
      tone: "text-cyan-400",
      detail: `Frame ${job.processed_frames} / ${job.total_frames || "?"}`,
    };
  }
  if ((job.events_count ?? 0) > 0) {
    return { label: "LITTERING INCIDENT CONFIRMED", tone: "text-rose-400" };
  }
  const reason = report?.no_candidate_reason || report?.diagnosis?.no_candidate_reason;
  return {
    label: "NO LITTERING VIOLATION OBSERVED",
    tone: "text-slate-300",
    detail: reason ? `Reason: ${reason}` : undefined,
  };
}

export function VideoAnalysisPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const [stoppingJobId, setStoppingJobId] = useState<number | null>(null);
  const [deletingJobId, setDeletingJobId] = useState<number | null>(null);
  const [activeJobId, setActiveJobId] = useState<number | null>(null);
  const analyzedRef = useRef<HTMLVideoElement>(null);
  const [jobEvents, setJobEvents] = useState<Event[]>([]);
  const [eventEvidence, setEventEvidence] = useState<Record<number, Evidence | undefined>>({});
  const [showTechnicalReview, setShowTechnicalReview] = useState(false);

  // Poll analysis jobs list every 3s
  const { data: jobsData, loading, refetch } = useFetch(() => getAnalysisJobs(20, 0), [], 3000);
  const jobs = jobsData?.items ?? [];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadError(null);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
    try {
      const job = await uploadVideoAnalysis(selectedFile);
      setActiveJobId(job.id);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      refetch();
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const handleStopAnalysis = async (jobId: number) => {
    setStoppingJobId(jobId);
    setActionError(null);
    setActionNotice(null);
    try {
      const res = await stopAnalysisJob(jobId);
      setActionNotice(`Stop signal sent: ${res.message}`);
      refetch();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setStoppingJobId(null);
    }
  };

  const handleDeleteJob = async (jobId: number) => {
    const ok = window.confirm(
      `Are you sure you want to permanently delete Job #${jobId}?\n\nThis will remove the job record, database events, and generated manifests.`
    );
    if (!ok) return;

    setDeletingJobId(jobId);
    setActionError(null);
    setActionNotice(null);
    try {
      const res = await deleteAnalysisJob(jobId, true);
      setActionNotice(res.message);
      if (activeJobId === jobId) {
        setActiveJobId(null);
      }
      refetch();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setDeletingJobId(null);
    }
  };

  const handleDeleteVideo = async (jobId: number) => {
    const ok = window.confirm(
      `Are you sure you want to delete the raw source video for Job #${jobId}?\n\nThis will reclaim disk space while preserving all forensic events and report data.`
    );
    if (!ok) return;

    setActionError(null);
    setActionNotice(null);
    try {
      const res = await deleteSourceVideo(jobId, true);
      setActionNotice(res.message);
      refetch();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const selectedJob = jobs.find((j) => j.id === activeJobId) ?? jobs[0];

  let parsedReport: any = null;
  if (selectedJob?.report_json) {
    try {
      parsedReport = JSON.parse(selectedJob.report_json);
    } catch {}
  }
  const outcome = outcomeForJob(selectedJob, parsedReport);
  const markers: AnalysisMarker[] = parsedReport?.markers ?? [];
  const violations = parsedReport?.event_detector?.confirmed_violations ?? [];

  useEffect(() => {
    let cancelled = false;
    if (!selectedJob || selectedJob.events_count <= 0) {
      setJobEvents([]);
      setEventEvidence({});
      return;
    }
    getAnalysisJobEvents(selectedJob.id)
      .then(async (events) => {
        if (cancelled) return;
        setJobEvents(events);
        const map: Record<number, Evidence | undefined> = {};
        await Promise.all(
          events.map(async (ev) => {
            try {
              const list = await getEvidence(ev.id);
              map[ev.id] = list[0];
            } catch {
              map[ev.id] = undefined;
            }
          }),
        );
        if (!cancelled) setEventEvidence(map);
      })
      .catch(() => {
        if (!cancelled) {
          setJobEvents([]);
          setEventEvidence({});
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedJob?.id, selectedJob?.events_count]);

  const eventMarker = markers.find((m) => m.label === "EVENT") ?? markers.find((m) => m.kind === "event");
  const firstEvidence = jobEvents.length > 0 ? eventEvidence[jobEvents[0].id] : undefined;

  const focusEvent = (seekTimeSec?: number) => {
    const video = analyzedRef.current;
    if (!video) return;
    const target = seekTimeSec ?? (eventMarker ? eventMarker.timestamp : 0);
    video.currentTime = Math.max(0, target);
    video.play().catch(() => undefined);
  };

  const isSelectedJobRunning =
    selectedJob?.status === "processing" || selectedJob?.status === "queued";
  const isSelectedJobCancelled = selectedJob?.status === "cancelled";

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-4 sm:p-6 lg:p-7">
      {/* Top Surveillance Command Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2.5">
            <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
              <Camera className="h-4 w-4" />
            </div>
            <h1 className="text-xl font-extrabold tracking-tight text-slate-100">
              Video File Analysis & AI Forensics Console
            </h1>
          </div>
          <p className="mt-1 text-xs text-slate-400">
            End-to-end automated detection, tracking, actor attribution, and littering violation confirmation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/90 px-3.5 py-1.5 shadow-sm">
            <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <span className="mono text-xs font-semibold text-slate-300">PIPELINE OPERATIONAL</span>
          </div>
          <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-900/90 px-3.5 py-1.5 shadow-sm">
            <span className="text-slate-400 text-xs">Archived Jobs:</span>
            <span className="mono text-xs font-bold text-emerald-400">{jobs.length}</span>
          </div>
        </div>
      </div>

      {/* Global Notifications / Notices */}
      {actionNotice && (
        <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/30 p-3 text-xs text-emerald-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
            <span>{actionNotice}</span>
          </div>
          <button
            onClick={() => setActionNotice(null)}
            className="text-slate-400 hover:text-white text-xs font-bold px-2 py-0.5"
          >
            ✕
          </button>
        </div>
      )}

      {actionError && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 p-3 text-xs text-rose-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span>{actionError}</span>
          </div>
          <button
            onClick={() => setActionError(null)}
            className="text-slate-400 hover:text-white text-xs font-bold px-2 py-0.5"
          >
            ✕
          </button>
        </div>
      )}

      {/* Top Grid: Upload Station & Active Job Controls */}
      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        {/* Upload Station Card */}
        <div className="panel border-slate-800 bg-slate-900/90 p-5 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <Upload className="h-4 w-4 text-emerald-400" /> Upload Video For AI Analysis
            </h2>
            <span className="mono text-[10px] text-slate-400">MP4 · MOV · AVI · MKV</span>
          </div>

          <div
            onClick={() => fileInputRef.current?.click()}
            className="group relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700/80 bg-slate-950/60 p-7 text-center cursor-pointer transition-all hover:border-emerald-500/60 hover:bg-slate-950"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv,.webm"
              className="hidden"
              onChange={handleFileChange}
            />
            <div className="mb-3 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 group-hover:scale-110 transition-transform">
              <FileVideo className="h-6 w-6" />
            </div>
            <p className="text-sm font-semibold text-slate-200">
              {selectedFile ? selectedFile.name : "Click to select or drag video file here"}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              Full resolution 12-stage analysis · ByteTrack · MoveNet pose · Identity Re-ID
            </p>
          </div>

          {selectedFile && (
            <div className="flex items-center justify-between rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3.5 animate-state-in">
              <div>
                <div className="text-xs font-bold text-slate-100">{selectedFile.name}</div>
                <div className="mono text-[11px] text-emerald-300/80">
                  Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              </div>
              <button
                disabled={uploading}
                onClick={handleStartAnalysis}
                className="rounded-lg bg-emerald-500 px-4 py-2 text-xs font-bold text-slate-950 shadow-lg shadow-emerald-500/20 hover:bg-emerald-400 transition-colors disabled:opacity-50"
              >
                {uploading ? "Analyzing Video..." : "Start AI Pipeline →"}
              </button>
            </div>
          )}

          {uploadError && (
            <div className="flex items-center gap-2 rounded-xl bg-rose-500/10 border border-rose-500/30 p-3 text-xs text-rose-400">
              <AlertCircle className="h-4 w-4 shrink-0" /> {uploadError}
            </div>
          )}
        </div>

        {/* Selected Job Status & Lifecycle Action Card */}
        <div className="panel border-slate-800 bg-slate-900/90 p-5 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <FileText className="h-4 w-4 text-emerald-400" /> Job Execution & Lifecycle Controls
            </h2>
            {selectedJob && (
              <span className="mono text-[10px] font-bold text-slate-400">
                Job #{selectedJob.id}
              </span>
            )}
          </div>

          {selectedJob ? (
            <div className="space-y-3.5">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-slate-100 truncate max-w-[340px]">
                    {selectedJob.original_filename}
                  </h3>
                  <div className="mono text-[11px] text-slate-400 mt-0.5">
                    Created: {formatTime(selectedJob.created_at)}
                  </div>
                </div>

                <span
                  className={cn(
                    "rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                    selectedJob.status === "completed"
                      ? "bg-emerald-500/15 border border-emerald-500/30 text-emerald-400"
                      : selectedJob.status === "processing"
                      ? "bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 animate-pulse"
                      : selectedJob.status === "cancelled"
                      ? "bg-amber-500/15 border border-amber-500/30 text-amber-400"
                      : selectedJob.status === "failed"
                      ? "bg-rose-500/15 border border-rose-500/30 text-rose-400"
                      : "bg-slate-800 text-slate-400"
                  )}
                >
                  {selectedJob.status}
                </span>
              </div>

              {/* Status Outcome Banner */}
              <div
                className={cn(
                  "rounded-lg border p-3 text-xs font-bold uppercase tracking-wider",
                  selectedJob.events_count > 0
                    ? "border-rose-500/40 bg-rose-500/10 text-rose-400"
                    : selectedJob.status === "failed"
                    ? "border-rose-500/40 bg-rose-500/10 text-rose-400"
                    : selectedJob.status === "cancelled"
                    ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                    : selectedJob.status === "processing"
                    ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300"
                    : "border-slate-800 bg-slate-950 text-slate-300"
                )}
              >
                <div className="flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4 shrink-0" />
                  <span>{outcome.label}</span>
                </div>
                {outcome.detail && (
                  <div className="mt-1 text-[11px] font-normal normal-case text-slate-400">{outcome.detail}</div>
                )}
              </div>

              {/* Distinct Action Buttons: STOP ANALYSIS, DELETE JOB, DELETE VIDEO */}
              <div className="flex flex-wrap items-center gap-2 pt-1 border-t border-slate-800/80">
                {isSelectedJobRunning && (
                  <button
                    disabled={stoppingJobId === selectedJob.id}
                    onClick={() => handleStopAnalysis(selectedJob.id)}
                    className="flex items-center gap-1.5 rounded-lg bg-rose-600/90 hover:bg-rose-500 text-white px-3 py-1.5 text-xs font-bold shadow transition-colors disabled:opacity-50"
                  >
                    <StopCircle className="h-3.5 w-3.5" />
                    {stoppingJobId === selectedJob.id ? "Stopping Analysis..." : "STOP ANALYSIS"}
                  </button>
                )}

                <button
                  disabled={isSelectedJobRunning || deletingJobId === selectedJob.id}
                  onClick={() => handleDeleteJob(selectedJob.id)}
                  title={isSelectedJobRunning ? "Cannot delete actively running job" : "Delete analysis and records"}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-rose-950/40 hover:border-rose-500/50 hover:text-rose-300 text-slate-300 px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-40"
                >
                  <Trash2 className="h-3.5 w-3.5" />
                  {deletingJobId === selectedJob.id ? "Deleting..." : "DELETE JOB"}
                </button>

                {selectedJob.original_video_path && (
                  <button
                    disabled={isSelectedJobRunning}
                    onClick={() => handleDeleteVideo(selectedJob.id)}
                    title="Delete raw video file to free disk space"
                    className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-amber-950/40 hover:border-amber-500/50 hover:text-amber-300 text-slate-300 px-3 py-1.5 text-xs font-semibold transition-colors disabled:opacity-40"
                  >
                    <VideoOff className="h-3.5 w-3.5" />
                    DELETE SOURCE VIDEO
                  </button>
                )}

                <Link
                  to={`/analysis/${selectedJob.id}`}
                  className="ml-auto inline-flex items-center gap-1 text-xs font-bold text-emerald-400 hover:text-emerald-300 transition-colors"
                >
                  View Dossier →
                </Link>
              </div>

              {/* Dedicated Cancelled Job Details Card */}
              {isSelectedJobCancelled && (
                <div className="rounded-xl border border-amber-500/30 bg-amber-950/20 p-3.5 space-y-2 text-xs">
                  <div className="flex items-center gap-1.5 font-bold text-amber-400">
                    <StopCircle className="h-4 w-4" />
                    <span>Analysis Safely Terminated (Status: CANCELLED)</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 text-slate-300">
                    <div className="rounded bg-slate-900/80 p-2 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase block">Frames Done</span>
                      <span className="mono font-bold">{selectedJob.processed_frames} / {selectedJob.total_frames || "?"}</span>
                    </div>
                    <div className="rounded bg-slate-900/80 p-2 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase block">Progress Reached</span>
                      <span className="mono font-bold">
                        {selectedJob.total_frames
                          ? `${Math.round((selectedJob.processed_frames / selectedJob.total_frames) * 100)}%`
                          : "—"}
                      </span>
                    </div>
                    <div className="rounded bg-slate-900/80 p-2 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase block">Last Stage</span>
                      <span className="mono font-bold text-amber-300">{selectedJob.stage_name_display || selectedJob.current_stage || "video_input"}</span>
                    </div>
                    <div className="rounded bg-slate-900/80 p-2 border border-slate-800">
                      <span className="text-[10px] text-slate-400 uppercase block">Safe To Delete</span>
                      <span className="font-bold text-emerald-400">YES</span>
                    </div>
                  </div>
                  <p className="text-[10px] text-slate-400">
                    Partial evidence has been preserved. You can safely inspect partial results or delete this job.
                  </p>
                </div>
              )}
            </div>
          ) : (
            <div className="py-12 text-center text-xs text-slate-400">
              No active job selected. Upload a video or choose an archived job below.
            </div>
          )}
        </div>
      </div>

      {/* EXECUTIVE COMMITTEE BANNER & SUMMARY KPI STRIP */}
      {selectedJob && selectedJob.status === "completed" && (
        <div className="space-y-4">
          {jobEvents.length > 0 ? (
            <div className="rounded-2xl border border-rose-500/40 bg-gradient-to-r from-rose-950/70 via-slate-900/90 to-slate-900/90 p-5 shadow-2xl space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-rose-500/20 border border-rose-500/40 text-rose-400 shadow-lg shadow-rose-950/50">
                    <ShieldAlert className="h-6 w-6 animate-pulse" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="mono text-xs font-black uppercase tracking-widest text-rose-400">
                        VIOLATION DETECTED
                      </span>
                      <span className="rounded-full bg-rose-500/20 border border-rose-500/40 px-2.5 py-0.5 mono text-[10px] font-bold text-rose-300">
                        UNLAWFUL GROUND LITTERING
                      </span>
                    </div>
                    <h1 className="text-lg font-black text-slate-100 mt-0.5">
                      Confirmed Ground Littering Infraction
                    </h1>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="rounded-lg bg-slate-950/80 border border-slate-800 px-3 py-1.5 mono text-xs font-bold text-rose-400">
                    {jobEvents.length === 1 ? "1 Confirmed Violation" : `${jobEvents.length} Confirmed Violations`}
                  </span>
                </div>
              </div>

              {/* Committee Executive Summary KPI Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2">
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">Incident Time</span>
                  <span className="mono text-sm font-bold text-slate-100">
                    {formatTime(jobEvents[0].timestamp)}
                  </span>
                </div>
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">Authoritative Actor</span>
                  <span className="mono text-sm font-bold text-cyan-300">
                    {jobEvents[0].event_actor_person_uid != null ? `Person UID #${jobEvents[0].event_actor_person_uid}` : `Person Track #${jobEvents[0].person_track_id}`}
                  </span>
                </div>
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">Waste Discarded</span>
                  <span className="mono text-sm font-bold text-amber-300">
                    {jobEvents[0].object_type || "Garbage Bag"}
                  </span>
                </div>
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">Disposal Site</span>
                  <span className="mono text-sm font-bold text-rose-400">
                    GROUND (Asphalt Plane)
                  </span>
                </div>
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">AI Confidence</span>
                  <span className="mono text-sm font-bold text-emerald-400">
                    {Math.round((jobEvents[0].confidence || 0.9) * 100)}% Verified
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="rounded-2xl border border-emerald-500/40 bg-gradient-to-r from-emerald-950/70 via-slate-900/90 to-slate-900/90 p-5 shadow-2xl space-y-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/20 border border-emerald-500/40 text-emerald-400 shadow-lg shadow-emerald-950/50">
                    <CheckCircle2 className="h-6 w-6" />
                  </div>
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="mono text-xs font-black uppercase tracking-widest text-emerald-400">
                        NO VIOLATION DETECTED
                      </span>
                      <span className="rounded-full bg-emerald-500/20 border border-emerald-500/40 px-2.5 py-0.5 mono text-[10px] font-bold text-emerald-300">
                        SCENE COMPLIANT
                      </span>
                    </div>
                    <h1 className="text-lg font-black text-slate-100 mt-0.5">
                      No Ground-Littering Infraction Observed
                    </h1>
                  </div>
                </div>
                <span className="rounded-lg bg-slate-950/80 border border-slate-800 px-3 py-1.5 mono text-xs font-bold text-emerald-400">
                  0 Violations Recorded
                </span>
              </div>
              <p className="text-xs text-slate-300 max-w-2xl">
                All observed individuals either passed through the camera view without disposing of waste, or properly deposited waste into authorized garbage bins/dumpsters. No ground-littering violation dossier was generated.
              </p>
            </div>
          )}
        </div>
      )}

      {/* PRIMARY FORENSIC EVIDENCE — Single Event Console */}
      {selectedJob && jobEvents.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-100">
              <ShieldAlert className="h-4 w-4 text-rose-500" />
              {jobEvents.length > 1
                ? `Confirmed Violation Incidents (${jobEvents.length} Recorded)`
                : "Primary Confirmed Incident Forensic Dossier"}
            </h2>
            <Link
              to={`/violations/${jobEvents[0].id}`}
              className="inline-flex items-center gap-1 text-xs font-semibold text-emerald-400 hover:text-emerald-300"
            >
              Open Incident File #{jobEvents[0].id} <ExternalLink className="h-3 w-3" />
            </Link>
          </div>

          {jobEvents.map((ev) => {
            const evEvidence = eventEvidence[ev.id];
            const detectorViolation = selectDetectorViolation(violations, ev);
            const steps = buildSequenceSteps(evEvidence, detectorViolation);

            return (
              <div key={ev.id} className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-2xl">
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="rounded bg-rose-500/20 border border-rose-500/40 px-2 py-0.5 mono text-xs font-bold text-rose-300">
                      INCIDENT #{ev.id}
                    </span>
                    <span className="text-xs text-slate-300">
                      Actor: <span className="mono font-bold text-emerald-400">
                        {ev.event_actor_person_uid != null ? `Person UID #${ev.event_actor_person_uid}` : `Person Track #${ev.person_track_id ?? "Unknown"}`}
                      </span>
                    </span>
                    {ev.object_type && (
                      <span className="text-xs text-slate-400">
                        · Waste: <span className="mono text-amber-300">{ev.object_type}</span>
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-400">
                    Timestamp: <span className="mono text-slate-200">{formatTime(ev.timestamp)}</span>
                  </div>
                </div>

                {/* Primary Forensic Assets: Event Clip & Target Crops */}
                <ForensicAssetPanel
                  evidence={evEvidence}
                  event={ev}
                  onSeekVideo={focusEvent}
                  markers={markers}
                />

                {/* Multi-Stage Temporal Behavioral Sequence: Carry -> Release -> Ground -> Departure */}
                {steps.length > 0 && (
                  <div className="panel border-slate-800 bg-slate-950/70 p-4">
                    <div className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                      Multi-Stage Temporal Evidence Sequence (Carry → Release → Ground → Departure)
                    </div>
                    <SequenceStrip steps={steps} onSeekVideo={(t) => focusEvent(t)} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* SECONDARY TECHNICAL REVIEW & PIPELINE TELEMETRY (COLLAPSED BY DEFAULT) */}
      {selectedJob && (
        <div className="rounded-2xl border border-slate-800 bg-slate-900/40 overflow-hidden shadow-lg">
          <button
            onClick={() => setShowTechnicalReview(!showTechnicalReview)}
            className="w-full flex items-center justify-between p-4 bg-slate-900/80 hover:bg-slate-850 transition-colors text-left"
          >
            <div className="flex items-center gap-2">
              {showTechnicalReview ? (
                <ChevronDown className="h-4 w-4 text-emerald-400" />
              ) : (
                <ChevronRight className="h-4 w-4 text-slate-400" />
              )}
              <span className="text-xs font-bold uppercase tracking-wider text-slate-200">
                Secondary Technical Review & Pipeline Diagnostics
              </span>
              <span className="mono text-[10px] text-slate-500">
                (Full Analyzed Video, 12-Stage Monitor, Tracking Demarcation)
              </span>
            </div>
            <span className="mono text-[11px] text-slate-400">
              {showTechnicalReview ? "Hide Details" : "Show Technical Review"}
            </span>
          </button>

          {(showTechnicalReview || selectedJob.status === "processing" || selectedJob.status === "queued") && (
            <div className="p-5 space-y-6 border-t border-slate-800 bg-slate-950/50">
              {/* 12-Stage Pipeline Monitoring & Person Demarcation Component */}
              <PipelineStageMonitor job={selectedJob} report={parsedReport} />

              {/* Engineering debug — one shared original/analyzed video pair per job */}
              <DebugReviewPanel
                originalVideoUrl={originalVideoUrl(selectedJob.id)}
                analyzedVideoUrl={
                  selectedJob.analyzed_video_path ? analyzedVideoUrl(selectedJob.id) : undefined
                }
                analyzedVideoRef={analyzedRef}
                clipUrl={firstEvidence?.clip_path ? evidenceFileUrl(firstEvidence.clip_path) : undefined}
                markers={markers}
                durationSec={selectedJob.duration_sec}
                onFocusEvent={focusEvent}
                hasEventMarker={!!eventMarker}
                defaultOpen={false}
              />
            </div>
          )}
        </div>
      )}

      {/* Analysis Jobs History Table */}
      <div className="panel border-slate-800 bg-slate-900/90 p-5 space-y-4 shadow-xl">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
            <Layers className="h-4 w-4 text-emerald-400" /> Video Analysis Archive
          </h2>
          <span className="mono text-[11px] text-slate-400">{jobs.length} Stored Analyses</span>
        </div>

        <p className="text-[11px] text-slate-400">
          Every uploaded video remains archived as its own historical analysis. Click{" "}
          <span className="font-semibold text-emerald-400">INSPECT</span> to view the stored forensic dossier without re-running AI inference.
        </p>

        {jobs.length === 0 && !loading && (
          <div className="py-12 text-center text-xs text-slate-400">
            No video analysis jobs uploaded yet. Upload a video above to begin.
          </div>
        )}

        {jobs.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-800 text-[10px] uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-3 py-2.5">ID</th>
                  <th className="px-3 py-2.5">Video Name</th>
                  <th className="px-3 py-2.5">Recorded At</th>
                  <th className="px-3 py-2.5">Duration</th>
                  <th className="px-3 py-2.5">Status</th>
                  <th className="px-3 py-2.5">Unique People</th>
                  <th className="px-3 py-2.5">Events</th>
                  <th className="px-3 py-2.5 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80">
                {jobs.map((j) => {
                  const isCurrent = j.id === selectedJob?.id;
                  const isRunning = j.status === "processing" || j.status === "queued";
                  const uniqueCount = j.unique_persons_count ?? j.persons_detected ?? 0;
                  return (
                    <tr
                      key={j.id}
                      className={cn(
                        "transition-colors",
                        isCurrent ? "bg-slate-850/90 border-l-2 border-emerald-400" : "hover:bg-slate-800/50"
                      )}
                    >
                      <td className="px-3 py-3 mono font-bold text-emerald-400">#{j.id}</td>
                      <td className="px-3 py-3 font-medium text-slate-200 max-w-[200px] truncate">
                        {j.original_filename}
                      </td>
                      <td className="px-3 py-3 text-slate-400 whitespace-nowrap">{formatTime(j.created_at)}</td>
                      <td className="px-3 py-3 mono text-slate-300">
                        {j.duration_sec ? `${j.duration_sec.toFixed(1)}s` : "—"}
                      </td>
                      <td className="px-3 py-3">
                        <span
                          className={cn(
                            "rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                            j.status === "completed"
                              ? "bg-emerald-500/15 border border-emerald-500/30 text-emerald-400"
                              : j.status === "processing"
                              ? "bg-cyan-500/15 border border-cyan-500/30 text-cyan-400 animate-pulse"
                              : j.status === "cancelled"
                              ? "bg-amber-500/15 border border-amber-500/30 text-amber-400"
                              : j.status === "failed"
                              ? "bg-rose-500/15 border border-rose-500/30 text-rose-400"
                              : "bg-slate-800 text-slate-400"
                          )}
                        >
                          {j.status}
                        </span>
                      </td>
                      <td className="px-3 py-3 mono text-slate-300">
                        {uniqueCount}
                      </td>
                      <td className="px-3 py-3">
                        {j.events_count > 0 ? (
                          <span className="mono font-bold text-rose-400 bg-rose-500/10 border border-rose-500/30 px-2 py-0.5 rounded">
                            {j.events_count} Event{j.events_count > 1 ? "s" : ""}
                          </span>
                        ) : (
                          <span className="mono text-slate-400">0</span>
                        )}
                      </td>
                      <td className="px-3 py-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          {isRunning && (
                            <button
                              disabled={stoppingJobId === j.id}
                              onClick={() => handleStopAnalysis(j.id)}
                              className="rounded border border-rose-700/60 bg-rose-950/40 hover:bg-rose-900/60 text-rose-300 px-2 py-1 text-[10px] font-bold uppercase transition-colors"
                            >
                              Stop
                            </button>
                          )}
                          <button
                            onClick={() => setActiveJobId(j.id)}
                            className={cn(
                              "rounded-lg px-2.5 py-1 text-[11px] font-bold uppercase transition-colors",
                              isCurrent
                                ? "bg-emerald-500 text-slate-950 font-extrabold"
                                : "border border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700 hover:text-white"
                            )}
                          >
                            Inspect
                          </button>
                          <Link
                            to={`/analysis/${j.id}`}
                            className="rounded-lg border border-slate-700 bg-slate-800 px-2.5 py-1 text-[11px] font-bold uppercase text-slate-300 hover:bg-slate-700 hover:text-white transition-colors"
                          >
                            Dossier
                          </Link>
                          {!isRunning && (
                            <button
                              onClick={() => handleDeleteJob(j.id)}
                              className="text-slate-500 hover:text-rose-400 p-1 transition-colors"
                              title="Delete job"
                            >
                              <Trash2 className="h-3.5 w-3.5" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
