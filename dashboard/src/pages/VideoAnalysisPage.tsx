import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Upload, FileVideo, Layers, FileText, AlertCircle } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { analyzedVideoUrl, evidenceFileUrl, getAnalysisJobEvents, getAnalysisJobs, getEvidence, originalVideoUrl, uploadVideoAnalysis } from "../lib/api";
import { selectDetectorViolation } from "../lib/detectorEvent";
import { ForensicAssetPanel } from "../components/ForensicAssetPanel";
import { SequenceStrip, buildSequenceSteps } from "../components/SequenceStrip";
import { DebugReviewPanel } from "../components/DebugReviewPanel";
import { cn, formatTime } from "../lib/utils";
import type { AnalysisMarker, Event, Evidence } from "../types";

function outcomeForJob(job: any, report: any): { label: string; tone: string; detail?: string } {
  if (!job) return { label: "NO JOB", tone: "text-[var(--text-muted)]" };
  if (job.status === "failed") {
    return { label: "ANALYSIS FAILED", tone: "text-[var(--danger)]", detail: job.error_message || undefined };
  }
  if (job.status === "processing" || job.status === "queued") {
    return { label: "ANALYSIS RUNNING", tone: "text-[var(--warning)]" };
  }
  if ((job.events_count ?? 0) > 0) {
    return { label: "LITTERING EVENT CANDIDATE DETECTED", tone: "text-[var(--danger)]" };
  }
  const reason = report?.no_candidate_reason || report?.diagnosis?.no_candidate_reason;
  return {
    label: "NO LITTERING EVENT CANDIDATE",
    tone: "text-[var(--text-secondary)]",
    detail: reason ? `Reason: ${reason}` : undefined,
  };
}

export function VideoAnalysisPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<number | null>(null);
  const analyzedRef = useRef<HTMLVideoElement>(null);
  const [jobEvents, setJobEvents] = useState<Event[]>([]);
  const [eventEvidence, setEventEvidence] = useState<Record<number, Evidence | undefined>>({});

  // Poll analysis jobs list every 3s
  const { data: jobsData, loading } = useFetch(() => getAnalysisJobs(20, 0), [], 3000);
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
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
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

  // Fetch the REAL, per-event DB rows for this job (not just the report's
  // compact detector dicts) plus each one's OWN evidence — every confirmed
  // event in a job used to render against the same reused `evidence[0]`
  // (the first event's crops/clip), so a job with 2+ confirmed events showed
  // one event's photos under every violation's header. Each event now gets
  // its own fetched Evidence row, matched by DB id, never shared.
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

  const focusEvent = () => {
    const video = analyzedRef.current;
    if (!video || !eventMarker) return;
    video.currentTime = Math.max(0, eventMarker.timestamp);
    video.play().catch(() => undefined);
  };

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-[var(--text-primary)]">Video File Analysis</h1>
          <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
            Upload and analyze recorded CCTV / benchmark videos through the full production AI pipeline.
          </p>
        </div>
      </div>

      {/* Upload Zone & Job Control */}
      <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
        {/* Upload & Active Execution Card */}
        <div className="panel p-5 space-y-5">
          <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)] flex items-center gap-2">
            <Upload className="h-4 w-4 text-[var(--accent)]" /> Upload Video For Full Analysis
          </h2>

          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-[var(--border-default)] hover:border-[var(--accent)] rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-colors bg-[var(--bg-elevated)]"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv,.webm"
              className="hidden"
              onChange={handleFileChange}
            />
            <FileVideo className="h-10 w-10 text-[var(--accent)] mb-3" />
            <p className="text-sm font-semibold text-[var(--text-primary)]">
              {selectedFile ? selectedFile.name : "Click to browse or drop video file"}
            </p>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Supports .mp4, .avi, .mov, .mkv (Max 200MB recommended)
            </p>
          </div>

          {selectedFile && (
            <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-4 flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-[var(--text-primary)]">{selectedFile.name}</div>
                <div className="text-[11px] text-[var(--text-muted)]">
                  Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              </div>
              <button
                disabled={uploading}
                onClick={handleStartAnalysis}
                className="rounded-lg bg-[var(--accent)] px-4 py-2 text-xs font-bold text-black hover:bg-[var(--accent-dim)] transition-colors disabled:opacity-50"
              >
                {uploading ? "Uploading & Starting..." : "Start AI Analysis →"}
              </button>
            </div>
          )}

          {uploadError && (
            <div className="rounded-lg bg-[var(--danger)]/15 border border-[var(--danger)]/30 p-3 text-xs text-[var(--danger)] flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" /> {uploadError}
            </div>
          )}

          {/* Active Job Real-Time Progress View */}
          {selectedJob && (
            <div className="border-t border-[var(--border-subtle)] pt-4 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--accent)]">
                    Active Job #{selectedJob.id}
                  </span>
                  <h3 className="text-sm font-bold text-[var(--text-primary)]">{selectedJob.original_filename}</h3>
                </div>
                <span
                  className={cn(
                    "rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase",
                    selectedJob.status === "completed"
                      ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                      : selectedJob.status === "processing"
                      ? "bg-[var(--warning)]/15 text-[var(--warning)] animate-pulse"
                      : selectedJob.status === "failed"
                      ? "bg-[var(--danger)]/15 text-[var(--danger)]"
                      : "bg-[var(--bg-elevated)] text-[var(--text-muted)]"
                  )}
                >
                  {selectedJob.status}
                </span>
              </div>

              {/* Real Progress Bar */}
              {selectedJob.total_frames && selectedJob.total_frames > 0 ? (
                <div>
                  <div className="flex items-center justify-between text-xs text-[var(--text-secondary)] mb-1">
                    <span>
                      Frames: {selectedJob.processed_frames} / {selectedJob.total_frames}
                    </span>
                    <span className="mono">
                      {Math.round((selectedJob.processed_frames / selectedJob.total_frames) * 100)}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-[var(--bg-base)] overflow-hidden">
                    <div
                      className="h-full bg-[var(--accent)] transition-all duration-300"
                      style={{
                        width: `${Math.min(100, Math.round((selectedJob.processed_frames / selectedJob.total_frames) * 100))}%`
                      }}
                    />
                  </div>
                </div>
              ) : null}

              {/* Final outcome banner */}
              <div className={cn("rounded-lg border p-3 text-xs font-bold uppercase tracking-wider",
                outcome.label === "LITTERING EVENT CANDIDATE DETECTED" ? "border-[var(--danger)]/30 bg-[var(--danger)]/10" :
                outcome.label === "ANALYSIS FAILED" ? "border-[var(--danger)]/30 bg-[var(--danger)]/10" :
                outcome.label === "ANALYSIS RUNNING" ? "border-[var(--warning)]/30 bg-[var(--warning)]/10" :
                "border-[var(--border-subtle)] bg-[var(--bg-base)]"
              )}>
                <div className={outcome.tone}>{outcome.label}</div>
                {outcome.detail && <div className="mt-1 text-[11px] font-normal normal-case text-[var(--text-muted)]">{outcome.detail}</div>}
              </div>

              {/* Job Metrics Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--text-primary)]">
                    {selectedJob.duration_sec ? `${selectedJob.duration_sec.toFixed(1)}s` : "—"}
                  </div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Video Length</div>
                </div>
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--text-primary)]">
                    {selectedJob.processing_fps ? `${selectedJob.processing_fps.toFixed(1)} FPS` : "—"}
                  </div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Processing Speed</div>
                </div>
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--text-primary)]">{selectedJob.persons_detected}</div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Persons Tracked</div>
                </div>
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--danger)]">{selectedJob.events_count}</div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Littering Events</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Diagnostic Report Panel */}
        <div className="panel p-5 space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)] flex items-center gap-2">
            <FileText className="h-4 w-4 text-[var(--accent)]" /> Diagnostic Inspection
          </h2>

          {parsedReport ? (
            <div className="space-y-4 text-xs">
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-3 space-y-2">
                <div className="font-bold text-[var(--text-primary)] mb-1">Pipeline Stages Verification:</div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">YOLO Person Detection:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.yolo_person === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.yolo_person}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">YOLO Object Detection:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.yolo_object === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.yolo_object}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">Color Waste-Bag Fallback:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.color_object === "PASS" ? "text-[var(--accent)]" : "text-[var(--warning)]")}>
                    {parsedReport.diagnosis.color_object ?? "UNKNOWN"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">Detector Source:</span>
                  <span className={cn("font-bold mono", parsedReport.detector_source === "color_fallback_only" ? "text-[var(--warning)]" : "text-[var(--accent)]")}>
                    {parsedReport.detector_source ?? "unknown"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">ByteTrack Tracking:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.tracking === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.tracking}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">Person-Object Association:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.association === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.association}
                  </span>
                </div>
                <div className="flex justify-between border-t border-[var(--border-subtle)] pt-1">
                  <span className="text-[var(--text-secondary)]">Final Outcome:</span>
                  <span className={cn("font-bold", parsedReport.confirmed_events > 0 ? "text-[var(--danger)]" : "text-[var(--text-muted)]")}>
                    {parsedReport.diagnosis.littering_candidate}
                  </span>
                </div>
                {parsedReport.no_candidate_reason && (
                  <div className="flex justify-between">
                    <span className="text-[var(--text-secondary)]">No-Candidate Reason:</span>
                    <span className="mono font-bold text-[var(--warning)]">{parsedReport.no_candidate_reason}</span>
                  </div>
                )}

                {parsedReport.stages && parsedReport.stages.length > 0 && (
                  <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2 space-y-1">
                    <div className="font-bold text-[var(--text-primary)] mb-0.5">Pipeline Execution Stages:</div>
                    {parsedReport.stages.map((s: any) => (
                      <div key={s.name} className="flex items-center justify-between rounded px-2 py-0.5 text-[11px]">
                        <span className="text-[var(--text-secondary)] capitalize">{String(s.name).replace(/_/g, " ")}</span>
                        <span className="flex items-center gap-2">
                          <span className="mono text-[10px] text-[var(--text-muted)] hidden sm:inline">{s.detail}</span>
                          <span
                            className={cn(
                              "font-bold",
                              s.status === "PASS" || s.status === "CONFIRMED"
                                ? "text-[var(--accent)]"
                                : s.status === "FAIL"
                                ? "text-[var(--danger)]"
                                : s.status === "CANDIDATE" || s.status === "WARN"
                                ? "text-[var(--warning)]"
                                : "text-[var(--text-muted)]"
                            )}
                          >
                            {s.status}
                          </span>
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {parsedReport.event_detector && (
                <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-3 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="font-bold text-[var(--text-primary)]">Temporal Event Detector</div>
                    <span className="mono text-[10px] text-[var(--text-muted)]">
                      {parsedReport.event_detector.summary?.acceptance_rate != null
                        ? `${Math.round((parsedReport.event_detector.summary.acceptance_rate || 0) * 100)}% accepted`
                        : "—"}
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2 text-center">
                    <div className="rounded bg-[var(--bg-base)] p-2">
                      <div className="mono text-sm font-bold text-[var(--danger)]">{parsedReport.event_detector.summary?.confirmed_violations ?? 0}</div>
                      <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Confirmed</div>
                    </div>
                    <div className="rounded bg-[var(--bg-base)] p-2">
                      <div className="mono text-sm font-bold text-[var(--warning)]">{parsedReport.event_detector.summary?.rejected_candidates ?? 0}</div>
                      <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Rejected</div>
                    </div>
                    <div className="rounded bg-[var(--bg-base)] p-2">
                      <div className="mono text-sm font-bold text-[var(--text-primary)]">{parsedReport.event_detector.summary?.total_candidates ?? 0}</div>
                      <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Candidates</div>
                    </div>
                  </div>

                  {parsedReport.event_detector.summary?.rejection_reason_counts && Object.keys(parsedReport.event_detector.summary.rejection_reason_counts).length > 0 && (
                    <div>
                      <div className="mb-1 text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Rejection Reasons</div>
                      <div className="space-y-1">
                        {Object.entries(parsedReport.event_detector.summary.rejection_reason_counts).map(([reason, count]: [string, any]) => (
                          <div key={reason} className="flex items-center justify-between rounded bg-[var(--bg-base)] px-2 py-1 text-[11px]">
                            <span className="mono text-[var(--text-secondary)]">{reason}</span>
                            <span className="mono font-bold text-[var(--text-primary)]">{count}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="rounded bg-[var(--bg-base)] p-2 text-[10px] leading-relaxed text-[var(--text-muted)]">
                    A confirmed littering event means the temporal detector observed carry → release → stationary ground → departure with sufficient evidence.
                    It is an assistive review candidate, not a legal determination and not 100% accurate. Full crops/clip for each confirmed
                    event are shown in the Primary Evidence panel(s) below.
                  </div>
                </div>
              )}

              {/* Timeline progression */}
              {parsedReport.timeline && parsedReport.timeline.length > 0 && (
                <div>
                  <div className="font-bold text-[var(--text-primary)] mb-2">Behavior Timeline:</div>
                  <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                    {parsedReport.timeline.map((item: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between rounded bg-[var(--bg-base)] p-1.5 text-[11px]">
                        <span className="mono text-[var(--accent)]">{item.timestamp}s</span>
                        <span className="mono font-semibold">{item.state}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-[var(--text-muted)]">
              Select or run a video analysis job to see step-by-step diagnostic breakdown.
            </div>
          )}
        </div>
      </div>

      {/* PRIMARY EVIDENCE — one panel per confirmed event, each anchored to
          its OWN fetched Evidence row (never a shared/reused one). */}
      {selectedJob && jobEvents.length > 0 && (
        <div className="space-y-5">
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">
            <AlertCircle className="h-4 w-4 text-[var(--danger)]" />
            {jobEvents.length > 1 ? `${jobEvents.length} Littering Event Candidates` : "Littering Event Candidate"}
          </h2>
          {jobEvents.map((ev) => {
            const detectorViolation = selectDetectorViolation(violations, ev);
            const evEvidence = eventEvidence[ev.id];
            return (
              <div key={ev.id} className="space-y-3">
                <ForensicAssetPanel event={ev} evidence={evEvidence} />
                <SequenceStrip steps={buildSequenceSteps(evEvidence, detectorViolation)} />
              </div>
            );
          })}
        </div>
      )}

      {/* Engineering debug — one shared original/analyzed video pair per job */}
      {selectedJob && (
        <DebugReviewPanel
          originalVideoUrl={originalVideoUrl(selectedJob.id)}
          analyzedVideoUrl={selectedJob.analyzed_video_path ? analyzedVideoUrl(selectedJob.id) : undefined}
          analyzedVideoRef={analyzedRef}
          clipUrl={firstEvidence?.clip_path ? evidenceFileUrl(firstEvidence.clip_path) : undefined}
          markers={markers}
          durationSec={selectedJob.duration_sec}
          onFocusEvent={focusEvent}
          hasEventMarker={!!eventMarker}
        />
      )}

      {/* Analysis Jobs History Table */}
      <div className="panel p-5 space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)] flex items-center gap-2">
          <Layers className="h-4 w-4 text-[var(--accent)]" /> Video Analysis History
        </h2>

        <p className="text-[11px] text-[var(--text-muted)]">
          Every uploaded video stays archived as its own historical analysis. Use{" "}
          <span className="font-semibold text-[var(--accent)]">VIEW RESULT</span> to reopen a past
          analysis — it loads the stored result, it never re-runs the AI.
        </p>

        {jobs.length === 0 && !loading && (
          <div className="py-12 text-center text-xs text-[var(--text-muted)]">
            No video analysis jobs uploaded yet. Upload an MP4 above to start.
          </div>
        )}

        {jobs.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-[var(--border-subtle)] text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                <tr>
                  <th className="px-3 py-2">ID</th>
                  <th className="px-3 py-2">Video</th>
                  <th className="px-3 py-2">Date / Time</th>
                  <th className="px-3 py-2">Dur.</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Persons</th>
                  <th className="px-3 py-2">Events</th>
                  <th className="px-3 py-2">Final Result</th>
                  <th className="px-3 py-2">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)]">
                {jobs.map((j) => {
                  return (
                    <tr key={j.id} className="hover:bg-[var(--bg-hover)] transition-colors">
                      <td className="px-3 py-2.5 mono font-bold text-[var(--accent)]">#{j.id}</td>
                      <td className="px-3 py-2.5 font-medium text-[var(--text-primary)] max-w-[180px] truncate">
                        {j.original_filename}
                      </td>
                      <td className="px-3 py-2.5 text-[var(--text-muted)] whitespace-nowrap">
                        {formatTime(j.created_at)}
                      </td>
                      <td className="px-3 py-2.5 mono">{j.duration_sec ? `${j.duration_sec.toFixed(1)}s` : "—"}</td>
                      <td className="px-3 py-2.5">
                        <span
                          className={cn(
                            "rounded px-2 py-0.5 text-[10px] font-bold uppercase",
                            j.status === "completed"
                              ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                              : j.status === "processing"
                              ? "bg-[var(--warning)]/15 text-[var(--warning)]"
                              : j.status === "failed"
                              ? "bg-[var(--danger)]/15 text-[var(--danger)]"
                              : "bg-[var(--bg-elevated)] text-[var(--text-muted)]"
                          )}
                        >
                          {j.status}
                        </span>
                      </td>
                      <td className="px-3 py-2.5 mono">{j.persons_detected}</td>
                      <td className="px-3 py-2.5 mono font-bold text-[var(--danger)]">{j.events_count}</td>
                      <td className="px-3 py-2.5 whitespace-nowrap">
                        {j.status === "completed" ? (
                          <span className={cn("text-[10px] font-bold uppercase", (j.events_count ?? 0) > 0 ? "text-[var(--danger)]" : "text-[var(--text-secondary)]")}>
                            {(j.events_count ?? 0) > 0 ? "LITTERING EVENT" : "NO EVENT"}
                          </span>
                        ) : (
                          <span className="text-[10px] uppercase text-[var(--text-muted)]">{j.status}</span>
                        )}
                      </td>
                      <td className="px-3 py-2.5">
                        <div className="flex items-center gap-1.5">
                          <Link
                            to={`/analysis/${j.id}`}
                            className="rounded bg-[var(--accent)] px-2.5 py-1 text-[11px] font-bold text-black hover:bg-[var(--accent-dim)] transition-colors whitespace-nowrap"
                          >
                            VIEW RESULT
                          </Link>
                          <button
                            onClick={() => setActiveJobId(j.id)}
                            className="rounded bg-[var(--bg-elevated)] px-2.5 py-1 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--border-default)] transition-colors"
                            title="Show in the active panel above (does not re-run analysis)"
                          >
                            Inspect
                          </button>
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
