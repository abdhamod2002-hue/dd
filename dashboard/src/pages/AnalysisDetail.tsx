import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CalendarClock,
  Clock,
  CloudDownload,
  Gauge,
  User,
  Users,
  ShieldAlert,
  StopCircle,
  Trash2,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
} from "lucide-react";
import {
  analyzedVideoUrl,
  evidenceFileUrl,
  getAnalysisJob,
  getAnalysisJobEvents,
  getAnalysisManifest,
  getEvidence,
  originalVideoUrl,
  stopAnalysisJob,
  deleteAnalysisJob,
} from "../lib/api";
import { formatDate } from "../lib/utils";
import { ForensicAssetPanel } from "../components/ForensicAssetPanel";
import { SequenceStrip, buildSequenceSteps } from "../components/SequenceStrip";
import { DebugReviewPanel } from "../components/DebugReviewPanel";
import { PipelineStageMonitor } from "../components/PipelineStageMonitor";
import { selectDetectorViolation } from "../lib/detectorEvent";
import type { AnalysisManifest, Evidence, Event, VideoAnalysisJob } from "../types";

function fmtBytes(b: number | null | undefined): string {
  if (b == null) return "—";
  const mb = b / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(b / 1024).toFixed(1)} KB`;
}

export function AnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  const jobId = Number(id);
  const navigate = useNavigate();

  const analyzedRef = useRef<HTMLVideoElement>(null);
  const [job, setJob] = useState<VideoAnalysisJob | null>(null);
  const [manifest, setManifest] = useState<AnalysisManifest | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [evidence, setEvidence] = useState<Record<number, Evidence[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [stopping, setStopping] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionNotice, setActionNotice] = useState<string | null>(null);
  const [pollTick, setPollTick] = useState(0);
  const [showTechnicalReview, setShowTechnicalReview] = useState(false);

  const fetchJobData = async (isInitial = false) => {
    if (isInitial) setLoading(true);
    try {
      const j = await getAnalysisJob(jobId);
      setJob(j);

      let m: AnalysisManifest | null = null;
      try {
        m = await getAnalysisManifest(jobId);
      } catch {
        m = null;
      }
      setManifest(m);

      const evs = await getAnalysisJobEvents(jobId).catch(() => []);
      setEvents(evs ?? []);

      const acc: Record<number, Evidence[]> = {};
      await Promise.all(
        (evs ?? []).map(async (e) => {
          const list = await getEvidence(e.id).catch(() => [] as Evidence[]);
          acc[e.id] = list;
        })
      );
      setEvidence(acc);
      setError(null);
    } catch (e) {
      if (isInitial) setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (isInitial) setLoading(false);
    }
  };

  useEffect(() => {
    fetchJobData(true);
  }, [jobId, pollTick]);

  // Polling if job is processing
  useEffect(() => {
    if (!job || (job.status !== "processing" && job.status !== "queued")) return;
    const timer = setInterval(() => {
      fetchJobData(false);
    }, 2500);
    return () => clearInterval(timer);
  }, [job?.status, jobId]);

  const handleStopAnalysis = async () => {
    if (!job) return;
    setStopping(true);
    setActionError(null);
    setActionNotice(null);
    try {
      const res = await stopAnalysisJob(job.id);
      setActionNotice(res.message);
      setPollTick((t) => t + 1);
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setStopping(false);
    }
  };

  const handleDeleteJob = async () => {
    if (!job) return;
    const ok = window.confirm(
      `Are you sure you want to permanently delete Analysis Job #${job.id}? This cannot be undone.`
    );
    if (!ok) return;

    setDeleting(true);
    setActionError(null);
    setActionNotice(null);
    try {
      await deleteAnalysisJob(job.id, true);
      navigate("/analysis");
    } catch (e) {
      setActionError(e instanceof Error ? e.message : String(e));
      setDeleting(false);
    }
  };

  if (loading) {
    return (
      <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
        <div className="panel p-10 text-center text-sm text-[var(--text-muted)]">
          Loading historical analysis #{jobId}…
        </div>
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
        <div className="panel p-10 text-center">
          <div className="text-xs text-[var(--danger)]">{error || `Analysis #${jobId} not found`}</div>
          <Link to="/analysis" className="mt-3 inline-block text-xs font-semibold text-[var(--accent)] hover:underline">
            ← Back to Analysis Archive
          </Link>
        </div>
      </div>
    );
  }

  const hasEvent = events.length > 0;
  const metadata = manifest?.metadata;
  const sizes = manifest?.sizes_bytes;
  const markers = manifest?.markers ?? [];
  const firstEventMarker =
    markers.find((m) => m.label === "EVENT" || m.kind === "event") ?? null;

  let parsedReport: any = null;
  if (job?.report_json) {
    try {
      parsedReport = JSON.parse(job.report_json);
    } catch {}
  }
  const violations = parsedReport?.event_detector?.confirmed_violations ?? [];

  const focusEvent = (seekTimeSec?: number) => {
    const video = analyzedRef.current;
    if (!video) return;
    const target = seekTimeSec ?? (firstEventMarker ? firstEventMarker.timestamp : 0);
    video.currentTime = Math.max(0, target);
    video.play().catch(() => undefined);
  };

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <Link to="/analysis" className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-400 hover:text-emerald-400 transition-colors">
            <ArrowLeft className="h-3.5 w-3.5" /> Back to Analysis Archive
          </Link>
          <h1 className="mt-2 text-xl font-bold text-slate-100 flex items-center gap-2.5">
            Analysis #{job.id}
            {hasEvent ? (
              <span className="text-rose-400 text-sm font-semibold rounded-full bg-rose-500/15 border border-rose-500/30 px-2.5 py-0.5">
                VIOLATION CONFIRMED
              </span>
            ) : job.status === "cancelled" ? (
              <span className="text-amber-400 text-sm font-semibold rounded-full bg-amber-500/15 border border-amber-500/30 px-2.5 py-0.5">
                CANCELLED BY USER
              </span>
            ) : (
              <span className="text-slate-400 text-sm font-normal">— No confirmed violation</span>
            )}
          </h1>
          <p className="mt-1 text-xs text-slate-400">
            {job.original_filename || job.filename} · {formatDate(job.created_at)}
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {(job.status === "processing" || job.status === "queued") && (
            <button
              disabled={stopping}
              onClick={handleStopAnalysis}
              className="flex items-center gap-1.5 rounded-lg bg-rose-600/90 hover:bg-rose-500 text-white px-3.5 py-2 text-xs font-bold shadow transition-colors disabled:opacity-50"
            >
              <StopCircle className="h-4 w-4" />
              {stopping ? "Stopping..." : "STOP ANALYSIS"}
            </button>
          )}

          <button
            disabled={job.status === "processing" || job.status === "queued" || deleting}
            onClick={handleDeleteJob}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 hover:bg-rose-950/40 hover:border-rose-500/50 hover:text-rose-300 text-slate-300 px-3 py-2 text-xs font-semibold transition-colors disabled:opacity-40"
          >
            <Trash2 className="h-4 w-4" />
            {deleting ? "Deleting..." : "DELETE JOB"}
          </button>
        </div>
      </div>

      {actionNotice && (
        <div className="rounded-xl bg-emerald-500/10 border border-emerald-500/30 p-3 text-xs text-emerald-300">
          {actionNotice}
        </div>
      )}

      {actionError && (
        <div className="rounded-xl bg-rose-500/10 border border-rose-500/30 p-3 text-xs text-rose-300">
          {actionError}
        </div>
      )}

      {/* EXECUTIVE COMMITTEE BANNER & SUMMARY KPI STRIP */}
      {job.status === "completed" && (
        <div className="space-y-4">
          {events.length > 0 ? (
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
                    {events.length === 1 ? "1 Confirmed Violation" : `${events.length} Confirmed Violations`}
                  </span>
                </div>
              </div>

              {/* Committee Executive Summary KPI Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-2">
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">Incident Time</span>
                  <span className="mono text-sm font-bold text-slate-100">
                    {formatDate(events[0].timestamp)}
                  </span>
                </div>
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">Authoritative Actor</span>
                  <span className="mono text-sm font-bold text-cyan-300">
                    {events[0].event_actor_person_uid != null ? `Person UID #${events[0].event_actor_person_uid}` : `Person Track #${events[0].person_track_id}`}
                  </span>
                </div>
                <div className="rounded-xl bg-slate-950/80 p-3 border border-slate-800/80">
                  <span className="text-[10px] uppercase tracking-wider text-slate-400 block font-semibold">Waste Discarded</span>
                  <span className="mono text-sm font-bold text-amber-300">
                    {events[0].object_type || "Garbage Bag"}
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
                    {Math.round((events[0].confidence || 0.9) * 100)}% Verified
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

      {/* PRIMARY: event-focused forensic results */}
      {events.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-rose-500" />
              Primary Confirmed Incident Forensic Dossier
            </h2>
            <Link
              to={`/violations/${events[0].id}`}
              className="inline-flex items-center gap-1 text-xs font-semibold text-emerald-400 hover:text-emerald-300"
            >
              Open Incident File #{events[0].id}
            </Link>
          </div>

          {events.map((ev) => {
            const evList = evidence[ev.id] ?? [];
            const current = evList[0];
            const detectorViolation = selectDetectorViolation(violations, ev);
            const steps = buildSequenceSteps(current, detectorViolation);

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
                    Timestamp: <span className="mono text-slate-200">{formatDate(ev.timestamp)}</span>
                  </div>
                </div>

                <ForensicAssetPanel
                  event={ev}
                  evidence={current}
                  detectorViolation={detectorViolation}
                  markers={markers}
                  onSeekVideo={focusEvent}
                />

                {steps.length > 0 && (
                  <div className="panel border-slate-800 bg-slate-950/70 p-4">
                    <div className="mb-2 text-[11px] font-bold uppercase tracking-wider text-slate-400">
                      Multi-Stage Temporal Evidence Sequence (Carry → Release → Ground → Departure)
                    </div>
                    <SequenceStrip
                      steps={steps}
                      onSeekVideo={focusEvent}
                      actorUid={ev.event_actor_person_uid}
                      objectUid={ev.event_object_uid}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* SECONDARY TECHNICAL REVIEW & PIPELINE TELEMETRY (COLLAPSED BY DEFAULT) */}
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
              (12-Stage Pipeline Monitor, Tracking Demarcation, Full Video Overlays)
            </span>
          </div>
          <span className="mono text-[11px] text-slate-400">
            {showTechnicalReview ? "Hide Details" : "Show Technical Review"}
          </span>
        </button>

        {(showTechnicalReview || job.status === "processing" || job.status === "queued") && (
          <div className="p-5 space-y-6 border-t border-slate-800 bg-slate-950/50">
            {/* 12-Stage Pipeline Monitor & Telemetry */}
            <PipelineStageMonitor job={job} report={parsedReport} />

            {/* Job meta strip */}
            <div className="panel border-slate-800 bg-slate-900/90 p-4 shadow-xl">
              <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 lg:grid-cols-6">
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                  <div className="mono font-bold text-slate-200 uppercase">{job.status}</div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Status</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                  <div className="mono font-bold text-slate-200 flex items-center gap-1">
                    <Clock className="h-3 w-3 text-emerald-400" /> {job.duration_sec != null ? `${job.duration_sec.toFixed(1)}s` : "—"}
                  </div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Duration</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                  <div className="mono font-bold text-emerald-400 flex items-center gap-1">
                    <User className="h-3 w-3" /> {job.unique_persons_count ?? job.persons_detected ?? 0}
                  </div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Unique People</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                  <div className="mono font-bold text-amber-400 flex items-center gap-1">
                    <Users className="h-3 w-3" /> {job.total_person_track_ids ?? job.persons_detected ?? 0}
                  </div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Track IDs</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                  <div className="mono font-bold text-rose-400 flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" /> {events.length}
                  </div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Confirmed Events</div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-slate-950 p-2.5">
                  <div className="mono font-bold text-slate-200 flex items-center gap-1">
                    <Gauge className="h-3 w-3 text-emerald-400" /> {job.processing_fps != null ? job.processing_fps.toFixed(1) : "—"}
                  </div>
                  <div className="text-[10px] text-slate-400 uppercase tracking-wider">Proc FPS</div>
                </div>
              </div>
              {(sizes || metadata) && (
                <div className="mt-3 flex flex-wrap gap-3 text-xs text-slate-400">
                  {sizes?.original != null && (
                    <span className="inline-flex items-center gap-1">
                      <CloudDownload className="h-3 w-3 text-slate-400" /> Original {fmtBytes(sizes.original)}
                    </span>
                  )}
                  {sizes?.analyzed != null && <span>Analyzed {fmtBytes(sizes.analyzed)}</span>}
                  <span className="inline-flex items-center gap-1">
                    <CalendarClock className="h-3 w-3 text-slate-400" /> {formatDate(job.completed_at ?? job.created_at)}
                  </span>
                </div>
              )}
            </div>

            {/* Debug Review Player */}
            <DebugReviewPanel
              originalVideoUrl={originalVideoUrl(job.id)}
              analyzedVideoUrl={analyzedVideoUrl(job.id)}
              analyzedVideoRef={analyzedRef}
              clipUrl={
                events[0]
                  ? (() => {
                      const clip = (evidence[events[0].id] ?? [])[0]?.clip_path;
                      return clip ? evidenceFileUrl(clip) : undefined;
                    })()
                  : undefined
              }
              markers={manifest?.markers ?? []}
              durationSec={metadata?.duration_sec ?? job.duration_sec}
              onFocusEvent={firstEventMarker ? focusEvent : undefined}
              hasEventMarker={!!firstEventMarker}
              defaultOpen={false}
            />
          </div>
        )}
      </div>
    </div>
  );
}
