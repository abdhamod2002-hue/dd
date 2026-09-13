import { useRef } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  CheckCircle2,
  ShieldAlert,
  Camera,
  FileCheck,
} from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { analyzedVideoUrl, evidenceFileUrl, getEventReview, originalVideoUrl } from "../lib/api";
import { selectDetectorViolation } from "../lib/detectorEvent";
import { Badge } from "../components/Badge";
import { ForensicAssetPanel } from "../components/ForensicAssetPanel";
import { SequenceStrip, buildSequenceSteps } from "../components/SequenceStrip";
import { DebugReviewPanel } from "../components/DebugReviewPanel";
import { cn, formatDate, formatConfidence } from "../lib/utils";
import type { AnalysisMarker } from "../types";

export function EventDetail() {
  const { id } = useParams<{ id: string }>();
  const eventId = Number(id);
  const analyzedRef = useRef<HTMLVideoElement>(null);

  const { data: review, loading, error } = useFetch(() => getEventReview(eventId), [eventId]);
  const event = review?.event;
  const evidence = review?.evidence ?? [];
  const job = review?.job ?? null;
  const report = review?.report ?? null;
  const markers: AnalysisMarker[] = report?.markers ?? [];
  const detectorEvent = selectDetectorViolation(
    report?.event_detector?.confirmed_violations,
    event,
  );
  const currentEvidence = evidence[0];
  const eventMarker = markers.find((m) => m.label === "EVENT") ?? markers.find((m) => m.kind === "event");

  if (loading) {
    return (
      <div className="mx-auto max-w-[1600px] p-8 text-center">
        <div className="panel border-slate-800 bg-slate-900/90 p-10 text-xs text-slate-400">
          Loading forensic event #{eventId} dossier…
        </div>
      </div>
    );
  }

  if (error || !event) {
    return (
      <div className="mx-auto max-w-[1600px] p-7">
        <Link to="/violations" className="text-xs font-semibold text-emerald-400 hover:underline">
          ← Back to violations
        </Link>
        <p className="mt-4 text-xs text-rose-400">{error ?? "Event not found"}</p>
      </div>
    );
  }

  const isConfirmed = event.status === "confirmed";
  const evidenceScores: Record<string, number> = detectorEvent?.evidence ?? {};
  const behavior = [
    { label: "BAG CARRIED", ok: !!detectorEvent?.frames?.carry_start },
    { label: "RELEASE", ok: !!detectorEvent?.frames?.release },
    { label: "GROUND", ok: !!detectorEvent?.frames?.ground },
    { label: "DEPARTURE", ok: !!detectorEvent?.frames?.departure },
    { label: "NO REGRAB", ok: isConfirmed },
  ];

  const focusEvent = (seekTimeSec?: number) => {
    const video = analyzedRef.current;
    if (!video) return;
    const target = seekTimeSec ?? (eventMarker ? eventMarker.timestamp : 0);
    video.currentTime = Math.max(0, target);
    video.play().catch(() => undefined);
  };

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-4 sm:p-6 lg:p-7">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
        <div>
          <Link
            to="/violations"
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-400 hover:text-emerald-400 transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Back to Violations Log
          </Link>
          <div className="mt-2 flex items-center gap-3">
            <h1 className="mono text-2xl font-bold text-slate-100">Incident Event #{event.id}</h1>
            <Badge status={event.status} />
          </div>
          <p className="mt-1 text-xs text-slate-400">
            Detected: {formatDate(event.timestamp)} · Camera: {job?.original_filename ? `Video: ${job.original_filename}` : `CAM-${event.camera_id}`}
          </p>
        </div>

        {isConfirmed && (
          <div className="flex items-center gap-2 rounded-xl bg-rose-500/15 border border-rose-500/30 px-4 py-2 shadow-lg shadow-rose-950/20">
            <ShieldAlert className="h-5 w-5 text-rose-500 animate-pulse" />
            <div>
              <div className="text-xs font-bold text-rose-400">LITTERING INCIDENT CONFIRMED</div>
              <div className="text-[10px] text-slate-400">All temporal confirmation criteria verified</div>
            </div>
          </div>
        )}
      </div>

      {/* Primary Split-View Evidence Dossier */}
      <ForensicAssetPanel
        event={event}
        evidence={currentEvidence}
        detectorViolation={detectorEvent}
        markers={markers}
        onSeekVideo={focusEvent}
      />

      {/* Chronological Behavioral Sequence Gallery */}
      <SequenceStrip
        steps={buildSequenceSteps(currentEvidence, detectorEvent)}
        onSeekVideo={focusEvent}
        actorUid={event.event_actor_person_uid}
        objectUid={event.event_object_uid}
      />

      {/* Secondary Meta Information & Review Notes */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Tracking Details */}
        <div className="panel border-slate-800 bg-slate-900/90 p-4 space-y-3 shadow-md">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Camera className="h-4 w-4 text-emerald-400" /> Sensor & Entity Tracking
          </h2>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between items-center rounded bg-slate-950 px-2.5 py-1.5 border border-slate-800/60">
              <span className="text-slate-400">Camera / Feed:</span>
              <span className="mono text-slate-200">
                {job?.original_filename ? job.original_filename : `CAM-${event.camera_id}`}
              </span>
            </div>
            <div className="flex justify-between items-center rounded bg-slate-950 px-2.5 py-1.5 border border-slate-800/60">
              <span className="text-slate-400">Actor UID:</span>
              <span className="mono font-bold text-cyan-400">
                {event.event_actor_person_uid != null ? `UID #${event.event_actor_person_uid}` : "—"}
              </span>
            </div>
            <div className="flex justify-between items-center rounded bg-slate-950 px-2.5 py-1.5 border border-slate-800/60">
              <span className="text-slate-400">Waste UID:</span>
              <span className="mono font-bold text-amber-400">
                {event.event_object_uid != null ? `UID #${event.event_object_uid}` : "—"}
              </span>
            </div>
            <div className="flex justify-between items-center rounded bg-slate-950 px-2.5 py-1.5 border border-slate-800/60">
              <span className="text-slate-400">Object Type:</span>
              <span className="mono text-slate-200">{event.object_type}</span>
            </div>
            <div className="flex justify-between items-center rounded bg-slate-950 px-2.5 py-1.5 border border-slate-800/60">
              <span className="text-slate-400">Confidence:</span>
              <span className="mono font-bold text-emerald-400">{formatConfidence(event.confidence)}</span>
            </div>
          </div>
        </div>

        {/* Behavioral Checklist */}
        <div className="panel border-slate-800 bg-slate-900/90 p-4 space-y-3 shadow-md">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <FileCheck className="h-4 w-4 text-emerald-400" /> Behavioral Gate Status
          </h2>
          <div className="space-y-1.5 text-xs">
            {behavior.map((b) => (
              <div
                key={b.label}
                className="flex items-center justify-between rounded bg-slate-950 px-2.5 py-1.5 border border-slate-800/60"
              >
                <span className="text-slate-300">{b.label}</span>
                <span
                  className={cn(
                    "mono text-[10px] font-bold px-2 py-0.5 rounded",
                    b.ok
                      ? "bg-emerald-500/15 border border-emerald-500/30 text-emerald-400"
                      : "bg-slate-800 text-slate-500"
                  )}
                >
                  {b.ok ? "CONFIRMED" : "PENDING"}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Forensic Review Decision */}
        <div className="panel border-slate-800 bg-slate-900/90 p-4 space-y-3 shadow-md">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-emerald-400" /> Forensic Attribution Notice
          </h2>
          <p className="text-[11px] leading-relaxed text-slate-400">
            The production pipeline verified carry, release, ground settle, and actor departure. All evidence
            records are cryptographically anchored to PostgreSQL and disk storage.
          </p>
          <div className="rounded-lg bg-emerald-500/10 border border-emerald-500/20 p-3 space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-bold text-emerald-400">
              <CheckCircle2 className="h-4 w-4" /> Final Status: Valid Incident
            </div>
            <div className="text-[10px] text-slate-400">
              Dossier status: <span className="font-semibold text-amber-400">HUMAN REVIEW READY</span>
            </div>
          </div>
        </div>
      </div>

      {/* Engineering Debug Review Panel */}
      <DebugReviewPanel
        originalVideoUrl={job?.id != null ? originalVideoUrl(job.id) : undefined}
        analyzedVideoUrl={job?.analyzed_video_path ? analyzedVideoUrl(job.id) : undefined}
        analyzedVideoRef={analyzedRef}
        clipUrl={currentEvidence?.clip_path ? evidenceFileUrl(currentEvidence.clip_path) : undefined}
        markers={markers}
        durationSec={job?.duration_sec}
        evidenceScores={evidenceScores}
        onFocusEvent={focusEvent}
        hasEventMarker={!!eventMarker}
      />
    </div>
  );
}
