import { useRef } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Clock, User, Package, Gauge, CheckCircle2, ShieldAlert, Camera } from "lucide-react";
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
    return <div className="p-7 text-[13px] text-[var(--text-muted)]">Loading event…</div>;
  }
  if (error || !event) {
    return (
      <div className="p-7">
        <Link to="/violations" className="text-[13px] font-semibold text-[var(--accent)] hover:underline">← Back to violations</Link>
        <p className="mt-4 text-[13px] text-[var(--danger)]">{error ?? "Event not found"}</p>
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

  const focusEvent = () => {
    const video = analyzedRef.current;
    if (!video || !eventMarker) return;
    video.currentTime = Math.max(0, eventMarker.timestamp);
    video.play().catch(() => undefined);
  };

  return (
    <div className="mx-auto max-w-[1600px] space-y-5 p-5 lg:p-7">
      <Link to="/violations" className="inline-flex items-center gap-1.5 text-[13px] font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
        <ArrowLeft className="h-4 w-4" /> Back to violations
      </Link>

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="mono text-2xl font-bold text-[var(--text-primary)]">Event #{event.id}</h1>
            <Badge status={event.status} />
          </div>
          <p className="mt-1 text-[13px] text-[var(--text-secondary)]">
            {formatDate(event.timestamp)}
          </p>
        </div>
        {isConfirmed && (
          <div className="flex items-center gap-2 rounded-lg bg-[var(--danger)]/15 px-4 py-2.5">
            <ShieldAlert className="h-5 w-5 text-[var(--danger)]" />
            <span className="text-[13px] font-bold text-[var(--danger)]">LITTERING EVENT CANDIDATE</span>
          </div>
        )}
      </div>

      <div className="grid gap-5 xl:grid-cols-[1fr_360px]">
        <div className="space-y-5">
          <ForensicAssetPanel event={event} evidence={currentEvidence} />

          <SequenceStrip steps={buildSequenceSteps(currentEvidence, detectorEvent)} />

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

        {/* Right column */}
        <div className="space-y-4">
          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">Tracking</h2>
            <div className="space-y-2.5">
              <Field icon={Camera} label="Camera" value={job?.original_filename ? `Video: ${job.original_filename}` : `CAM-${event.camera_id}`} />
              <Field icon={Clock} label="Time" value={formatDate(event.timestamp)} />
              <Field
                icon={User}
                label="Person"
                value={
                  event.event_actor_person_uid != null
                    ? `UID #${event.event_actor_person_uid} (track ${event.event_actor_person_track_id ?? "?"})`
                    : event.event_actor_person_track_id != null
                      ? `Track #${event.event_actor_person_track_id}`
                      : event.person_track_id
                        ? `Track #${event.person_track_id} (legacy)`
                        : "—"
                }
                mono
              />
              <Field
                icon={Package}
                label="Object"
                value={
                  event.event_object_uid != null
                    ? `${event.object_type} UID #${event.event_object_uid} (track ${event.event_object_track_id ?? "?"})`
                    : event.event_object_track_id != null
                      ? `${event.object_type} track #${event.event_object_track_id}`
                      : `${event.object_type}${event.object_track_id ? ` #${event.object_track_id}` : ""}`
                }
              />
              <Field icon={Gauge} label="Confidence" value={formatConfidence(event.confidence)} mono />
            </div>
          </div>

          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">Behavior</h2>
            <div className="space-y-2.5">
              {behavior.map((b) => (
                <div key={b.label} className="flex items-center justify-between rounded bg-[var(--bg-base)] px-3 py-2">
                  <span className="text-[12px] font-semibold text-[var(--text-primary)]">{b.label}</span>
                  <span className={cn("text-[12px] font-bold", b.ok ? "text-[var(--accent)]" : "text-[var(--text-muted)]")}>{b.ok ? "✅" : "—"}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">Why flagged</h2>
            <p className="text-[11px] leading-relaxed text-[var(--text-secondary)]">
              The production temporal detector observed a real person track, a real waste-object track, wrist/torso association,
              carry, release, stationary ground, and departure without re-grab. This is an assistive review candidate,
              not a legal determination and not 100% accurate.
            </p>
            {isConfirmed && (
              <div className="mt-4 flex flex-col gap-2 rounded-lg bg-[var(--accent)]/10 p-3">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-[var(--accent)]" />
                  <span className="text-[12px] font-semibold text-[var(--accent)]">Final Decision: Littering Event Candidate</span>
                </div>
                <div className="text-[11px] text-[var(--text-secondary)]">
                  Status: <span className="font-semibold text-[var(--warning)]">HUMAN REVIEW REQUIRED</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ icon: Icon, label, value, mono }: { icon: typeof Camera; label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <Icon className="h-4 w-4 shrink-0 text-[var(--text-muted)]" />
      <div className="min-w-0 flex-1">
        <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
        <div className={cn("truncate text-[13px] text-[var(--text-primary)]", mono && "mono")}>{value}</div>
      </div>
    </div>
  );
}
