import { useRef } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  Camera,
  Clock,
  User,
  Package,
  Gauge,
  CheckCircle2,
  ShieldAlert,
  Play,
  Crosshair,
  FileVideo,
} from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { analyzedVideoUrl, evidenceFileUrl, getEventReview, originalVideoUrl } from "../lib/api";
import { Badge } from "../components/Badge";
import { TimelineMarkers } from "../components/TimelineMarkers";
import { cn, formatDate, formatConfidence } from "../lib/utils";
import type { AnalysisMarker, Event } from "../types";

/**
 * One entry of `report.event_detector.confirmed_violations` (the compact
 * detector event produced by `_compact_detector_event` in
 * backend/routers/analysis.py).
 */
interface DetectorViolation {
  person_track_id?: number | string | null;
  bag_track_id?: number | string | null;
  event_actor_person_track_id?: number | null;
  event_actor_person_uid?: number | null;
  event_object_track_id?: number | null;
  event_object_uid?: number | null;
  frames?: Record<string, unknown> | null;
  evidence?: Record<string, number> | null;
  [key: string]: unknown;
}

/**
 * Select the detector violation that belongs to the event being viewed.
 *
 * Never a fixed index: every Event row sharing one job report used to read
 * `confirmed_violations[0]`, so with 2+ events per job each event displayed
 * another event's detector state (MASTER_REPAIR_PLAN P0-2).
 *
 * Match priority:
 *   1. frozen stable UIDs (authoritative actor/object identity),
 *   2. frozen stable track ids,
 *   3. legacy raw track ids (historical rows with NULL stable identity),
 *   4. a single unambiguous candidate.
 * Returns null rather than guessing among multiple unmatched violations.
 */
function selectDetectorViolation(
  violations: DetectorViolation[] | undefined,
  event: Event | undefined,
): DetectorViolation | null {
  if (!violations?.length || !event) return null;
  if (event.event_actor_person_uid != null && event.event_object_uid != null) {
    const byUid = violations.find(
      (v) =>
        v.event_actor_person_uid === event.event_actor_person_uid &&
        v.event_object_uid === event.event_object_uid,
    );
    if (byUid) return byUid;
  }
  if (event.event_actor_person_track_id != null && event.event_object_track_id != null) {
    const byStableTrack = violations.find(
      (v) =>
        v.event_actor_person_track_id === event.event_actor_person_track_id &&
        v.event_object_track_id === event.event_object_track_id,
    );
    if (byStableTrack) return byStableTrack;
  }
  const personId = event.person_track_id != null ? Number(event.person_track_id) : NaN;
  const objectId = event.object_track_id != null ? Number(event.object_track_id) : NaN;
  if (!Number.isNaN(personId) && !Number.isNaN(objectId)) {
    const byLegacy = violations.find(
      (v) => Number(v.person_track_id) === personId && Number(v.bag_track_id) === objectId,
    );
    if (byLegacy) return byLegacy;
  }
  return violations.length === 1 ? violations[0] : null;
}

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
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
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

      <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
        <div className="space-y-6">
          {/* PRIMARY EVENT EVIDENCE — event clip + actor/object (P1-3) */}
          <div className="panel space-y-4 border-[var(--danger)]/30 p-5">
            <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--danger)]">
              <ShieldAlert className="h-4 w-4" /> Event Evidence — Actor + Object
            </h2>

            <div className="grid grid-cols-2 gap-3 text-[11px] sm:grid-cols-3">
              <div className="rounded bg-[var(--bg-base)] p-2">
                <div className="text-[10px] uppercase text-[var(--text-muted)]">Actor</div>
                <div className="mono font-bold text-[var(--text-primary)]">
                  {event.event_actor_person_uid != null
                    ? `PERSON UID #${event.event_actor_person_uid}`
                    : event.event_actor_person_track_id != null
                      ? `PERSON track #${event.event_actor_person_track_id}`
                      : "—"}
                  {event.event_actor_person_track_id != null && (
                    <span className="ml-1 text-[10px] font-normal text-[var(--text-muted)]">
                      track {event.event_actor_person_track_id}
                    </span>
                  )}
                </div>
              </div>
              <div className="rounded bg-[var(--bg-base)] p-2">
                <div className="text-[10px] uppercase text-[var(--text-muted)]">Object</div>
                <div className="mono font-bold text-[var(--text-primary)]">
                  {event.event_object_uid != null
                    ? `WASTE UID #${event.event_object_uid}`
                    : event.event_object_track_id != null
                      ? `WASTE track #${event.event_object_track_id}`
                      : event.object_type}
                  {event.event_object_track_id != null && (
                    <span className="ml-1 text-[10px] font-normal text-[var(--text-muted)]">
                      track {event.event_object_track_id}
                    </span>
                  )}
                </div>
              </div>
              <div className="rounded bg-[var(--bg-base)] p-2">
                <div className="text-[10px] uppercase text-[var(--text-muted)]">Confidence</div>
                <div className="mono font-bold text-[var(--text-primary)]">{formatConfidence(event.confidence)}</div>
              </div>
            </div>

            {currentEvidence?.clip_path ? (
              <div className="space-y-2">
                <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--accent)]">
                  Event Clip — carry to departure (primary evidence)
                </div>
                <video controls className="w-full rounded-lg border-2 border-[var(--accent)]/50 bg-black object-contain" src={evidenceFileUrl(currentEvidence.clip_path)} />
              </div>
            ) : (
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-4 text-center text-xs text-[var(--text-muted)]">
                Event clip was not captured for this event.
              </div>
            )}

            <div className="grid gap-3 sm:grid-cols-3">
              {currentEvidence?.image_path && (
                <figure className="rounded-lg border-2 border-[var(--danger)]/30 bg-[var(--bg-base)] p-2">
                  <img src={evidenceFileUrl(currentEvidence.image_path)} alt="Event snapshot" className="h-44 w-full rounded object-contain" />
                  <figcaption className="mt-1 text-[10px] font-bold uppercase text-[var(--danger)]">Event Snapshot — Actor + Object</figcaption>
                </figure>
              )}
              {/* P1-2: temporal sequence stills (carry -> release -> ground) */}
              {currentEvidence?.carry_image_path && (
                <figure className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <img src={evidenceFileUrl(currentEvidence.carry_image_path)} alt="Carry moment" className="h-44 w-full rounded object-contain" />
                  <figcaption className="mt-1 text-[10px] uppercase text-[var(--accent)]">1 — Carry</figcaption>
                </figure>
              )}
              {currentEvidence?.release_image_path && (
                <figure className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <img src={evidenceFileUrl(currentEvidence.release_image_path)} alt="Release moment" className="h-44 w-full rounded object-contain" />
                  <figcaption className="mt-1 text-[10px] uppercase text-[var(--accent)]">2 — Release</figcaption>
                </figure>
              )}
              {currentEvidence?.ground_image_path && (
                <figure className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <img src={evidenceFileUrl(currentEvidence.ground_image_path)} alt="Ground moment" className="h-44 w-full rounded object-contain" />
                  <figcaption className="mt-1 text-[10px] uppercase text-[var(--accent)]">3 — Ground</figcaption>
                </figure>
              )}
              {currentEvidence?.person_image_path && (
                <figure className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <img src={evidenceFileUrl(currentEvidence.person_image_path)} alt="Person evidence" className="h-44 w-full rounded object-contain" />
                  <figcaption className="mt-1 text-[10px] uppercase text-[var(--text-muted)]">
                    Person — stable UID #{event.event_actor_person_uid ?? event.event_actor_person_track_id ?? "—"}
                  </figcaption>
                </figure>
              )}
              {currentEvidence?.waste_image_path && (
                <figure className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <img src={evidenceFileUrl(currentEvidence.waste_image_path)} alt="Waste evidence" className="h-44 w-full rounded object-contain" />
                  <figcaption className="mt-1 text-[10px] uppercase text-[var(--text-muted)]">
                    Waste — stable UID #{event.event_object_uid ?? event.event_object_track_id ?? "—"}
                  </figcaption>
                </figure>
              )}
              {currentEvidence?.face_image_path && (
                <figure className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <img src={evidenceFileUrl(currentEvidence.face_image_path)} alt="Face evidence (sensitive — human review required)" className="h-44 w-full rounded object-contain" />
                  <figcaption className="mt-1 flex items-center justify-between gap-1 text-[10px] uppercase text-[var(--text-muted)]">
                    <span>Face (sensitive)</span>
                    <span className="rounded bg-[var(--warning)]/20 px-1.5 py-0.5 text-[9px] font-bold text-[var(--warning)]">HUMAN REVIEW</span>
                  </figcaption>
                </figure>
              )}
            </div>

            {currentEvidence?.video_path && (
              <div className="space-y-2">
                <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Evidence clip</div>
                <video controls className="w-full rounded-lg border border-[var(--border-subtle)] bg-black object-contain" src={evidenceFileUrl(currentEvidence.video_path)} />
              </div>
            )}
          </div>

          {/* SECONDARY — full analyzed video, technical/debug review only (P1-3) */}
          <div className="panel space-y-4 p-5">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
                <FileVideo className="h-4 w-4" /> Technical Review — Full Video (Debug)
              </h2>
              <div className="flex items-center gap-2">
                {eventMarker && (
                  <button onClick={focusEvent} className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--danger)]/40 bg-[var(--danger)]/10 px-3 py-1.5 text-[11px] font-bold uppercase text-[var(--danger)] hover:bg-[var(--danger)]/20">
                    <Crosshair className="h-3.5 w-3.5" /> View Event
                  </button>
                )}
                {currentEvidence?.clip_path && (
                  <a href={evidenceFileUrl(currentEvidence.clip_path)} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] px-3 py-1.5 text-[11px] font-bold uppercase text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
                    <Play className="h-3.5 w-3.5" /> Event Clip
                  </a>
                )}
              </div>
            </div>

            {job?.analyzed_video_path ? (
              <>
                <div className="grid gap-4 lg:grid-cols-2">
                  <div className="space-y-2">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Original video</div>
                    <video controls className="w-full rounded-lg border border-[var(--border-subtle)] bg-black object-contain" src={originalVideoUrl(job.id)} />
                  </div>
                  <div className="space-y-2">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Detection overlays — debug</div>
                    <video ref={analyzedRef} controls className="w-full rounded-lg border border-[var(--border-subtle)] bg-black object-contain" src={analyzedVideoUrl(job.id)} />
                  </div>
                </div>
                {markers.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Timeline — click to jump</div>
                    <TimelineMarkers markers={markers} durationSec={job.duration_sec} videoRef={analyzedRef} />
                  </div>
                )}
              </>
            ) : (
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-6 text-center text-xs text-[var(--text-muted)]">
                No analyzed video is linked to this event.
              </div>
            )}
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-4">
          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">Tracking</h2>
            <div className="space-y-2.5">
              <Field icon={Camera} label="Camera" value={job?.original_filename ? `Video: ${job.original_filename}` : `CAM-${event.camera_id}`} />
              <Field icon={Clock} label="Time" value={formatDate(event.timestamp)} />
              {/* P1-12: stable identity first — raw person_track_id churns across
                  tracker ID switches, the frozen UID/track does not. */}
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
            {Object.keys(evidenceScores).length > 0 && (
              <div className="mt-3 space-y-1 text-[11px] mono text-[var(--text-muted)]">
                {Object.entries(evidenceScores).map(([k, v]: [string, any]) => (
                  <div key={k} className="flex justify-between">
                    <span>{k}</span>
                    <span>{Number(v).toFixed(2)}</span>
                  </div>
                ))}
              </div>
            )}
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
