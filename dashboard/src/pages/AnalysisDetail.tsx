import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  AlertTriangle,
  ArrowLeft,
  CalendarClock,
  Clock,
  CloudDownload,
  FileVideo,
  Files,
  Film,
  Gauge,
  Package,
  User,
} from "lucide-react";
import {
  analyzedVideoUrl,
  evidenceFileUrl,
  getAnalysisJob,
  getAnalysisJobEvents,
  getAnalysisManifest,
  getEvidence,
  originalVideoUrl,
} from "../lib/api";
import { cn, formatDate, formatTime } from "../lib/utils";
import { TimelineMarkers } from "../components/TimelineMarkers";
import type { AnalysisManifest, Evidence, Event, VideoAnalysisJob } from "../types";

function fmtBytes(b: number | null | undefined): string {
  if (b == null) return "—";
  const mb = b / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${(b / 1024).toFixed(1)} KB`;
}

export function AnalysisDetail() {
  const { id } = useParams<{ id: string }>();
  const jobId = Number(id);

  const analyzedRef = useRef<HTMLVideoElement>(null);
  const [job, setJob] = useState<VideoAnalysisJob | null>(null);
  const [manifest, setManifest] = useState<AnalysisManifest | null>(null);
  const [events, setEvents] = useState<Event[]>([]);
  const [evidence, setEvidence] = useState<Record<number, Evidence[]>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const j = await getAnalysisJob(jobId);
        if (cancelled) return;
        setJob(j);
        let m: AnalysisManifest | null = null;
        try {
          m = await getAnalysisManifest(jobId);
        } catch {
          m = null;
        }
        if (cancelled) return;
        setManifest(m);
        const evs = await getAnalysisJobEvents(jobId).catch(() => []);
        if (cancelled) return;
        setEvents(evs ?? []);
        const acc: Record<number, Evidence[]> = {};
        await Promise.all(
          (evs ?? []).map(async (e) => {
            const list = await getEvidence(e.id).catch(() => [] as Evidence[]);
            acc[e.id] = list;
          })
        );
        if (!cancelled) setEvidence(acc);
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

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

  const hasEvent = (job.events_count ?? 0) > 0;
  const metadata = manifest?.metadata;
  const sizes = manifest?.sizes_bytes;

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link to="/analysis" className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-[var(--text-secondary)] hover:text-[var(--accent)]">
            <ArrowLeft className="h-3.5 w-3.5" /> Back to Analysis Archive
          </Link>
          <h1 className="mt-2 text-xl font-bold text-[var(--text-primary)]">
            Analysis #{job.id}
            <span className="ml-3 text-sm font-normal text-[var(--text-secondary)]">{job.original_filename}</span>
          </h1>
          <p className="mt-0.5 text-[12px] text-[var(--text-muted)] flex items-center gap-1.5">
            <CalendarClock className="h-3.5 w-3.5" />
            Created {formatDate(job.created_at)}
            {job.started_at && <> · Started {formatTime(job.started_at)}</>}
            {job.completed_at && <> · Completed {formatTime(job.completed_at)}</>}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn("rounded px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider",
            job.status === "completed" ? "bg-[var(--accent)]/15 text-[var(--accent)]"
            : job.status === "failed" ? "bg-[var(--danger)]/15 text-[var(--danger)]"
            : "bg-[var(--warning)]/15 text-[var(--warning)]")}>
            {job.status}
          </span>
          <span className={cn("rounded px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider",
            hasEvent ? "bg-[var(--danger)]/15 text-[var(--danger)]" : "bg-[var(--bg-elevated)] text-[var(--text-secondary)]")}>
            {hasEvent ? "LITTERING EVENT CANDIDATE" : "NO EVENT"}
          </span>
        </div>
      </div>
{/* Metadata strip */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {[
          { label: "Duration", value: job.duration_sec ? `${job.duration_sec.toFixed(1)}s` : "—", icon: Clock },
          { label: "Source FPS", value: job.fps ? `${job.fps.toFixed(1)}` : "—", icon: Gauge },
          { label: "Resolution", value: metadata?.resolution ? `${metadata.resolution[0]}×${metadata.resolution[1]}` : "—", icon: Film },
          { label: "Persons", value: String(job.persons_detected), icon: User },
          { label: "Objects", value: String(job.objects_detected), icon: Package },
          { label: "Events", value: String(job.events_count), icon: AlertTriangle },
        ].map((s) => (
          <div key={s.label} className="panel flex items-center gap-2 px-3 py-2.5">
            <s.icon className="h-4 w-4 text-[var(--accent)]" />
            <div>
              <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">{s.label}</div>
              <div className="mono text-xs font-bold text-[var(--text-primary)]">{s.value}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Storage sizes */}
      <div className="panel p-4">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-[var(--text-muted)]">
          <CloudDownload className="h-4 w-4" /> Storage per analysis
        </div>
        <div className="mt-2 grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
          <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2.5">
            <div className="mono font-bold text-[var(--text-primary)]">{fmtBytes(sizes?.original)}</div>
            <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Original</div>
          </div>
          <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2.5">
            <div className="mono font-bold text-[var(--text-primary)]">{fmtBytes(sizes?.analyzed)}</div>
            <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Analyzed</div>
          </div>
          <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2.5">
            <div className="mono font-bold text-[var(--text-primary)]">{fmtBytes(sizes?.frames_jsonl)}</div>
            <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Frames JSONL</div>
          </div>
          <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2.5">
            <div className="mono font-bold text-[var(--text-primary)]">{events.length}</div>
            <div className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider">Events</div>
          </div>
        </div>
      </div>

      {/* Videos: Original + Analyzed */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="panel p-4 space-y-2">
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">
            <FileVideo className="h-4 w-4 text-[var(--accent)]" /> Original Video
          </h2>
          <video
            key={`orig-${job.id}`}
            controls
            className="w-full rounded-lg border border-[var(--border-subtle)] bg-black"
            src={originalVideoUrl(job.id)}
          />
        </div>
        <div className="panel p-4 space-y-2">
          <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">
            <Film className="h-4 w-4 text-[var(--accent)]" /> Full AI Analyzed Video
          </h2>
          <video
            key={`an-${job.id}`}
            ref={analyzedRef}
            controls
            className="w-full rounded-lg border border-[var(--border-subtle)] bg-black"
            src={analyzedVideoUrl(job.id)}
          />
        </div>
      </div>

      {/* Behavior timeline */}
      <div className="panel p-4 space-y-2">
        <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">
          <Files className="h-4 w-4 text-[var(--accent)]" /> AI Pipeline Timeline & Markers
        </h2>
        {manifest?.timeline && manifest.timeline.length > 0 ? (
          <div className="space-y-1.5">
            {manifest.timeline.map((item: any, idx: number) => (
              <div key={idx} className="flex items-center justify-between rounded bg-[var(--bg-base)] px-3 py-2 text-[11px]">
                <span className="mono text-[var(--accent)]">{item.timestamp}s</span>
                <span className="mono font-semibold text-[var(--text-primary)]">{item.state}</span>
                {item.frame != null && <span className="mono text-[var(--text-muted)]">frame {item.frame}</span>}
              </div>
            ))}
          </div>
        ) : (
          <div className="rounded bg-[var(--bg-base)] p-4 text-center text-xs text-[var(--text-muted)]">
            No state transitions recorded for this analysis.
          </div>
        )}
        {(manifest?.markers ?? []).length > 0 && (
          <TimelineMarkers
            markers={manifest?.markers ?? []}
            durationSec={metadata?.duration_sec ?? job.duration_sec}
            videoRef={analyzedRef}
          />
        )}
      </div>

      {/* Confirmed events */}
      <div className="space-y-4">
        {events.length === 0 && (
          <div className="panel p-4 text-center text-xs text-[var(--text-secondary)]">
            No littering events were confirmed for this analysis.
          </div>
        )}
        {events.map((ev) => {
          const evs = evidence[ev.id] ?? [];
          const img = evs.find((e) => e.image_path);
          const person = evs.find((e) => e.person_image_path);
          const waste = evs.find((e) => e.waste_image_path);
          const face = evs.find((e) => e.face_image_path);
          const clip = evs.find((e) => e.clip_path) ?? evs.find((e) => e.video_path);
          return (
            <div key={ev.id} className="panel p-4 space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-[var(--danger)]" />
                  <h3 className="text-sm font-bold text-[var(--text-primary)]">
                    Event #{ev.id} — <span className="text-[var(--danger)]">LITTERING EVENT CANDIDATE</span>
                  </h3>
                </div>
                <Link to={`/violations/${ev.id}`} className="rounded bg-[var(--accent)] px-3 py-1.5 text-[11px] font-bold text-black hover:bg-[var(--accent-dim)]">
                  Open Event Detail →
                </Link>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4">
                <div className="rounded bg-[var(--bg-base)] p-2"><span className="text-[var(--text-muted)]">Confidence</span><div className="mono font-bold">{Math.round((ev.confidence || 0) * 100)}%</div></div>
                <div className="rounded bg-[var(--bg-base)] p-2"><span className="text-[var(--text-muted)]">Person</span><div className="mono font-bold">#{ev.person_track_id}</div></div>
                <div className="rounded bg-[var(--bg-base)] p-2"><span className="text-[var(--text-muted)]">Waste</span><div className="mono font-bold">#{ev.object_track_id} · {ev.object_type}</div></div>
                <div className="rounded bg-[var(--bg-base)] p-2"><span className="text-[var(--text-muted)]">Timestamp</span><div className="mono font-bold">{formatDate(ev.timestamp)}</div></div>
              </div>

              {(img || person || waste || face) && (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                  {img && (
                    <figure className="space-y-1">
                      <img src={evidenceFileUrl(img.image_path!)} className="h-40 w-full rounded-lg border border-[var(--border-subtle)] object-cover bg-black" alt="event snapshot" />
                      <figcaption className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Event Snapshot</figcaption>
                    </figure>
                  )}
                  {person && (
                    <figure className="space-y-1">
                      <img src={evidenceFileUrl(person.person_image_path!)} className="h-40 w-full rounded-lg border border-[var(--border-subtle)] object-cover bg-black" alt="person evidence" />
                      <figcaption className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Person Evidence</figcaption>
                    </figure>
                  )}
                  {waste && (
                    <figure className="space-y-1">
                      <img src={evidenceFileUrl(waste.waste_image_path!)} className="h-40 w-full rounded-lg border border-[var(--border-subtle)] object-cover bg-black" alt="waste evidence" />
                      <figcaption className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Waste Evidence</figcaption>
                    </figure>
                  )}
                  {face && (
                    <figure className="space-y-1">
                      <img src={evidenceFileUrl(face.face_image_path!)} className="h-40 w-full rounded-lg border border-[var(--border-subtle)] object-cover bg-black" alt="face evidence (sensitive)" />
                      <figcaption className="flex items-center justify-between text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                        <span>Face (sensitive)</span>
                        <span className="rounded bg-[var(--warning)]/20 px-1.5 py-0.5 text-[9px] font-bold text-[var(--warning)]">HUMAN REVIEW</span>
                      </figcaption>
                    </figure>
                  )}
                </div>
              )}

              {clip && (
                <div className="space-y-1.5">
                  <h4 className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-[var(--text-primary)]">
                    <Film className="h-3.5 w-3.5 text-[var(--accent)]" /> Event Clip
                  </h4>
                  <video key={`clip-${ev.id}`} controls className="w-full max-w-2xl rounded-lg border border-[var(--border-subtle)] bg-black"
                    src={evidenceFileUrl(clip.clip_path ?? clip.video_path!)} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
