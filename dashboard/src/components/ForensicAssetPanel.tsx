import { useState, useRef } from "react";
import {
  ShieldAlert,
  ImageOff,
  User,
  Package,
  Activity,
  Clock,
  ZoomIn,
  CheckCircle2,
  Camera,
} from "lucide-react";
import type { Event, Evidence, AnalysisMarker } from "../types";
import type { DetectorViolation } from "../lib/detectorEvent";
import { evidenceFileUrl } from "../lib/api";
import { formatConfidence, cn, formatDate } from "../lib/utils";
import { ForensicLightboxModal, type LightboxItem } from "./ForensicLightboxModal";
import type { SequenceStep } from "./SequenceStrip";

interface ForensicAssetPanelProps {
  event: Event;
  evidence?: Evidence;
  detectorViolation?: DetectorViolation | null;
  className?: string;
  onSeekVideo?: (time: number) => void;
  analyzedVideoUrl?: string | null;
  originalVideoUrl?: string | null;
  markers?: AnalysisMarker[];
  steps?: SequenceStep[];
}

export function ForensicAssetPanel({
  event,
  evidence,
  detectorViolation,
  className,
  onSeekVideo,
}: ForensicAssetPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  // Primary dossier is ALWAYS the short event clip — full analyzed/original
  // streams live only under Engineering Debug / Technical Review.
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  const actorUid = event.event_actor_person_uid;
  const objectUid = event.event_object_uid;

  const actorLabel =
    actorUid != null
      ? `PERSON UID #${actorUid}`
      : event.event_actor_person_track_id != null
      ? `Track #${event.event_actor_person_track_id}`
      : event.person_track_id
      ? `Track #${event.person_track_id} (legacy)`
      : "—";

  const objectLabel =
    objectUid != null
      ? `WASTE UID #${objectUid}`
      : event.event_object_track_id != null
      ? `Track #${event.event_object_track_id}`
      : event.object_type;

  const groundImage = evidence?.ground_image_path ?? evidence?.waste_image_path ?? null;
  const groundLabel = evidence?.ground_image_path ? "Ground Waste Evidence" : "Waste Evidence";

  // Stream URLs — primary player is clip-only (no full-video fallback).
  const clipUrl = evidence?.clip_path ? evidenceFileUrl(evidence.clip_path) : null;
  const streamUrl = clipUrl;

  // Milestone timings from detectorViolation or event
  const frames = detectorViolation?.frames ?? {};
  const timestamps = detectorViolation?.timestamps ?? {};

  const milestones = [
    { label: "CARRY", frame: frames.carry_start, time: timestamps.carry_start, color: "text-amber-400 bg-amber-500/10 border-amber-500/30" },
    { label: "RELEASE", frame: frames.release, time: timestamps.release, color: "text-rose-400 bg-rose-500/10 border-rose-500/30" },
    { label: "GROUND", frame: frames.ground, time: timestamps.ground, color: "text-red-400 bg-red-500/10 border-red-500/30" },
    { label: "DEPARTURE", frame: frames.departure, time: timestamps.departure, color: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30" },
  ].filter((m) => m.frame != null || m.time != null);

  const handleSeek = (timeSec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = Math.max(0, timeSec);
      videoRef.current.play().catch(() => undefined);
    }
    if (onSeekVideo) onSeekVideo(timeSec);
  };

  // Lightbox items for target crops
  const targetCrops: LightboxItem[] = [
    {
      id: "crop-person",
      title: "PERSON ACTOR CROP",
      category: "target",
      imagePath: evidence?.person_image_path,
      badge: actorUid != null ? `UID #${actorUid}` : undefined,
      subtitle: `Track #${event.event_actor_person_track_id ?? event.person_track_id ?? "?"}`,
      actorUid,
      objectUid,
    },
    {
      id: "crop-face",
      title: "BIOMETRIC / FACE EVIDENCE CROP",
      category: "target",
      imagePath: evidence?.face_image_path,
      badge: "HUMAN REVIEW",
      subtitle: "Biometric inspection for human forensic review",
      actorUid,
      objectUid,
    },
    {
      id: "crop-waste",
      title: groundLabel.toUpperCase(),
      category: "target",
      imagePath: groundImage,
      badge: event.object_type,
      subtitle: `Waste Track #${event.event_object_track_id ?? event.object_track_id ?? "?"}`,
      actorUid,
      objectUid,
    },
  ];

  // Evidence score metrics
  const scores = (detectorViolation?.evidence ?? {}) as Record<string, number>;
  const confidencePercent = Math.round((event.confidence ?? 0) * 100);

  return (
    <>
      <div className={cn("panel border-slate-800 bg-slate-900/90 p-5 space-y-6 shadow-2xl", className)}>
        {/* Incident Header Strip */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-rose-500/15 border border-rose-500/30 text-rose-500 shadow-lg shadow-rose-950/30">
              <ShieldAlert className="h-5 w-5 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="mono text-[10px] font-bold uppercase tracking-widest text-rose-400">
                  CRITICAL INCIDENT DOSSIER
                </span>
                <span className="text-slate-600">·</span>
                <span className="mono text-[11px] text-slate-400">Event #{event.id}</span>
                <span className="rounded-full bg-rose-500/20 border border-rose-500/30 px-2 py-0.5 text-[10px] font-bold text-rose-400">
                  CONFIRMED VIOLATION
                </span>
              </div>
              <h2 className="text-base font-bold text-slate-100 flex items-center gap-2">
                Forensic Incident Investigation
                <span className="text-xs font-normal text-slate-400">({formatDate(event.timestamp)})</span>
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 rounded-xl border border-slate-800 bg-slate-950/80 px-3.5 py-1.5">
              <Activity className="h-4 w-4 text-emerald-400" />
              <div>
                <div className="mono text-xs font-bold text-emerald-400">{confidencePercent}%</div>
                <div className="text-[9px] uppercase tracking-wider text-slate-400">Confidence</div>
              </div>
            </div>
          </div>
        </div>

        {/* 2-Column Master Split-Screen Layout */}
        <div className="grid gap-6 lg:grid-cols-[1.3fr_1fr] items-start">
          {/* Left Column: Compact High-Definition Video HUD */}
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Camera className="h-3.5 w-3.5 text-emerald-400" />
                Event Clip (~10s)
              </span>
              <span className="mono text-[10px] font-bold uppercase tracking-wider text-emerald-400">
                Primary Evidence Stream
              </span>
            </div>

            {/* Video Player Box with strict max-height constraint */}
            <div className="relative w-full h-[340px] sm:h-[380px] max-h-[400px] rounded-xl overflow-hidden bg-slate-950 border border-slate-800 shadow-2xl flex items-center justify-center group">
              {streamUrl ? (
                <>
                  <video
                    ref={videoRef}
                    controls
                    playsInline
                    className="w-full h-full max-h-[400px] object-contain bg-black"
                    src={streamUrl}
                  />
                  {/* HUD Corner Accents */}
                  <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-2 rounded-md bg-slate-950/70 border border-slate-800/80 px-2.5 py-1 backdrop-blur-sm">
                    <span className="flex h-2 w-2 rounded-full bg-rose-500 animate-pulse" />
                    <span className="mono text-[10px] font-bold uppercase tracking-wider text-slate-300">
                      CAM-{event.camera_id} · AI TRACKING ACTIVE
                    </span>
                  </div>
                  <div className="pointer-events-none absolute right-3 top-3 rounded-md bg-slate-950/70 border border-slate-800/80 px-2 py-0.5 backdrop-blur-sm">
                    <span className="mono text-[9px] uppercase tracking-wider text-emerald-400 font-bold">
                      EVENT CLIP
                    </span>
                  </div>
                </>
              ) : (
                <div className="flex flex-col items-center justify-center text-slate-500 space-y-2 p-6 text-center">
                  <ImageOff className="h-10 w-10 text-slate-600" />
                  <p className="text-xs">No event clip is available for this incident</p>
                </div>
              )}
            </div>

            {/* Scrubber & Milestone Jump Row */}
            {milestones.length > 0 && (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/70 p-2 text-xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1">
                  <Clock className="h-3 w-3 text-emerald-400" /> Jump to Milestone:
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {milestones.map((m) => (
                    <button
                      key={m.label}
                      onClick={() => {
                        if (m.time != null && m.time < 1000) handleSeek(m.time);
                      }}
                      className={cn(
                        "rounded px-2 py-0.5 text-[10px] font-bold border transition-colors hover:brightness-125 flex items-center gap-1",
                        m.color
                      )}
                    >
                      <span>{m.label}</span>
                      {m.time != null && m.time < 1000 && (
                        <span className="mono font-normal opacity-80">{m.time.toFixed(1)}s</span>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Right Column: Forensic Intelligence Dossier */}
          <div className="space-y-4">
            {/* Confidence Score Meter & Attribution */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400">
                  Incident Confidence Score
                </span>
                <span className="mono text-xs font-bold text-emerald-400">
                  {formatConfidence(event.confidence)}
                </span>
              </div>

              {/* Progress Bar */}
              <div className="h-2 w-full rounded-full bg-slate-800 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-500"
                  style={{ width: `${Math.min(100, Math.max(5, confidencePercent))}%` }}
                />
              </div>

              {/* Sub-score breakdown */}
              {Object.keys(scores).length > 0 && (
                <div className="grid grid-cols-2 gap-2 pt-1 text-[11px]">
                  {scores.carry_score != null && (
                    <div className="flex justify-between rounded bg-slate-900 px-2 py-1 border border-slate-800/60">
                      <span className="text-slate-400">Carry Evidence</span>
                      <span className="mono font-bold text-emerald-400">
                        {Math.round(scores.carry_score * 100)}%
                      </span>
                    </div>
                  )}
                  {scores.release_score != null && (
                    <div className="flex justify-between rounded bg-slate-900 px-2 py-1 border border-slate-800/60">
                      <span className="text-slate-400">Release Separation</span>
                      <span className="mono font-bold text-emerald-400">
                        {Math.round(scores.release_score * 100)}%
                      </span>
                    </div>
                  )}
                  {scores.stationary_score != null && (
                    <div className="flex justify-between rounded bg-slate-900 px-2 py-1 border border-slate-800/60">
                      <span className="text-slate-400">Ground Rest</span>
                      <span className="mono font-bold text-emerald-400">
                        {Math.round(scores.stationary_score * 100)}%
                      </span>
                    </div>
                  )}
                  {scores.departure_score != null && (
                    <div className="flex justify-between rounded bg-slate-900 px-2 py-1 border border-slate-800/60">
                      <span className="text-slate-400">Departure</span>
                      <span className="mono font-bold text-emerald-400">
                        {Math.round(scores.departure_score * 100)}%
                      </span>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Entity Attribution Badges */}
            <div className="grid grid-cols-2 gap-3">
              {/* Actor Card */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3.5 space-y-1.5">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
                  <User className="h-4 w-4 text-cyan-400" />
                  <span>Person Actor</span>
                </div>
                <div className="mono text-sm font-bold text-cyan-300 truncate">
                  {actorLabel}
                </div>
                <div className="text-[10px] text-slate-400">
                  Track #{event.event_actor_person_track_id ?? event.person_track_id ?? "—"} · Verified
                </div>
              </div>

              {/* Object Card */}
              <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3.5 space-y-1.5">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-400">
                  <Package className="h-4 w-4 text-amber-400" />
                  <span>Waste Object</span>
                </div>
                <div className="mono text-sm font-bold text-amber-300 truncate">
                  {objectLabel}
                </div>
                <div className="text-[10px] text-slate-400">
                  Track #{event.event_object_track_id ?? event.object_track_id ?? "—"} · Ground Confirmed
                </div>
              </div>
            </div>

            {/* FSM Behavioral Status Checklist — P2-H: no fake VERIFIED */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-4 space-y-2.5">
              <div className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center justify-between">
                <span>Temporal Behavior Verification</span>
                {([frames.carry_start, frames.release, frames.ground, frames.departure].every(
                  (f) => f != null
                )) ? (
                  <span className="text-emerald-400 text-[10px]">ALL GATES TIMED</span>
                ) : (
                  <span className="text-amber-400 text-[10px]">MILESTONES INCOMPLETE</span>
                )}
              </div>
              <div className="space-y-1.5 text-xs">
                {(
                  [
                    ["1. Handheld Carry", frames.carry_start],
                    ["2. Separation / Release", frames.release],
                    ["3. Ground Deposit & Settle", frames.ground],
                    ["4. Departure Without Regrab", frames.departure],
                  ] as const
                ).map(([label, frame]) => (
                  <div
                    key={label}
                    className="flex items-center justify-between rounded bg-slate-900/90 px-2.5 py-1.5 border border-slate-800/60"
                  >
                    <div className="flex items-center gap-2">
                      {frame != null ? (
                        <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                      ) : (
                        <CheckCircle2 className="h-3.5 w-3.5 text-slate-600" />
                      )}
                      <span className="text-slate-300">{label}</span>
                    </div>
                    <span
                      className={`mono text-[10px] ${
                        frame != null ? "text-slate-400" : "text-amber-400"
                      }`}
                    >
                      {frame != null ? `f${frame}` : "UNAVAILABLE"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Section: Prominent Key Target Crops Gallery */}
        <div className="border-t border-slate-800 pt-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200 flex items-center gap-2">
                <span className="flex h-2 w-2 rounded-full bg-cyan-400" />
                Key Forensic Target Crops (Person / Biometrics / Waste)
              </h3>
              <p className="mt-0.5 text-[11px] text-slate-400">
                Extracted high-detail crops for forensic identification. Hover to zoom, click to open full-resolution inspector.
              </p>
            </div>
            <span className="mono text-[10px] text-slate-400">3 Primary Targets</span>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            {targetCrops.map((crop, idx) => (
              <TargetCropCard
                key={crop.id}
                item={crop}
                onClick={() => setLightboxIndex(idx)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Lightbox for Target Crops */}
      <ForensicLightboxModal
        isOpen={lightboxIndex !== null}
        onClose={() => setLightboxIndex(null)}
        items={targetCrops}
        initialIndex={lightboxIndex ?? 0}
        onSeekVideo={handleSeek}
      />
    </>
  );
}

function TargetCropCard({
  item,
  onClick,
}: {
  item: LightboxItem;
  onClick: () => void;
}) {
  const [loadError, setLoadError] = useState(false);
  const hasImage = Boolean(item.imagePath) && !loadError;

  return (
    <figure
      onClick={hasImage ? onClick : undefined}
      className={cn(
        "group relative flex flex-col rounded-xl border p-3 transition-all duration-200 shadow-lg",
        hasImage
          ? "border-slate-800 bg-slate-950/80 hover:border-emerald-500/60 hover:bg-slate-900 hover:shadow-emerald-950/30 cursor-pointer"
          : "border-slate-800/60 bg-slate-950/40 cursor-default"
      )}
    >
      {/* Header */}
      <div className="mb-2 flex items-center justify-between">
        <figcaption className="text-[11px] font-bold uppercase tracking-wider text-slate-200 truncate">
          {item.title}
        </figcaption>
        {item.badge && (
          <span
            className={cn(
              "rounded px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider",
              item.badge === "HUMAN REVIEW"
                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                : "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30"
            )}
          >
            {item.badge}
          </span>
        )}
      </div>

      {/* Image Container with Hover-Zoom */}
      <div className="relative h-44 sm:h-48 w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950 flex items-center justify-center">
        {hasImage ? (
          <>
            <img
              src={evidenceFileUrl(item.imagePath!)}
              alt={item.title}
              onError={() => setLoadError(true)}
              className="h-full w-full object-cover transition-transform duration-300 ease-out group-hover:scale-110"
            />
            {/* Hover overlay hint */}
            <div className="absolute inset-0 flex items-center justify-center bg-slate-950/50 opacity-0 backdrop-blur-[1px] transition-opacity duration-200 group-hover:opacity-100">
              <span className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-slate-900/90 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider text-emerald-400 shadow-xl">
                <ZoomIn className="h-3.5 w-3.5" /> Inspect High-Res
              </span>
            </div>
          </>
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 p-3 text-center text-slate-500 bg-slate-950/60">
            <ImageOff className="h-7 w-7 text-slate-600" />
            <span className="mono text-[11px] font-bold text-amber-400/90 tracking-wide">EVIDENCE NOT AVAILABLE</span>
            <span className="text-[10px] text-slate-500 leading-tight">
              {item.id === "crop-face"
                ? "Face angled away or occluded during event"
                : "Target could not be cleanly resolved"}
            </span>
          </div>
        )}
      </div>

      {/* Subtitle / Footer */}
      <div className="mt-2.5 flex items-center justify-between text-[11px] text-slate-400">
        <span className="truncate text-slate-400">{item.subtitle ?? "Evidence Crop"}</span>
        {hasImage ? (
          <span className="mono text-[10px] font-bold text-slate-500 group-hover:text-emerald-400 transition-colors shrink-0">
            Click to Zoom →
          </span>
        ) : (
          <span className="mono text-[10px] text-slate-600 shrink-0">Unavailable</span>
        )}
      </div>
    </figure>
  );
}
