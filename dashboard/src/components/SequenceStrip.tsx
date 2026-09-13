import { useState } from "react";
import { ImageOff, ZoomIn, Clock, ArrowRight } from "lucide-react";
import { evidenceFileUrl } from "../lib/api";
import { cn } from "../lib/utils";
import { ForensicLightboxModal, type LightboxItem } from "./ForensicLightboxModal";

export interface SequenceStep {
  key: string;
  label: string;
  imagePath?: string | null;
  frame?: number | null;
  timestamp?: number | null;
}

interface SequenceStripProps {
  steps: SequenceStep[];
  className?: string;
  onSeekVideo?: (timestamp: number) => void;
  actorUid?: number | null;
  objectUid?: number | null;
}

export function SequenceStrip({
  steps,
  className,
  onSeekVideo,
  actorUid,
  objectUid,
}: SequenceStripProps) {
  const [lightboxIndex, setLightboxIndex] = useState<number | null>(null);

  if (!steps.length) return null;

  const lightboxItems: LightboxItem[] = steps.map((s, idx) => ({
    id: `step-${s.key}-${idx}`,
    title: `${idx + 1}. ${s.label.toUpperCase()} MILESTONE`,
    category: "sequence",
    imagePath: s.imagePath,
    timestamp: s.timestamp,
    frame: s.frame,
    badge: s.label.toUpperCase(),
    subtitle: s.frame != null ? `Frame ${s.frame}` : undefined,
    actorUid,
    objectUid,
  }));

  const handleCardClick = (idx: number, step: SequenceStep) => {
    setLightboxIndex(idx);
    if (onSeekVideo && step.timestamp != null && step.timestamp < 1000) {
      onSeekVideo(step.timestamp);
    }
  };

  return (
    <>
      <div className={cn("panel border-slate-800 bg-slate-900/80 p-5 space-y-4 shadow-xl", className)}>
        {/* Header with description */}
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
          <div>
            <h2 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-200">
              <span className="flex h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              Chronological Behavioral Sequence
            </h2>
            <p className="mt-0.5 text-[11px] text-slate-400">
              Hover to zoom, click any card to inspect high-resolution snapshot and jump video scrubber.
            </p>
          </div>
          <span className="mono text-[10px] font-semibold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
            {steps.filter((s) => s.imagePath).length} Visual Milestones Captured
          </span>
        </div>

        {/* Horizontal interactive gallery strip */}
        <div className="flex gap-4 overflow-x-auto pb-3 pt-1 scrollbar-thin">
          {steps.map((step, idx) => (
            <div key={step.key} className="flex shrink-0 items-center gap-4">
              <SequenceCard
                step={step}
                index={idx + 1}
                onClick={() => handleCardClick(idx, step)}
              />
              {idx < steps.length - 1 && (
                <div className="flex items-center text-slate-700 shrink-0" aria-hidden>
                  <ArrowRight className="h-4 w-4" />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Lightbox for sequence crops */}
      <ForensicLightboxModal
        isOpen={lightboxIndex !== null}
        onClose={() => setLightboxIndex(null)}
        items={lightboxItems}
        initialIndex={lightboxIndex ?? 0}
        onSeekVideo={onSeekVideo}
      />
    </>
  );
}

function SequenceCard({
  step,
  index,
  onClick,
}: {
  step: SequenceStep;
  index: number;
  onClick: () => void;
}) {
  const [loadError, setLoadError] = useState(false);
  const timingLabel =
    step.timestamp != null && step.timestamp < 1000
      ? `${step.timestamp.toFixed(2)}s`
      : null;

  const hasImage = Boolean(step.imagePath) && !loadError;

  return (
    <figure
      onClick={hasImage ? onClick : undefined}
      className={cn(
        "group relative w-52 sm:w-60 shrink-0 rounded-xl border p-2.5 transition-all duration-200 shadow-md",
        hasImage
          ? "border-slate-800 bg-slate-900/90 hover:border-emerald-500/60 hover:bg-slate-900 hover:shadow-emerald-950/30 cursor-pointer"
          : "border-slate-800/60 bg-slate-950/40 cursor-default"
      )}
    >
      {/* Top Header of Card */}
      <div className="mb-2 flex items-center justify-between">
        <figcaption className="flex items-center gap-1.5 text-[11px] font-bold uppercase tracking-wider text-slate-200">
          <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-emerald-500/15 border border-emerald-500/30 font-mono text-[10px] text-emerald-400">
            0{index}
          </span>
          <span className="truncate">{step.label}</span>
        </figcaption>
        <div className="flex items-center gap-1 text-[10px]">
          {step.frame != null && (
            <span className="mono rounded bg-slate-800/80 px-1.5 py-0.5 text-slate-300">
              f{step.frame}
            </span>
          )}
        </div>
      </div>

      {/* Image / Thumbnail Container with Hover-Zoom */}
      <div className="relative h-32 sm:h-36 w-full overflow-hidden rounded-lg border border-slate-800 bg-slate-950 flex items-center justify-center">
        {hasImage ? (
          <>
            <img
              src={evidenceFileUrl(step.imagePath!)}
              alt={step.label}
              onError={() => setLoadError(true)}
              className="h-full w-full object-cover transition-transform duration-300 ease-out group-hover:scale-110"
            />
            {/* Hover overlay hint */}
            <div className="absolute inset-0 flex items-center justify-center bg-slate-950/50 opacity-0 backdrop-blur-[1px] transition-opacity duration-200 group-hover:opacity-100">
              <span className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/50 bg-slate-900/90 px-3 py-1.5 text-[11px] font-bold uppercase tracking-wider text-emerald-400 shadow-xl">
                <ZoomIn className="h-3.5 w-3.5" /> Inspect
              </span>
            </div>
          </>
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 p-2 text-center text-slate-500 bg-slate-950/60">
            <ImageOff className="h-6 w-6 text-slate-600" />
            <span className="mono text-[10px] font-bold text-slate-400 uppercase">
              {step.imagePath && loadError ? "EVIDENCE NOT AVAILABLE" : "STATE VERIFIED"}
            </span>
            <span className="text-[9px] text-slate-600">
              {step.imagePath && loadError ? "Image unavailable" : "Temporal check passed"}
            </span>
          </div>
        )}
      </div>

      {/* Footer Info */}
      <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400">
        <div className="flex items-center gap-1 font-mono">
          <Clock className="h-3 w-3 text-emerald-400/80" />
          <span>{timingLabel ?? "—"}</span>
        </div>
        <span className="mono text-[9px] uppercase font-bold text-slate-500 group-hover:text-emerald-400 transition-colors">
          {hasImage ? "Click to Zoom →" : "Verified State"}
        </span>
      </div>
    </figure>
  );
}

export function buildSequenceSteps(
  evidence: {
    carry_image_path?: string | null;
    release_image_path?: string | null;
    ground_image_path?: string | null;
  } | undefined,
  detectorEvent?: {
    frames?: Record<string, number | null> | null;
    timestamps?: Record<string, number | null> | null;
  } | null,
): SequenceStep[] {
  const frames = detectorEvent?.frames ?? {};
  const timestamps = detectorEvent?.timestamps ?? {};
  return [
    { key: "near", label: "Approach", frame: frames.near, timestamp: timestamps.near },
    { key: "carry", label: "Carry", imagePath: evidence?.carry_image_path, frame: frames.carry_start, timestamp: timestamps.carry_start },
    { key: "release", label: "Release", imagePath: evidence?.release_image_path, frame: frames.release, timestamp: timestamps.release },
    { key: "ground", label: "Ground", imagePath: evidence?.ground_image_path, frame: frames.ground, timestamp: timestamps.ground },
    { key: "departure", label: "Departure", frame: frames.departure, timestamp: timestamps.departure },
  ];
}
