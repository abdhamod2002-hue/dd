import { ImageOff } from "lucide-react";
import { evidenceFileUrl } from "../lib/api";
import { cn } from "../lib/utils";

/**
 * One step of the behavioral sequence (approach -> carry -> release ->
 * ground -> departure). `imagePath` is optional because only carry/release/
 * ground currently have a dedicated capture (P1-2) — approach and departure,
 * and any carry/release/ground crop that failed to capture (both actor and
 * object bboxes must be visible at that exact frame), render as a
 * timestamp-only chip instead of a broken image.
 */
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
}

export function SequenceStrip({ steps, className }: SequenceStripProps) {
  if (!steps.length) return null;

  return (
    <div className={cn("panel space-y-3 p-4", className)}>
      <h2 className="text-[11px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Sequence</h2>
      <div className="flex gap-3 overflow-x-auto pb-1">
        {steps.map((step, idx) => (
          <div key={step.key} className="flex shrink-0 items-center gap-3">
            <SequenceCard step={step} index={idx + 1} />
            {idx < steps.length - 1 && (
              <div className="h-px w-6 shrink-0 bg-[var(--border-subtle)]" aria-hidden />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function SequenceCard({ step, index }: { step: SequenceStep; index: number }) {
  // Prefer the frame number: it's what the rest of the app already displays
  // (EventDetail's Behavior checklist reads detectorEvent.frames.*) and is
  // unambiguous. `timestamp` is source-dependent — the video-upload path
  // stamps wall-clock epoch seconds, not seconds-from-video-start, so
  // rendering it directly as "…s" can show something like "1788262570.1s".
  const timingLabel = step.frame != null ? `f${step.frame}` : step.timestamp != null ? `${step.timestamp.toFixed(1)}s` : null;
  return (
    <figure className="w-32 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-1.5">
      {step.imagePath ? (
        <img src={evidenceFileUrl(step.imagePath)} alt={step.label} className="h-20 w-full rounded object-cover" />
      ) : (
        <div className="flex h-20 w-full flex-col items-center justify-center gap-1 rounded bg-[var(--bg-panel)] text-[var(--text-muted)]">
          {timingLabel ? <span className="mono text-[10px]">{timingLabel}</span> : <ImageOff className="h-4 w-4" />}
        </div>
      )}
      <figcaption className="mt-1 flex items-center gap-1 text-[9px] font-bold uppercase text-[var(--text-secondary)]">
        <span className="flex h-3.5 w-3.5 shrink-0 items-center justify-center rounded-full bg-[var(--accent)]/20 text-[8px] text-[var(--accent)]">
          {index}
        </span>
        <span className="truncate">{step.label}</span>
      </figcaption>
    </figure>
  );
}

/**
 * Build the standard 5-step sequence from an Evidence row + the compact
 * detector-event dict (`_compact_detector_event` in backend/routers/analysis.py
 * — has `frames`/`timestamps` keyed by state name). Kept as a plain function,
 * not baked into the component, so a page can override/extend it.
 */
export function buildSequenceSteps(
  evidence: { carry_image_path?: string | null; release_image_path?: string | null; ground_image_path?: string | null } | undefined,
  detectorEvent?: { frames?: Record<string, number | null> | null; timestamps?: Record<string, number | null> | null } | null,
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
