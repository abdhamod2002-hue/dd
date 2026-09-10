import type { RefObject } from "react";
import { ChevronDown, Crosshair, FileVideo, Play } from "lucide-react";
import type { AnalysisMarker } from "../types";
import { TimelineMarkers } from "./TimelineMarkers";
import { cn } from "../lib/utils";

/**
 * Engineering-only content: raw original/analyzed video comparison (full
 * detection overlay, every tracked entity — not just the confirmed actor +
 * object), the frame timeline, and the raw evidence-score breakdown.
 * Collapsed by default via native <details> (no JS state, no layout jump on
 * first paint) so it never competes with the primary evidence above it.
 */
interface DebugReviewPanelProps {
  originalVideoUrl?: string | null;
  analyzedVideoUrl?: string | null;
  analyzedVideoRef?: RefObject<HTMLVideoElement>;
  clipUrl?: string | null;
  markers?: AnalysisMarker[];
  durationSec?: number | null;
  evidenceScores?: Record<string, number>;
  onFocusEvent?: () => void;
  hasEventMarker?: boolean;
  defaultOpen?: boolean;
  className?: string;
}

export function DebugReviewPanel({
  originalVideoUrl,
  analyzedVideoUrl,
  analyzedVideoRef,
  clipUrl,
  markers = [],
  durationSec,
  evidenceScores,
  onFocusEvent,
  hasEventMarker,
  defaultOpen = false,
  className,
}: DebugReviewPanelProps) {
  return (
    <details className={cn("panel group p-5", className)} open={defaultOpen}>
      <summary className="flex cursor-pointer list-none flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--text-muted)]">
          <FileVideo className="h-4 w-4" /> Engineering Debug / Technical Review
        </h2>
        <div className="flex items-center gap-2">
          {hasEventMarker && onFocusEvent && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onFocusEvent();
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--danger)]/40 bg-[var(--danger)]/10 px-3 py-1.5 text-[11px] font-bold uppercase text-[var(--danger)] hover:bg-[var(--danger)]/20"
            >
              <Crosshair className="h-3.5 w-3.5" /> View Event
            </button>
          )}
          {clipUrl && (
            <a
              href={clipUrl}
              target="_blank"
              rel="noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] px-3 py-1.5 text-[11px] font-bold uppercase text-[var(--text-secondary)] hover:text-[var(--text-primary)]"
            >
              <Play className="h-3.5 w-3.5" /> Event Clip
            </a>
          )}
          <ChevronDown className="h-4 w-4 shrink-0 text-[var(--text-muted)] transition-transform group-open:rotate-180" />
        </div>
      </summary>

      <div className="mt-4 space-y-4">
        {originalVideoUrl || analyzedVideoUrl ? (
          <>
            <div className="grid gap-4 lg:grid-cols-2">
              {originalVideoUrl && (
                <div className="space-y-2">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Original video</div>
                  <video controls className="w-full rounded-lg border border-[var(--border-subtle)] bg-black object-contain" src={originalVideoUrl} />
                </div>
              )}
              {analyzedVideoUrl && (
                <div className="space-y-2">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Detection overlays — all tracks</div>
                  <video ref={analyzedVideoRef} controls className="w-full rounded-lg border border-[var(--border-subtle)] bg-black object-contain" src={analyzedVideoUrl} />
                </div>
              )}
            </div>
            {markers.length > 0 && analyzedVideoRef && (
              <div className="space-y-2">
                <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Timeline — click to jump</div>
                <TimelineMarkers markers={markers} durationSec={durationSec} videoRef={analyzedVideoRef} />
              </div>
            )}
          </>
        ) : (
          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-6 text-center text-xs text-[var(--text-muted)]">
            No analyzed video is linked to this event.
          </div>
        )}

        {evidenceScores && Object.keys(evidenceScores).length > 0 && (
          <div>
            <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-[var(--text-muted)]">Evidence score breakdown</div>
            <div className="mono space-y-1 text-[11px] text-[var(--text-muted)]">
              {Object.entries(evidenceScores).map(([k, v]) => (
                <div key={k} className="flex justify-between rounded bg-[var(--bg-base)] px-2 py-1">
                  <span>{k}</span>
                  <span>{Number(v).toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </details>
  );
}
