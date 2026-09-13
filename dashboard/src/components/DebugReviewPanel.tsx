import type { RefObject } from "react";
import { ChevronDown, Crosshair, FileVideo, Play } from "lucide-react";
import type { AnalysisMarker } from "../types";
import { TimelineMarkers } from "./TimelineMarkers";
import { cn } from "../lib/utils";

/**
 * Engineering-only content: raw original/analyzed video comparison (full
 * detection overlay, every tracked entity — not just the confirmed actor +
 * object), the frame timeline, and the raw evidence-score breakdown.
 * Bounded cleanly (max-h-[360px]) so it never stretches vertically.
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
    <details className={cn("panel border-slate-800 bg-slate-900/80 group p-5 shadow-xl", className)} open={defaultOpen}>
      <summary className="flex cursor-pointer list-none flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-300">
          <FileVideo className="h-4 w-4 text-emerald-400" /> Engineering Debug / Technical Review
        </h2>
        <div className="flex items-center gap-2">
          {hasEventMarker && onFocusEvent && (
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onFocusEvent();
              }}
              className="inline-flex items-center gap-1.5 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-1.5 text-[11px] font-bold uppercase text-rose-400 hover:bg-rose-500/20 transition-colors"
            >
              <Crosshair className="h-3.5 w-3.5" /> View Event Marker
            </button>
          )}
          {clipUrl && (
            <a
              href={clipUrl}
              target="_blank"
              rel="noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-[11px] font-bold uppercase text-slate-300 hover:text-white transition-colors"
            >
              <Play className="h-3.5 w-3.5 text-emerald-400" /> Event Clip
            </a>
          )}
          <ChevronDown className="h-4 w-4 shrink-0 text-slate-400 transition-transform group-open:rotate-180" />
        </div>
      </summary>

      <div className="mt-4 space-y-4 border-t border-slate-800/80 pt-4">
        {originalVideoUrl || analyzedVideoUrl ? (
          <>
            <div className="grid gap-4 lg:grid-cols-2">
              {originalVideoUrl && (
                <div className="space-y-1.5">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    Original Source Video Feed
                  </div>
                  <div className="relative w-full h-[280px] sm:h-[320px] max-h-[360px] rounded-lg overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
                    <video
                      controls
                      playsInline
                      className="w-full h-full max-h-[360px] object-contain bg-black"
                      src={originalVideoUrl}
                    />
                  </div>
                </div>
              )}
              {analyzedVideoUrl && (
                <div className="space-y-1.5">
                  <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                    AI Detection Overlays — All Entity Tracks
                  </div>
                  <div className="relative w-full h-[280px] sm:h-[320px] max-h-[360px] rounded-lg overflow-hidden bg-slate-950 border border-slate-800 flex items-center justify-center">
                    <video
                      ref={analyzedVideoRef}
                      controls
                      playsInline
                      className="w-full h-full max-h-[360px] object-contain bg-black"
                      src={analyzedVideoUrl}
                    />
                  </div>
                </div>
              )}
            </div>
            {markers.length > 0 && analyzedVideoRef && (
              <div className="space-y-2 pt-2">
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                  Timeline Scrub Bar — Click Milestone to Jump
                </div>
                <TimelineMarkers markers={markers} durationSec={durationSec} videoRef={analyzedVideoRef} />
              </div>
            )}
          </>
        ) : (
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-6 text-center text-xs text-slate-400">
            No analyzed video is linked to this event.
          </div>
        )}

        {evidenceScores && Object.keys(evidenceScores).length > 0 && (
          <div className="rounded-lg border border-slate-800 bg-slate-950 p-3.5 space-y-2">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
              Low-Level Evidence Score Breakdown
            </div>
            <div className="mono grid grid-cols-2 sm:grid-cols-3 gap-2 text-[11px] text-slate-300">
              {Object.entries(evidenceScores).map(([k, v]) => (
                <div key={k} className="flex justify-between rounded bg-slate-900 px-2.5 py-1.5 border border-slate-800/80">
                  <span className="text-slate-400 capitalize">{k.replace(/_/g, " ")}:</span>
                  <span className="font-bold text-emerald-400">{Number(v).toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </details>
  );
}
