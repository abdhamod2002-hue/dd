import { useRef, useState } from "react";
import type { AnalysisMarker } from "../types";
import { cn } from "../lib/utils";
import { Filter, Flag, Crosshair } from "lucide-react";

interface Props {
  markers: AnalysisMarker[];
  durationSec?: number | null;
  videoRef: React.RefObject<HTMLVideoElement>;
  className?: string;
}

export function TimelineMarkers({ markers, durationSec, videoRef, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [showAllProposals, setShowAllProposals] = useState(false);

  const duration = Math.max(
    0.1,
    durationSec ?? (markers.length ? Math.max(...markers.map((m) => m.timestamp)) + 1 : 1)
  );

  const jump = (marker: AnalysisMarker) => {
    const key = `${marker.label}-${marker.frame}`;
    setSelected(key);
    const video = videoRef.current;
    if (video) {
      video.currentTime = Math.max(0, marker.timestamp);
      video.play().catch(() => undefined);
    }
  };

  if (!markers.length) return null;

  // Filter out proposal clutter (OBJECT #11000x, etc.) so only key behavioral milestones are prominent
  const isBehavioralEvent = (m: AnalysisMarker) => {
    if (m.kind === "event") return true;
    const l = m.label.toUpperCase();
    return (
      l.includes("CARRIED") ||
      l.includes("RELEASE") ||
      l.includes("GROUND") ||
      l.includes("DEPARTURE") ||
      l.includes("EVENT") ||
      l.includes("VIOLATION") ||
      l.includes("PERSON")
    );
  };

  const keyMilestones = markers.filter(isBehavioralEvent);
  const rawProposals = markers.filter((m) => !isBehavioralEvent(m));

  const displayMarkers = showAllProposals ? markers : keyMilestones.length > 0 ? keyMilestones : markers;

  return (
    <div className={cn("space-y-3 rounded-xl border border-slate-800 bg-slate-950/80 p-3.5", className)}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Flag className="h-3.5 w-3.5 text-emerald-400" />
          Timeline Milestone Scrubber
        </span>
        {rawProposals.length > 0 && (
          <button
            type="button"
            onClick={() => setShowAllProposals((prev) => !prev)}
            className="flex items-center gap-1 text-[10px] font-semibold text-slate-400 hover:text-emerald-400 transition-colors"
          >
            <Filter className="h-3 w-3" />
            {showAllProposals
              ? `Hide Proposals (${rawProposals.length})`
              : `Show All Track Proposals (+${rawProposals.length})`}
          </button>
        )}
      </div>

      {/* Scrubber bar */}
      <div
        ref={containerRef}
        className="relative h-8 rounded-lg border border-slate-800 bg-slate-900/90 overflow-hidden"
      >
        <div className="absolute inset-x-3 top-1/2 h-0.5 -translate-y-1/2 bg-slate-800" />
        {displayMarkers.map((m, idx) => {
          const left = Math.min(98, Math.max(2, (m.timestamp / duration) * 100));
          const isEvent = m.kind === "event" || m.label.includes("RELEASE") || m.label.includes("GROUND");
          const isPerson = m.kind === "person" || m.label.includes("PERSON");
          const color = isEvent ? "#f43f5e" : isPerson ? "#06b6d4" : "#10b981";

          return (
            <button
              key={`${m.label}-${m.frame}-${idx}`}
              onClick={() => jump(m)}
              title={`${m.label} · ${m.timestamp.toFixed(2)}s · frame ${m.frame}`}
              className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-slate-950 transition-transform hover:scale-150 z-10 shadow-sm"
              style={{ left: `${left}%`, backgroundColor: color }}
            />
          );
        })}
      </div>

      {/* Clean, bounded milestone chips */}
      <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto pr-1">
        {displayMarkers.map((m, idx) => {
          const key = `${m.label}-${m.frame}`;
          const isSelected = selected === key;
          const isEvent = m.kind === "event" || m.label.includes("RELEASE") || m.label.includes("GROUND");
          const isPerson = m.kind === "person" || m.label.includes("PERSON");

          return (
            <button
              key={`${key}-${idx}`}
              onClick={() => jump(m)}
              className={cn(
                "rounded-md border px-2 py-1 text-[10px] font-bold uppercase tracking-wider transition-all flex items-center gap-1",
                isSelected
                  ? "border-emerald-500 bg-emerald-500/20 text-emerald-300 shadow-sm"
                  : isEvent
                  ? "border-rose-500/40 bg-rose-500/10 text-rose-300 hover:border-rose-500"
                  : isPerson
                  ? "border-cyan-500/40 bg-cyan-500/10 text-cyan-300 hover:border-cyan-500"
                  : "border-slate-800 bg-slate-900 text-slate-300 hover:border-slate-700 hover:text-white"
              )}
            >
              <Crosshair className="h-2.5 w-2.5 opacity-70" />
              <span>{m.label}</span>
              <span className="mono font-normal opacity-75">{m.timestamp.toFixed(1)}s</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
