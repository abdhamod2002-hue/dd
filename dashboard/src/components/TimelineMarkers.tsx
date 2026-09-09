import { useRef, useState } from "react";
import type { AnalysisMarker } from "../types";
import { cn } from "../lib/utils";

interface Props {
  markers: AnalysisMarker[];
  durationSec?: number | null;
  videoRef: React.RefObject<HTMLVideoElement>;
  className?: string;
}

export function TimelineMarkers({ markers, durationSec, videoRef, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const duration = Math.max(0.1, durationSec ?? (markers.length ? Math.max(...markers.map((m) => m.timestamp)) + 1 : 1));

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

  return (
    <div className={cn("space-y-2", className)}>
      <div ref={containerRef} className="relative h-10 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)]">
        <div className="absolute inset-x-3 top-1/2 h-0.5 -translate-y-1/2 bg-[var(--border-subtle)]" />
        {markers.map((m, idx) => {
          const left = Math.min(98, Math.max(2, (m.timestamp / duration) * 100));
          const color =
            m.kind === "event"
              ? "var(--danger)"
              : m.kind === "object"
              ? "var(--warning)"
              : "var(--accent)";
          return (
            <button
              key={`${m.label}-${m.frame}-${idx}`}
              onClick={() => jump(m)}
              title={`${m.label} · ${m.timestamp.toFixed(2)}s · frame ${m.frame}`}
              className="absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-black transition-transform hover:scale-125"
              style={{ left: `${left}%`, background: color }}
            />
          );
        })}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {markers.map((m, idx) => {
          const key = `${m.label}-${m.frame}`;
          return (
            <button
              key={`${key}-${idx}`}
              onClick={() => jump(m)}
              className={cn(
                "rounded border px-2 py-1 text-[10px] font-bold uppercase tracking-wider transition-colors",
                selected === key
                  ? "border-[var(--accent)] bg-[var(--accent)]/15 text-[var(--accent)]"
                  : "border-[var(--border-subtle)] bg-[var(--bg-base)] text-[var(--text-secondary)] hover:border-[var(--accent)] hover:text-[var(--text-primary)]"
              )}
            >
              {m.label} · {m.timestamp.toFixed(2)}s
            </button>
          );
        })}
      </div>
    </div>
  );
}
