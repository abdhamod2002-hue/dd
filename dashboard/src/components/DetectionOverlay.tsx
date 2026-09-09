import { cn } from "../lib/utils";

// Represents one tracked entity to draw over the live feed.
export interface OverlayEntity {
  trackId: number;
  label: string; // "Person" | "Plastic Bottle" | ...
  bbox: { x: number; y: number; w: number; h: number }; // normalized 0..1
  confidence: number;
  isPerson: boolean;
  source?: string;
  state?: string | null;
  associatedObjectId?: number | null;
  associatedPersonId?: number | null;
}

interface Props {
  entities: OverlayEntity[];
  // current AI state for the bound pair (shown top-center)
  aiState?: string;
  className?: string;
}

/**
 * DetectionOverlay renders bounding boxes + track IDs + confidence over a
 * video feed. It is positioned absolute over the <img>/<video> and uses
 * normalized coordinates so it scales with the feed.
 *
 * Person boxes are cyan; object boxes are amber; a confirmed-littering
 * pair flashes red.
 */
export function DetectionOverlay({ entities, aiState, className }: Props) {
  const confirmed = aiState === "VIOLATION_CONFIRMED";
  return (
    <div className={cn("pointer-events-none absolute inset-0", className)}>
      {entities.map((e) => {
        const color = e.isPerson ? "var(--accent)" : "var(--warning)";
        return (
          <div
            key={e.trackId}
            className="absolute border-2"
            style={{
              left: `${e.bbox.x * 100}%`,
              top: `${e.bbox.y * 100}%`,
              width: `${e.bbox.w * 100}%`,
              height: `${e.bbox.h * 100}%`,
              borderColor: color,
              boxShadow: confirmed ? "0 0 16px var(--danger-glow)" : `0 0 12px -4px ${color}`,
            }}
          >
            <div
              className="absolute -top-6 left-0 whitespace-nowrap px-1.5 py-0.5 text-[10px] font-bold mono"
              style={{ background: color, color: "#0a1020" }}
            >
              {e.label} #{e.trackId} · {Math.round(e.confidence * 100)}%
              {e.source && e.source !== "yolo" ? ` · ${e.source.toUpperCase()}` : ""}
              {e.state ? ` · ${e.state.replace(/_/g, " ")}` : ""}
            </div>
          </div>
        );
      })}

      {aiState && (
        <div className="absolute left-1/2 top-3 -translate-x-1/2 animate-state-in">
          <div
            className={cn(
              "rounded-full px-4 py-1.5 text-[12px] font-bold backdrop-blur",
              confirmed
                ? "bg-[var(--danger)]/25 text-[var(--danger)] glow-danger"
                : "bg-[var(--bg-base)]/70 text-[var(--text-primary)]",
            )}
          >
            <span className="mono uppercase tracking-wider">{aiState.replace(/_/g, " ")}</span>
          </div>
        </div>
      )}
    </div>
  );
}
