import { useEffect, useRef, useState } from "react";
import { Video, VideoOff, Maximize2 } from "lucide-react";
import { cn } from "../lib/utils";
import { cameraStreamUrl } from "../lib/api";
import { DetectionOverlay, type OverlayEntity } from "./DetectionOverlay";

interface Props {
  cameraId: number;
  cameraName?: string;
  aiState?: string;
  entities?: OverlayEntity[];
  className?: string;
  /** compact mode hides the chrome (used on the main dashboard tile) */
  compact?: boolean;
  /** set false when the MJPEG stream already contains AI overlays */
  showOverlayBoxes?: boolean;
}

export function LiveCamera({
  cameraId,
  cameraName,
  aiState,
  entities = [],
  className,
  compact,
  showOverlayBoxes = true,
}: Props) {
  const imgRef = useRef<HTMLImageElement>(null);
  const [live, setLive] = useState(false);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setLoaded(false);
  }, [cameraId]);

  return (
    <div className={cn("panel relative overflow-hidden", className)}>
      {!compact && (
        <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-4 py-2.5">
          <div className="flex items-center gap-2">
            <Video className="h-4 w-4 text-[var(--text-secondary)]" />
            <span className="text-[12px] font-semibold text-[var(--text-primary)]">
              {cameraName ?? `CAM-${cameraId}`}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider",
                live
                  ? "bg-[var(--danger)]/15 text-[var(--danger)]"
                  : "bg-[var(--bg-elevated)] text-[var(--text-muted)]",
              )}
            >
              <span className={cn("h-1.5 w-1.5 rounded-full", live ? "bg-[var(--danger)] animate-live-pulse" : "bg-[var(--text-muted)]")} />
              {live ? "Live" : "Offline"}
            </span>
            <button
              className="text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              onClick={() => imgRef.current?.requestFullscreen?.()}
              aria-label="Fullscreen"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      <div className="relative aspect-video bg-[var(--bg-base)]">
        <img
          ref={imgRef}
          src={cameraStreamUrl(cameraId)}
          alt={`Camera ${cameraId} live feed`}
          className="h-full w-full object-contain"
          onLoad={() => {
            setLoaded(true);
            setLive(true);
          }}
          onError={() => {
            setLive(false);
            setLoaded(false);
          }}
        />

        {/* detection overlays (optional; live stream is already annotated) */}
        {loaded && showOverlayBoxes && <DetectionOverlay entities={entities} aiState={aiState} />}

        {/* loading / waiting state */}
        {!loaded && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
            <VideoOff className="h-10 w-10 text-[var(--text-muted)]" />
            <p className="mt-3 text-[12px] font-medium text-[var(--text-secondary)]">
              Waiting for camera
            </p>
            <p className="mono mt-1 text-[10px] text-[var(--text-muted)]">
              connect iPhone via Camo/Iriun to start the stream
            </p>
          </div>
        )}

        {/* footer metrics bar */}
        {!compact && loaded && (
          <div className="absolute bottom-0 left-0 right-0 flex items-center gap-4 bg-gradient-to-t from-[var(--bg-base)]/90 to-transparent px-4 py-2 text-[10px] mono text-[var(--text-secondary)]">
            <span>{live ? "STREAMING" : "—"}</span>
            {entities.length > 0 && <span>{entities.length} TRACKED</span>}
            {aiState && <span className="uppercase">{aiState.replace(/_/g, " ")}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
