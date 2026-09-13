import { useState, useEffect } from "react";
import { X, ZoomIn, ZoomOut, RotateCcw, ChevronLeft, ChevronRight, Download, Clock, Hash, User, Package, ShieldAlert } from "lucide-react";
import { evidenceFileUrl } from "../lib/api";

export interface LightboxItem {
  id: string;
  title: string;
  category: "target" | "sequence";
  imagePath?: string | null;
  timestamp?: number | null;
  frame?: number | null;
  badge?: string;
  subtitle?: string;
  actorUid?: number | null;
  objectUid?: number | null;
  metadata?: Record<string, string | number | undefined | null>;
}

interface ForensicLightboxModalProps {
  isOpen: boolean;
  onClose: () => void;
  items: LightboxItem[];
  initialIndex?: number;
  onSeekVideo?: (timestamp: number) => void;
}

export function ForensicLightboxModal({
  isOpen,
  onClose,
  items,
  initialIndex = 0,
  onSeekVideo,
}: ForensicLightboxModalProps) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [zoom, setZoom] = useState(1);

  useEffect(() => {
    setCurrentIndex(initialIndex);
    setZoom(1);
  }, [initialIndex, isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") handlePrev();
      if (e.key === "ArrowRight") handleNext();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, items.length, currentIndex]);

  if (!isOpen || !items.length) return null;

  const currentItem = items[currentIndex] || items[0];

  const handlePrev = () => {
    setCurrentIndex((prev) => (prev > 0 ? prev - 1 : items.length - 1));
    setZoom(1);
  };

  const handleNext = () => {
    setCurrentIndex((prev) => (prev < items.length - 1 ? prev + 1 : 0));
    setZoom(1);
  };

  const handleZoomIn = () => setZoom((z) => Math.min(3, +(z + 0.25).toFixed(2)));
  const handleZoomOut = () => setZoom((z) => Math.max(0.75, +(z - 0.25).toFixed(2)));
  const handleResetZoom = () => setZoom(1);

  const fullImageUrl = currentItem.imagePath ? evidenceFileUrl(currentItem.imagePath) : null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/85 p-3 backdrop-blur-md animate-state-in">
      {/* Background click to close */}
      <div className="absolute inset-0" onClick={onClose} />

      <div
        className="relative z-10 flex h-[92vh] max-h-[860px] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-700/80 bg-slate-900 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-slate-800 bg-slate-950/80 px-5 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/15 border border-emerald-500/30 text-emerald-400">
              <ShieldAlert className="h-4 w-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="mono text-[11px] font-bold uppercase tracking-wider text-emerald-400">
                  {currentItem.category === "target" ? "FORENSIC ENTITY CROP" : "BEHAVIORAL SEQUENCE FRAME"}
                </span>
                <span className="text-slate-600">·</span>
                <span className="mono text-[11px] text-slate-400">
                  {currentIndex + 1} of {items.length}
                </span>
                {currentItem.badge && (
                  <span className="rounded bg-rose-500/20 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-rose-400">
                    {currentItem.badge}
                  </span>
                )}
              </div>
              <h3 className="text-sm font-bold text-slate-100 flex items-center gap-2">
                {currentItem.title}
                {currentItem.subtitle && (
                  <span className="text-xs font-normal text-slate-400">({currentItem.subtitle})</span>
                )}
              </h3>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <div className="flex items-center rounded-lg border border-slate-800 bg-slate-900/90 p-0.5">
              <button
                onClick={handleZoomOut}
                disabled={zoom <= 0.75}
                title="Zoom Out"
                className="rounded p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
              >
                <ZoomOut className="h-4 w-4" />
              </button>
              <span className="mono w-12 text-center text-[11px] font-medium text-slate-300">
                {Math.round(zoom * 100)}%
              </span>
              <button
                onClick={handleZoomIn}
                disabled={zoom >= 3}
                title="Zoom In"
                className="rounded p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200 disabled:opacity-40"
              >
                <ZoomIn className="h-4 w-4" />
              </button>
              <button
                onClick={handleResetZoom}
                title="Reset Zoom"
                className="rounded p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>

            {fullImageUrl && (
              <a
                href={fullImageUrl}
                target="_blank"
                rel="noreferrer"
                download
                title="Download High-Res Snapshot"
                className="rounded-lg border border-slate-800 bg-slate-900/90 p-2 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
              >
                <Download className="h-4 w-4" />
              </a>
            )}

            <button
              onClick={onClose}
              title="Close (Esc)"
              className="rounded-lg border border-slate-800 bg-slate-900/90 p-2 text-slate-400 hover:bg-rose-500/20 hover:text-rose-400 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Central Stage */}
        <div className="relative flex flex-1 items-center justify-center overflow-hidden bg-slate-950 p-4">
          {/* Previous Button */}
          {items.length > 1 && (
            <button
              onClick={handlePrev}
              title="Previous Snapshot (Left Arrow)"
              className="absolute left-4 top-1/2 z-20 -translate-y-1/2 rounded-full border border-slate-700/80 bg-slate-900/90 p-2.5 text-slate-300 shadow-xl backdrop-blur hover:bg-slate-800 hover:text-white"
            >
              <ChevronLeft className="h-5 w-5" />
            </button>
          )}

          {/* Image Display */}
          <div className="relative flex h-full w-full items-center justify-center overflow-auto">
            {fullImageUrl ? (
              <div
                className="transition-transform duration-200 ease-out"
                style={{ transform: `scale(${zoom})`, transformOrigin: "center center" }}
              >
                <img
                  src={fullImageUrl}
                  alt={currentItem.title}
                  className="max-h-[60vh] max-w-full rounded-lg border border-slate-800 object-contain shadow-2xl"
                />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center text-slate-500 space-y-2">
                <ShieldAlert className="h-10 w-10 text-slate-600" />
                <p className="text-sm">Visual crop was not captured for this state</p>
              </div>
            )}
          </div>

          {/* Next Button */}
          {items.length > 1 && (
            <button
              onClick={handleNext}
              title="Next Snapshot (Right Arrow)"
              className="absolute right-4 top-1/2 z-20 -translate-y-1/2 rounded-full border border-slate-700/80 bg-slate-900/90 p-2.5 text-slate-300 shadow-xl backdrop-blur hover:bg-slate-800 hover:text-white"
            >
              <ChevronRight className="h-5 w-5" />
            </button>
          )}
        </div>

        {/* Bottom Dossier & Metadata Footer */}
        <div className="border-t border-slate-800 bg-slate-950/90 px-6 py-3.5">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-wrap items-center gap-4 text-xs">
              {currentItem.frame != null && (
                <div className="flex items-center gap-1.5 text-slate-300">
                  <Hash className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="text-slate-500">Frame:</span>
                  <span className="mono font-bold text-slate-200">{currentItem.frame}</span>
                </div>
              )}
              {currentItem.timestamp != null && (
                <div className="flex items-center gap-1.5 text-slate-300">
                  <Clock className="h-3.5 w-3.5 text-emerald-400" />
                  <span className="text-slate-500">Time:</span>
                  <span className="mono font-bold text-slate-200">
                    {currentItem.timestamp < 1000 ? `${currentItem.timestamp.toFixed(2)}s` : "Recorded"}
                  </span>
                </div>
              )}
              {currentItem.actorUid != null && (
                <div className="flex items-center gap-1.5 text-slate-300">
                  <User className="h-3.5 w-3.5 text-cyan-400" />
                  <span className="text-slate-500">Actor:</span>
                  <span className="mono font-bold text-cyan-300">UID #{currentItem.actorUid}</span>
                </div>
              )}
              {currentItem.objectUid != null && (
                <div className="flex items-center gap-1.5 text-slate-300">
                  <Package className="h-3.5 w-3.5 text-amber-400" />
                  <span className="text-slate-500">Waste Object:</span>
                  <span className="mono font-bold text-amber-300">UID #{currentItem.objectUid}</span>
                </div>
              )}
            </div>

            {/* Jump to video button */}
            {onSeekVideo && currentItem.timestamp != null && (
              <button
                onClick={() => {
                  if (currentItem.timestamp != null) onSeekVideo(currentItem.timestamp);
                  onClose();
                }}
                className="rounded-lg border border-emerald-500/40 bg-emerald-500/10 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-emerald-400 hover:bg-emerald-500/20 transition-colors"
              >
                Seek Video Player Here →
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
