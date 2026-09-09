import { useState } from "react";
import { Activity, Cpu, Gauge, Package, User, Video } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getCameras, getStatus } from "../lib/api";
import { LiveCamera } from "../components/LiveCamera";
import { EventTimeline } from "../components/EventTimeline";
import { cn } from "../lib/utils";
import type { Camera as CameraType } from "../types";

export function LiveMonitoring() {
  const { data: cameras } = useFetch(getCameras, []);
  const { data: status } = useFetch(getStatus, [], 1500);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  const cam: CameraType | undefined = cameras?.find((c) => c.id === selectedId) ?? cameras?.[0];

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-[var(--text-primary)]">Live Monitoring</h1>
          <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
            Real-time AI tracking, behavior analysis & event detection
          </p>
        </div>
        {cameras && cameras.length > 0 && (
          <select
            value={cam?.id ?? ""}
            onChange={(e) => setSelectedId(Number(e.target.value))}
            className="rounded-lg border border-[var(--border-default)] bg-[var(--bg-elevated)] px-3 py-2 text-[13px] text-[var(--text-primary)] outline-none focus:border-[var(--accent)]"
          >
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} — {c.location}
              </option>
            ))}
          </select>
        )}
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_340px]">
        {/* Live feed + AI state */}
        <div className="space-y-4">
          {cam ? (
            <LiveCamera
              cameraId={cam.id}
              cameraName={`${cam.name} · ${cam.location}`}
              aiState={status?.live_state?.ai_state}
              entities={status?.live_state?.entities || []}
              showOverlayBoxes={false}
            />
          ) : (
            <div className="panel flex aspect-video flex-col items-center justify-center">
              <Video className="h-10 w-10 text-[var(--text-muted)]" />
              <p className="mt-3 text-[13px] text-[var(--text-secondary)]">No camera available</p>
            </div>
          )}

          {/* Live metrics strip */}
          <div className="panel grid grid-cols-2 gap-px overflow-hidden bg-[var(--border-subtle)] sm:grid-cols-3 xl:grid-cols-6">
            <MetricTile icon={Cpu} label="AI Engine" value={status?.ai_engine.status ?? "offline"} ok={status?.ai_engine.status === "online"} />
            <MetricTile icon={Video} label="Camera" value={status?.camera.status ?? "offline"} ok={status?.camera.status === "online"} />
            <MetricTile icon={Gauge} label="FPS" value={status?.processing.fps != null ? status.processing.fps.toFixed(1) : "—"} ok={status?.processing.fps != null} />
            <MetricTile icon={Activity} label="Latency" value={status?.processing.latency_ms != null ? `${status.processing.latency_ms.toFixed(0)}ms` : "—"} ok={status?.processing.latency_ms != null} />
            <MetricTile icon={User} label="Persons" value={String((status?.live_state?.entities || []).filter((e) => e.isPerson).length)} ok={(status?.live_state?.entities || []).some((e) => e.isPerson)} />
            <MetricTile icon={Package} label="Objects" value={String((status?.live_state?.entities || []).filter((e) => !e.isPerson).length)} ok={(status?.live_state?.entities || []).some((e) => !e.isPerson)} />
          </div>
        </div>

        {/* AI Behavior timeline */}
        <div className="space-y-4">
          <div className="panel p-4">
            <h2 className="mb-1 text-[13px] font-semibold text-[var(--text-primary)]">AI Behavior State</h2>
            <p className="mb-4 text-[11px] text-[var(--text-muted)]">
              The temporal state machine advances through this sequence to confirm a littering event.
            </p>
            <EventTimeline currentState={status?.live_state?.ai_state} variant="timeline" />
          </div>

          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">Detection Engine</h2>
            <div className="space-y-2.5">
              <EngineRow label="YOLO Detection" online={status?.ai_engine.status === "online"} />
              <EngineRow label="ByteTrack Tracking" online={status?.ai_engine.status === "online"} />
              <EngineRow label="MoveNet Pose" online={status?.ai_engine.status === "online"} />
              <EngineRow label="Person-Object Association" online={status?.ai_engine.status === "online"} />
              <EngineRow label="Temporal State Machine" online={status?.ai_engine.status === "online"} />
              <EngineRow label="Temporal Voting" online={status?.ai_engine.status === "online"} />
            </div>
          </div>

          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">Buffer</h2>
            <div className="grid grid-cols-3 gap-3 text-center">
              <BufferStat label="Window" value={status ? `${status.buffer.window_seconds}s` : "—"} />
              <BufferStat label="Frames" value={status?.buffer.frames_buffered ?? "—"} />
              <BufferStat label="Duration" value={status ? `${status.buffer.buffer_duration.toFixed(1)}s` : "—"} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricTile({ icon: Icon, label, value, ok }: { icon: typeof Activity; label: string; value: string; ok?: boolean }) {
  return (
    <div className="flex items-center gap-3 bg-[var(--bg-panel)] px-4 py-3">
      <Icon className={cn("h-4 w-4", ok ? "text-[var(--accent)]" : "text-[var(--text-muted)]")} />
      <div>
        <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
        <div className={cn("mono text-[12px] font-bold capitalize", ok ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]")}>{value}</div>
      </div>
    </div>
  );
}

function EngineRow({ label, online }: { label: string; online: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-[12px] text-[var(--text-secondary)]">{label}</span>
      <span className={cn("flex items-center gap-1.5 text-[11px] font-semibold", online ? "text-[var(--accent)]" : "text-[var(--text-muted)]")}>
        <span className={cn("h-1.5 w-1.5 rounded-full", online ? "bg-[var(--accent)] animate-live-pulse" : "bg-[var(--text-muted)]")} />
        {online ? "ONLINE" : "OFFLINE"}
      </span>
    </div>
  );
}

function BufferStat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div className="mono text-[16px] font-bold text-[var(--text-primary)]">{value}</div>
      <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
    </div>
  );
}
