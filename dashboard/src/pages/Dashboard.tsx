import { Link } from "react-router-dom";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Gauge,
  TrendingUp,
  Video,
  Cpu,
  Upload,
  Smartphone
} from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getCameras, getEvents, getStatistics, getStatus } from "../lib/api";
import { StatCard } from "../components/StatCard";
import { LiveCamera } from "../components/LiveCamera";
import { Badge } from "../components/Badge";
import { cn, formatTime, formatConfidence, severityAccent } from "../lib/utils";
import type { Event } from "../types";

export function Dashboard() {
  const { data: cameras } = useFetch(getCameras, []);
  const { data: eventsList } = useFetch(() => getEvents(8, 0), [], 3000);
  const { data: stats } = useFetch(getStatistics, [], 5000);
  const { data: status } = useFetch(getStatus, [], 2000);

  const events: Event[] = eventsList?.items ?? [];
  const primaryCam = cameras?.[0];

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-[var(--text-primary)]">Command Center Dashboard</h1>
          <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
            Real-time dual-input AI littering analysis & video evidence surveillance
          </p>
        </div>

        {/* Dual Mode Quick Actions */}
        <div className="flex items-center gap-3">
          <Link
            to="/cameras"
            className="flex items-center gap-2 rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-default)] px-3.5 py-2 text-xs font-bold text-[var(--text-primary)] hover:border-[var(--accent)] transition-all"
          >
            <Smartphone className="h-4 w-4 text-[var(--accent)]" /> Connect Camera
          </Link>
          <Link
            to="/analysis"
            className="flex items-center gap-2 rounded-lg bg-[var(--accent)] px-3.5 py-2 text-xs font-bold text-black hover:bg-[var(--accent-dim)] transition-all"
          >
            <Upload className="h-4 w-4 text-black" /> Upload Video Analysis
          </Link>
        </div>
      </div>

      {/* Status area */}
      <div className="panel grid grid-cols-2 gap-px overflow-hidden bg-[var(--border-subtle)] sm:grid-cols-4">
        {[
          { label: "System", value: status?.system_online ? "Online" : "Offline", ok: status?.system_online, icon: CheckCircle2 },
          { label: "AI Engine", value: status?.ai_engine.status ?? "—", ok: status?.ai_engine.status === "online", icon: Cpu },
          { label: "Camera", value: status?.camera.status ?? "—", ok: status?.camera.status === "online", icon: Video },
          { label: "Processing", value: status?.processing.fps != null ? `${status.processing.fps.toFixed(0)} FPS` : "Idle", ok: status?.processing.fps != null, icon: Gauge },
        ].map((s) => (
          <div key={s.label} className="flex items-center gap-3 bg-[var(--bg-panel)] px-4 py-3">
            <s.icon className={cn("h-5 w-5", s.ok ? "text-[var(--accent)]" : "text-[var(--text-muted)]")} />
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">{s.label}</div>
              <div className={cn("mono text-[13px] font-bold capitalize", s.ok ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]")}>
                {s.value}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Main area: live camera + side info */}
      <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
        <div className="space-y-4">
          {primaryCam ? (
            <LiveCamera
              cameraId={primaryCam.id}
              cameraName={primaryCam.name}
              aiState={status?.live_state?.ai_state}
              entities={status?.live_state?.entities || []}
              showOverlayBoxes={false}
            />
          ) : (
            <div className="panel flex aspect-video flex-col items-center justify-center p-6 text-center">
              <Camera className="h-10 w-10 text-[var(--text-muted)] mb-3" />
              <p className="text-sm font-semibold text-[var(--text-primary)]">No Active Camera Feed</p>
              <p className="text-xs text-[var(--text-muted)] max-w-sm mt-1">
                You can connect your iPhone via Camo or directly upload recorded video files for full AI analysis.
              </p>
              <div className="flex items-center gap-3 mt-4">
                <Link to="/cameras" className="rounded-lg bg-[var(--bg-elevated)] border border-[var(--border-default)] px-3 py-1.5 text-xs font-semibold text-[var(--text-primary)] hover:border-[var(--accent)]">
                  Connect Phone →
                </Link>
                <Link to="/analysis" className="rounded-lg bg-[var(--accent)] px-3 py-1.5 text-xs font-bold text-black hover:bg-[var(--accent-dim)]">
                  Analyze Video File →
                </Link>
              </div>
            </div>
          )}

          {/* Stats row */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Violations Today" value={stats?.events_today ?? 0} icon={AlertTriangle} accent="danger" loading={!stats} />
            <StatCard label="Active Cameras" value={cameras?.length ?? 0} icon={Camera} loading={!cameras} />
            <StatCard label="Total Events" value={stats?.total_events ?? 0} icon={TrendingUp} accent="info" loading={!stats} />
            <StatCard
              label="Avg Confidence"
              value={stats?.avg_confidence ? formatConfidence(stats.avg_confidence) : "—"}
              icon={Gauge}
              accent="warning"
              loading={!stats}
            />
          </div>
        </div>

        {/* Side: recent violations */}
        <div className="panel flex flex-col">
          <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-4 py-3">
            <h2 className="text-[13px] font-semibold text-[var(--text-primary)]">Recent Event Candidates</h2>
            <Link to="/violations" className="text-[11px] font-semibold text-[var(--accent)] hover:underline">
              View all
            </Link>
          </div>
          <div className="divide-y divide-[var(--border-subtle)]">
            {events.length === 0 && (
              <div className="px-4 py-10 text-center text-[12px] text-[var(--text-muted)]">
                No violations detected yet
              </div>
            )}
            {events.map((ev) => (
              <Link
                key={ev.id}
                to={`/violations/${ev.id}`}
                className="flex items-center gap-3 border-l-2 px-4 py-3 transition-colors hover:bg-[var(--bg-hover)]"
                style={{ borderLeftColor: severityAccent(ev.status) }}
              >
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--danger)]/12">
                  <AlertTriangle className="h-4 w-4 text-[var(--danger)]" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="mono text-[12px] font-bold text-[var(--text-primary)]">#{ev.id}</span>
                    <span className="truncate text-[12px] text-[var(--text-secondary)]">{ev.object_type}</span>
                  </div>
                  <div className="mt-0.5 text-[11px] text-[var(--text-muted)]">
                    {formatTime(ev.timestamp)} · {formatConfidence(ev.confidence)}
                  </div>
                </div>
                <Badge status={ev.status} />
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
