import { Check, Clock, AlertCircle } from "lucide-react";
import { cn } from "../lib/utils";

interface SettingRow {
  label: string;
  value: string;
  status: "implemented" | "future";
  group: string;
}

const SETTINGS: SettingRow[] = [
  // Camera source
  { group: "Camera Source", label: "Input device", value: "iPhone via Camo/Iriun (USB UVC)", status: "implemented" },
  { group: "Camera Source", label: "Resolution", value: "1920×1080 (configurable)", status: "implemented" },
  { group: "Camera Source", label: "Target FPS", value: "30", status: "implemented" },
  { group: "Camera Source", label: "RTSP IP camera", value: "Future — pipeline is source-agnostic", status: "future" },

  // Detection
  { group: "Detection", label: "Person model", value: "yolov8n.pt (COCO person)", status: "implemented" },
  { group: "Detection", label: "Litter model", value: "best.pt (5 litter classes, YOLO conf 0.35) + garbage_bag_v2.pt (waste bag, loaded via WASTE_BAG_WEIGHTS when present)", status: "implemented" },
  { group: "Detection", label: "Object tracker", value: "ByteTrack (YOLO-confirmed bags only; CSRT fallback bridged ≤ max_fallback_tracker_gap_frames = 16, then rejected as BAG_NOT_DETECTED)", status: "implemented" },
  { group: "Detection", label: "Person confidence", value: "0.40", status: "implemented" },
  { group: "Detection", label: "Litter confidence", value: "0.35", status: "implemented" },

  // Behavior / event engine — production FSM (LitteringEventDetector, config/events.yaml)
  { group: "Behavior Engine (FSM)", label: "Min carried frames", value: "6 (min_carried_frames)", status: "implemented" },
  { group: "Behavior Engine (FSM)", label: "Min stationary frames", value: "8 (min_stationary_frames)", status: "implemented" },
  { group: "Behavior Engine (FSM)", label: "Min departed frames", value: "2 (min_departed_frames)", status: "implemented" },
  { group: "Behavior Engine (FSM)", label: "Confirmation grace", value: "8 frames (confirmation_grace_frames)", status: "implemented" },
  { group: "Behavior Engine (FSM)", label: "Release detection", value: "windowed distance ratio 0.35 over release_window_frames=4; static-bag and feet-put-down variants gated by anti-furniture check", status: "implemented" },
  { group: "Behavior Engine (FSM)", label: "Abandonment confirmation", value: "8 frames on ground, unreclaimed (min_abandonment_frames)", status: "implemented" },
  { group: "Behavior Engine (FSM)", label: "Bin vs ground gate", value: "require_ground_confirmation=true; bin-zone deposits rejected as BIN_ZONE_DEPOSIT", status: "implemented" },
  { group: "Behavior Engine (FSM)", label: "Min event confidence", value: "0.80 (min_event_confidence)", status: "implemented" },

  // Buffer / evidence
  { group: "Evidence", label: "Buffer window", value: "6.0s", status: "implemented" },
  { group: "Evidence", label: "Pre-event recording", value: "3.0s", status: "implemented" },
  { group: "Evidence", label: "Post-event recording", value: "3.0s", status: "implemented" },
  { group: "Evidence", label: "Analysis FPS", value: "10 (throttled; capture at full FPS)", status: "implemented" },

  // Object classes
  { group: "Object Classes", label: "Litter candidates", value: "bottle, cup, can, tissue paper, wrapper, …", status: "implemented" },
  { group: "Object Classes", label: "Custom classes", value: "Garbage Bag — added via garbage_bag_v2.pt (YOLOv8n; resolved by _resolve_bag_weights: WASTE_BAG_WEIGHTS env → garbage_bag_v2.pt → None)", status: "implemented" },

  // System
  { group: "System", label: "Database", value: "PostgreSQL (docker-compose)", status: "implemented" },
  { group: "System", label: "Backend", value: "FastAPI on :8000", status: "implemented" },
  { group: "System", label: "Dashboard", value: "React on :5173", status: "implemented" },
  { group: "System", label: "Face recognition (DeepFace)", value: "Future — optional, post-event only", status: "future" },
];

export function Settings() {
  const groups = Array.from(new Set(SETTINGS.map((s) => s.group)));

  return (
    <div className="mx-auto max-w-[1000px] space-y-6 p-5 lg:p-7">
      <div>
        <h1 className="text-xl font-bold text-[var(--text-primary)]">Settings</h1>
        <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
          System configuration. Implemented settings reflect the actual running system; future settings are scoped but not yet built.
        </p>
      </div>

      {groups.map((group) => (
        <div key={group} className="panel overflow-hidden">
          <div className="border-b border-[var(--border-subtle)] bg-[var(--bg-elevated)] px-4 py-2.5">
            <h2 className="text-[12px] font-bold uppercase tracking-wider text-[var(--text-secondary)]">{group}</h2>
          </div>
          <div className="divide-y divide-[var(--border-subtle)]">
            {SETTINGS.filter((s) => s.group === group).map((s) => (
              <div key={s.label} className="flex items-center justify-between px-4 py-3">
                <div className="min-w-0">
                  <div className="text-[13px] font-medium text-[var(--text-primary)]">{s.label}</div>
                  <div className="mono mt-0.5 truncate text-[12px] text-[var(--text-secondary)]">{s.value}</div>
                </div>
                <StatusTag status={s.status} />
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="panel flex items-start gap-3 p-4">
        <AlertCircle className="mt-0.5 h-5 w-5 shrink-0 text-[var(--warning)]" />
        <p className="text-[12px] text-[var(--text-secondary)]">
          These values reflect the defaults in <code className="mono text-[var(--accent)]">inference/pipeline.py</code> and{" "}
          <code className="mono text-[var(--accent)]">inference/detection/yolo_detector.py</code>. Runtime overrides are passed via
          the <code className="mono text-[var(--accent)]">run_pipeline.py</code> CLI flags (<code className="mono">--buffer</code>,{" "}
          <code className="mono">--analysis-fps</code>, <code className="mono">--pre</code>, <code className="mono">--post</code>).
          A web-editable settings store is future work.
        </p>
      </div>
    </div>
  );
}

function StatusTag({ status }: { status: "implemented" | "future" }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider",
        status === "implemented"
          ? "bg-[var(--accent)]/12 text-[var(--accent)]"
          : "bg-[var(--warning)]/12 text-[var(--warning)]",
      )}
    >
      {status === "implemented" ? <Check className="h-3 w-3" /> : <Clock className="h-3 w-3" />}
      {status === "implemented" ? "Implemented" : "Future"}
    </span>
  );
}
