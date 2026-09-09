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
  { group: "Detection", label: "Litter model", value: "best.pt (5 litter classes) + waste_bag_real_v1.pt (waste bag, trained on real D:\\22 videos; model quality audit: NOT production-ready — see real_train/audit_evidence.json)", status: "implemented" },
  { group: "Detection", label: "Person confidence", value: "0.40", status: "implemented" },
  { group: "Detection", label: "Litter confidence", value: "0.35", status: "implemented" },

  // Behavior / event
  { group: "Behavior Engine", label: "Event threshold (voting)", value: "5.0 (weighted score)", status: "implemented" },
  { group: "Behavior Engine", label: "Revert threshold", value: "1.5", status: "implemented" },
  { group: "Behavior Engine", label: "Hold dwell", value: "0.25s", status: "implemented" },
  { group: "Behavior Engine", label: "Abandon window", value: "3.0s (put-down reversion)", status: "implemented" },

  // Buffer / evidence
  { group: "Evidence", label: "Buffer window", value: "6.0s", status: "implemented" },
  { group: "Evidence", label: "Pre-event recording", value: "3.0s", status: "implemented" },
  { group: "Evidence", label: "Post-event recording", value: "3.0s", status: "implemented" },
  { group: "Evidence", label: "Analysis FPS", value: "10 (throttled; capture at full FPS)", status: "implemented" },

  // Object classes
  { group: "Object Classes", label: "Litter candidates", value: "bottle, cup, can, tissue paper, wrapper, …", status: "implemented" },
  { group: "Object Classes", label: "Custom classes", value: "waste_bag — added via waste_bag_real_v1.pt (YOLOv8n, trained on real D:\\22 videos; audit verdict: over-fires on full video, needs more annotation+training before production use)", status: "implemented" },

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
