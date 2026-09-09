import { Check, Minus } from "lucide-react";
import { cn } from "../lib/utils";

/**
 * The canonical littering state sequence, shown as a vertical timeline
 * on the live monitoring page and as a checklist on the event detail page.
 */
export const STATE_SEQUENCE = [
  { key: "BAG_NEAR_PERSON", label: "Object associated with person" },
  { key: "BAG_CARRIED", label: "Object carried by person" },
  { key: "BAG_RELEASED", label: "Object released from hand" },
  { key: "BAG_ON_GROUND", label: "Object reached the ground" },
  { key: "PERSON_DEPARTED", label: "Person moved away" },
  { key: "VIOLATION_CONFIRMED", label: "No re-grab — littering candidate confirmed" },
] as const;

interface Props {
  // the current/reached state; all steps up to and including the match are "done"
  currentState?: string;
  // for the event detail page: an explicit satisfied-map overrides sequence
  satisfied?: Record<string, boolean>;
  variant?: "timeline" | "checklist";
}

export function EventTimeline({ currentState, satisfied, variant = "timeline" }: Props) {
  const reachedIndex = currentState
    ? STATE_SEQUENCE.findIndex((s) => s.key === currentState)
    : -1;

  return (
    <div className={cn(variant === "timeline" ? "space-y-0" : "space-y-2.5")}>
      {STATE_SEQUENCE.map((step, i) => {
        const isSatisfied = satisfied ? !!satisfied[step.key] : i <= reachedIndex;
        const isCurrent = variant === "timeline" && step.key === currentState;

        if (variant === "checklist") {
          return (
            <div key={step.key} className="flex items-center gap-3">
              <div
                className={cn(
                  "flex h-6 w-6 shrink-0 items-center justify-center rounded-full",
                  isSatisfied
                    ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                    : "bg-[var(--bg-elevated)] text-[var(--text-muted)]",
                )}
              >
                {isSatisfied ? <Check className="h-3.5 w-3.5" /> : <Minus className="h-3.5 w-3.5" />}
              </div>
              <span
                className={cn(
                  "text-[13px]",
                  isSatisfied ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]",
                )}
              >
                {step.label}
              </span>
            </div>
          );
        }

        // timeline variant
        return (
          <div key={step.key} className="relative flex gap-3 pb-5 last:pb-0">
            {/* connector line */}
            {i < STATE_SEQUENCE.length - 1 && (
              <div
                className={cn(
                  "absolute left-[11px] top-6 h-full w-0.5",
                  i < reachedIndex ? "bg-[var(--accent)]" : "bg-[var(--border-subtle)]",
                )}
              />
            )}
            <div
              className={cn(
                "relative z-10 flex h-6 w-6 shrink-0 items-center justify-center rounded-full border-2",
                isSatisfied
                  ? "border-[var(--accent)] bg-[var(--accent)]/15"
                  : "border-[var(--border-default)] bg-[var(--bg-panel)]",
                isCurrent && "ring-2 ring-[var(--accent)]/40",
              )}
            >
              {isSatisfied && <Check className="h-3 w-3 text-[var(--accent)]" />}
            </div>
            <div className="pt-0.5">
              <div
                className={cn(
                  "mono text-[11px] font-bold uppercase tracking-wider",
                  isSatisfied ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]",
                )}
              >
                {step.key.replace(/_/g, " ")}
              </div>
              <div className="text-[11px] text-[var(--text-secondary)]">{step.label}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
