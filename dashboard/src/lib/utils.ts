import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function formatTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString(undefined, { hour12: false });
}

export function formatConfidence(c: number | null | undefined): string {
  if (c == null) return "—";
  return `${Math.round(c * 100)}%`;
}

// map a LitterState to a display color + label for the live overlay
export function stateVisual(state: LitterStateLike): {
  label: string;
  color: string;
  bg: string;
  glow: boolean;
} {
  const map: Record<string, { label: string; color: string; bg: string; glow: boolean }> = {
    UNKNOWN: { label: "Idle", color: "text-[var(--text-muted)]", bg: "bg-[var(--bg-elevated)]", glow: false },
    INTERACTING: { label: "Interacting", color: "text-[var(--text-secondary)]", bg: "bg-[var(--bg-elevated)]", glow: false },
    HOLDING: { label: "Holding", color: "text-[var(--info)]", bg: "bg-[var(--info)]/10", glow: false },
    RELEASE: { label: "Release Detected", color: "text-[var(--warning)]", bg: "bg-[var(--warning)]/15", glow: false },
    OBJECT_ON_GROUND: { label: "Object on Ground", color: "text-[var(--warning)]", bg: "bg-[var(--warning)]/15", glow: false },
    PERSON_AWAY: { label: "Person Away", color: "text-[var(--warning)]", bg: "bg-[var(--warning)]/15", glow: false },
    SUSPICIOUS: { label: "Suspicious", color: "text-[var(--warning)]", bg: "bg-[var(--warning)]/15", glow: false },
    LITTERING_CONFIRMED: { label: "Littering Confirmed", color: "text-[var(--danger)]", bg: "bg-[var(--danger)]/20", glow: true },
    NORMAL: { label: "Normal", color: "text-[var(--accent)]", bg: "bg-[var(--accent)]/10", glow: false },
  };
  return map[state] ?? map.UNKNOWN;
}

type LitterStateLike = string;

/**
 * A left-edge severity accent color, shared across Violations rows,
 * Dashboard's recent-event list, and EvidencePage cards, so a reviewer
 * scanning any of the three gets the same at-a-glance severity read
 * (confirmed = danger red, reviewing = warning amber, etc.) rather than
 * three independently-invented color schemes.
 */
export function severityAccent(status: string): string {
  switch (status) {
    case "confirmed":
      return "var(--danger)";
    case "reviewing":
      return "var(--warning)";
    case "new":
      return "var(--info)";
    case "rejected":
      return "var(--text-muted)";
    default:
      return "var(--border-default)";
  }
}

export function statusBadge(s: string): { label: string; className: string } {
  switch (s) {
    case "new":
      return { label: "New", className: "bg-[var(--info)]/15 text-[var(--info)]" };
    case "reviewing":
      return { label: "Reviewing", className: "bg-[var(--warning)]/15 text-[var(--warning)]" };
    case "confirmed":
      return { label: "Confirmed", className: "bg-[var(--danger)]/20 text-[var(--danger)]" };
    case "rejected":
      return { label: "Rejected", className: "bg-[var(--text-muted)]/15 text-[var(--text-secondary)]" };
    default:
      return { label: s, className: "bg-[var(--bg-elevated)] text-[var(--text-secondary)]" };
  }
}
