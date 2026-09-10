import { ShieldAlert, ImageOff } from "lucide-react";
import type { Event, Evidence } from "../types";
import { evidenceFileUrl } from "../lib/api";
import { formatConfidence } from "../lib/utils";

/**
 * Primary, human-facing incident proof: the event clip on top, then exactly
 * three forensic crops (Person Actor / Face / Ground Waste Evidence) below.
 * Used identically by EventDetail and VideoAnalysisPage (NEW_PRODUCTION_ROADMAP
 * §2.2-2.3) so the two pages never drift into two different "primary
 * evidence" layouts again.
 */
interface ForensicAssetPanelProps {
  event: Event;
  evidence?: Evidence;
  className?: string;
}

export function ForensicAssetPanel({ event, evidence, className }: ForensicAssetPanelProps) {
  const actorLabel =
    event.event_actor_person_uid != null
      ? `PERSON UID #${event.event_actor_person_uid}`
      : event.event_actor_person_track_id != null
        ? `PERSON track #${event.event_actor_person_track_id}`
        : event.person_track_id
          ? `Track #${event.person_track_id} (legacy)`
          : "—";

  const objectLabel =
    event.event_object_uid != null
      ? `WASTE UID #${event.event_object_uid}`
      : event.event_object_track_id != null
        ? `WASTE track #${event.event_object_track_id}`
        : event.object_type;

  // Ground evidence prefers the dedicated ground-moment crop (P1-2); most
  // historical events predate that fix and only have the generic waste crop,
  // so fall back to it rather than showing an empty tile.
  const groundImage = evidence?.ground_image_path ?? evidence?.waste_image_path ?? null;
  const groundLabel = evidence?.ground_image_path ? "Ground Waste Evidence" : "Waste Evidence";

  return (
    <div className={className ? `panel space-y-4 border-[var(--danger)]/30 p-5 ${className}` : "panel space-y-4 border-[var(--danger)]/30 p-5"}>
      <h2 className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-[var(--danger)]">
        <ShieldAlert className="h-4 w-4" /> Primary Evidence
      </h2>

      <div className="grid grid-cols-3 gap-3 text-[11px]">
        <StatTile label="Actor" value={actorLabel} />
        <StatTile label="Object" value={objectLabel} />
        <StatTile label="Confidence" value={formatConfidence(event.confidence)} />
      </div>

      {evidence?.clip_path ? (
        <div className="space-y-2">
          <div className="text-[10px] font-bold uppercase tracking-wider text-[var(--accent)]">
            Event Clip — carry to departure
          </div>
          <video
            controls
            className="w-full rounded-lg border-2 border-[var(--accent)]/50 bg-black object-contain"
            src={evidenceFileUrl(evidence.clip_path)}
          />
        </div>
      ) : (
        <EmptyState label="Event clip was not captured for this event." />
      )}

      <div className="grid gap-3 sm:grid-cols-3">
        <ForensicCrop
          label="Person Actor"
          sub={event.event_actor_person_uid != null ? `UID #${event.event_actor_person_uid}` : undefined}
          imagePath={evidence?.person_image_path}
        />
        <ForensicCrop
          label="Face Crop"
          sub="HUMAN REVIEW"
          sensitive
          imagePath={evidence?.face_image_path}
        />
        <ForensicCrop
          label={groundLabel}
          sub={event.object_type}
          imagePath={groundImage}
        />
      </div>
    </div>
  );
}

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-[var(--bg-base)] p-2">
      <div className="text-[10px] uppercase text-[var(--text-muted)]">{label}</div>
      <div className="mono truncate font-bold text-[var(--text-primary)]">{value}</div>
    </div>
  );
}

function ForensicCrop({
  label,
  sub,
  imagePath,
  sensitive,
}: {
  label: string;
  sub?: string;
  imagePath?: string | null;
  sensitive?: boolean;
}) {
  return (
    <figure
      className={
        sensitive
          ? "rounded-lg border-2 border-[var(--warning)]/30 bg-[var(--bg-base)] p-2"
          : "rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2"
      }
    >
      {imagePath ? (
        <img src={evidenceFileUrl(imagePath)} alt={label} className="h-44 w-full rounded object-contain" />
      ) : (
        <div className="flex h-44 w-full flex-col items-center justify-center gap-2 rounded bg-[var(--bg-panel)] text-[var(--text-muted)]">
          <ImageOff className="h-6 w-6" />
          <span className="text-[10px] uppercase">Not captured</span>
        </div>
      )}
      <figcaption className="mt-1 flex items-center justify-between gap-1 text-[10px] uppercase text-[var(--text-muted)]">
        <span>{label}</span>
        {sub && (
          <span
            className={
              sensitive
                ? "rounded bg-[var(--warning)]/20 px-1.5 py-0.5 text-[9px] font-bold text-[var(--warning)]"
                : "text-[9px] font-normal normal-case text-[var(--text-secondary)]"
            }
          >
            {sub}
          </span>
        )}
      </figcaption>
    </figure>
  );
}

function EmptyState({ label }: { label: string }) {
  return (
    <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-4 text-center text-xs text-[var(--text-muted)]">
      {label}
    </div>
  );
}
