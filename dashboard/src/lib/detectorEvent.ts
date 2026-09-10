import type { Event } from "../types";

/**
 * One entry of `report.event_detector.confirmed_violations` (the compact
 * detector event produced by `_compact_detector_event` in
 * backend/routers/analysis.py).
 */
export interface DetectorViolation {
  person_track_id?: number | string | null;
  bag_track_id?: number | string | null;
  event_actor_person_track_id?: number | null;
  event_actor_person_uid?: number | null;
  event_object_track_id?: number | null;
  event_object_uid?: number | null;
  frames?: Record<string, number | null> | null;
  timestamps?: Record<string, number | null> | null;
  evidence?: Record<string, number> | null;
  [key: string]: unknown;
}

/**
 * Select the detector violation that belongs to a specific DB `Event` row.
 *
 * Never a fixed index: a job report's `confirmed_violations` array can hold
 * several events, and every consumer that once read `[0]` (or paired it with
 * a single, reused Evidence record) attached one event's detector state/
 * clip/crops to a DIFFERENT event's page (MASTER_REPAIR_PLAN P0-2, and the
 * equivalent bug in VideoAnalysisPage before this fix — multiple confirmed
 * violations in one job all rendered against `evidence[0]`, i.e. the first
 * event's images, regardless of which violation was being displayed).
 *
 * Match priority:
 *   1. frozen stable UIDs (authoritative actor/object identity),
 *   2. frozen stable track ids,
 *   3. legacy raw track ids (historical rows with NULL stable identity),
 *   4. a single unambiguous candidate.
 * Returns null rather than guessing among multiple unmatched violations.
 */
export function selectDetectorViolation(
  violations: DetectorViolation[] | undefined,
  event: Event | undefined,
): DetectorViolation | null {
  if (!violations?.length || !event) return null;
  if (event.event_actor_person_uid != null && event.event_object_uid != null) {
    const byUid = violations.find(
      (v) =>
        v.event_actor_person_uid === event.event_actor_person_uid &&
        v.event_object_uid === event.event_object_uid,
    );
    if (byUid) return byUid;
  }
  if (event.event_actor_person_track_id != null && event.event_object_track_id != null) {
    const byStableTrack = violations.find(
      (v) =>
        v.event_actor_person_track_id === event.event_actor_person_track_id &&
        v.event_object_track_id === event.event_object_track_id,
    );
    if (byStableTrack) return byStableTrack;
  }
  const personId = event.person_track_id != null ? Number(event.person_track_id) : NaN;
  const objectId = event.object_track_id != null ? Number(event.object_track_id) : NaN;
  if (!Number.isNaN(personId) && !Number.isNaN(objectId)) {
    const byLegacy = violations.find(
      (v) => Number(v.person_track_id) === personId && Number(v.bag_track_id) === objectId,
    );
    if (byLegacy) return byLegacy;
  }
  return violations.length === 1 ? violations[0] : null;
}
