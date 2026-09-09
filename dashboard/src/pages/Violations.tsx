import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getCameras, getEvents } from "../lib/api";
import { Badge } from "../components/Badge";
import { formatTime, formatConfidence } from "../lib/utils";

const STATUSES = ["new", "reviewing", "confirmed", "rejected"] as const;
const PAGE_SIZE = 20;

export function Violations() {
  const { data: cameras } = useFetch(getCameras, []);
  const [page, setPage] = useState(0);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [cameraFilter, setCameraFilter] = useState<string>("");
  const [objectFilter, setObjectFilter] = useState<string>("");

  const { data, loading } = useFetch(
    () => getEvents(PAGE_SIZE, page * PAGE_SIZE),
    [page],
    4000
  );

  const events = data?.items ?? [];
  const total = data?.total ?? 0;

  // client-side filtering on the fetched page (real backend pagination is by offset)
  const filtered = useMemo(() => {
    return events.filter((e) => {
      if (statusFilter && e.status !== statusFilter) return false;
      if (cameraFilter && String(e.camera_id) !== cameraFilter) return false;
      if (objectFilter && !e.object_type.toLowerCase().includes(objectFilter.toLowerCase())) return false;
      if (search) {
        const q = search.toLowerCase();
        const hay = `#${e.id} ${e.object_type} ${e.person_track_id ?? ""} ${e.status}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });
  }, [events, statusFilter, cameraFilter, objectFilter, search]);

  const objectTypes = useMemo(() => {
    return Array.from(new Set(events.map((e) => e.object_type))).sort();
  }, [events]);

  return (
    <div className="mx-auto max-w-[1600px] space-y-5 p-5 lg:p-7">
      <div>
        <h1 className="text-xl font-bold text-[var(--text-primary)]">Violations</h1>
        <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
          Littering event management & review. Confirmed events are assistive candidates backed by temporal evidence; human review is required.
        </p>
      </div>

      {/* Filter bar */}
      <div className="panel flex flex-wrap items-center gap-3 p-3">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-muted)]" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by ID, object, person track…"
            className="w-full rounded-lg border border-[var(--border-default)] bg-[var(--bg-elevated)] py-2 pl-9 pr-3 text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-[var(--accent)]"
          />
        </div>

        <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} className={filterSelectCls}>
          <option value="">All statuses</option>
          {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>

        <select value={cameraFilter} onChange={(e) => setCameraFilter(e.target.value)} className={filterSelectCls}>
          <option value="">All cameras</option>
          {cameras?.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
        </select>

        <select value={objectFilter} onChange={(e) => setObjectFilter(e.target.value)} className={filterSelectCls}>
          <option value="">All objects</option>
          {objectTypes.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>

        {(statusFilter || cameraFilter || objectFilter || search) && (
          <button
            onClick={() => { setStatusFilter(""); setCameraFilter(""); setObjectFilter(""); setSearch(""); }}
            className="text-[12px] font-semibold text-[var(--text-muted)] hover:text-[var(--text-primary)]"
          >
            Clear
          </button>
        )}
      </div>

      {/* Table (desktop) / cards (mobile) */}
      <div className="panel overflow-hidden">
        {/* desktop table */}
        <div className="hidden md:block">
          <table className="w-full text-left text-[13px]">
            <thead className="border-b border-[var(--border-subtle)] text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
              <tr>
                <th className="px-4 py-3 font-semibold">ID</th>
                <th className="px-4 py-3 font-semibold">Time</th>
                <th className="px-4 py-3 font-semibold">Camera</th>
                <th className="px-4 py-3 font-semibold">Person</th>
                <th className="px-4 py-3 font-semibold">Object</th>
                <th className="px-4 py-3 font-semibold">Confidence</th>
                <th className="px-4 py-3 font-semibold">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--border-subtle)]">
              {loading && (
                <tr><td colSpan={7} className="px-4 py-10 text-center text-[12px] text-[var(--text-muted)]">Loading…</td></tr>
              )}
              {!loading && filtered.length === 0 && (
                <tr><td colSpan={7} className="px-4 py-10 text-center text-[12px] text-[var(--text-muted)]">No violations match the filters</td></tr>
              )}
              {filtered.map((ev) => (
                <tr key={ev.id} className="transition-colors hover:bg-[var(--bg-hover)]">
                  <td className="px-4 py-3">
                    <Link to={`/violations/${ev.id}`} className="mono font-bold text-[var(--accent)] hover:underline">#{ev.id}</Link>
                  </td>
                  <td className="px-4 py-3 text-[var(--text-secondary)]">{formatTime(ev.timestamp)}</td>
                  <td className="px-4 py-3 text-[var(--text-secondary)]">{cameras?.find((c) => c.id === ev.camera_id)?.name ?? `CAM-${ev.camera_id}`}</td>
                  <td className="px-4 py-3 mono text-[var(--text-secondary)]">{ev.person_track_id ?? "—"}</td>
                  <td className="px-4 py-3 text-[var(--text-primary)]">{ev.object_type}</td>
                  <td className="px-4 py-3 mono">{formatConfidence(ev.confidence)}</td>
                  <td className="px-4 py-3"><Badge status={ev.status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* mobile cards */}
        <div className="divide-y divide-[var(--border-subtle)] md:hidden">
          {filtered.map((ev) => (
            <Link key={ev.id} to={`/violations/${ev.id}`} className="block px-4 py-3 hover:bg-[var(--bg-hover)]">
              <div className="flex items-center justify-between">
                <span className="mono font-bold text-[var(--accent)]">#{ev.id}</span>
                <Badge status={ev.status} />
              </div>
              <div className="mt-1 text-[12px] text-[var(--text-secondary)]">{ev.object_type} · {formatConfidence(ev.confidence)}</div>
              <div className="mt-0.5 text-[11px] text-[var(--text-muted)]">{formatTime(ev.timestamp)} · CAM-{ev.camera_id}</div>
            </Link>
          ))}
          {filtered.length === 0 && !loading && (
            <div className="px-4 py-10 text-center text-[12px] text-[var(--text-muted)]">No violations</div>
          )}
        </div>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-[12px] text-[var(--text-muted)]">
        <span>
          {total > 0 ? `${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, total)} of ${total}` : "0 events"}
        </span>
        <div className="flex items-center gap-2">
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border-subtle)] text-[var(--text-secondary)] disabled:opacity-40 hover:bg-[var(--bg-hover)]"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="mono">Page {page + 1}</span>
          <button
            disabled={(page + 1) * PAGE_SIZE >= total}
            onClick={() => setPage((p) => p + 1)}
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border-subtle)] text-[var(--text-secondary)] disabled:opacity-40 hover:bg-[var(--bg-hover)]"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
}

const filterSelectCls =
  "rounded-lg border border-[var(--border-default)] bg-[var(--bg-elevated)] px-3 py-2 text-[13px] text-[var(--text-primary)] outline-none focus:border-[var(--accent)]";
