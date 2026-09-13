import type {
  AnalysisManifest,
  Camera,
  Event,
  EventList,
  EventReview,
  Evidence,
  Statistics,
  SystemStatus,
  VideoAnalysisJob,
  VideoAnalysisJobList,
} from "../types";

const BASE_URL = "/api";

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

/* ----------------------------- Cameras ----------------------------- */
export function getCameras(): Promise<Camera[]> {
  return apiFetch<Camera[]>("/cameras");
}
export function getCamera(id: number): Promise<Camera> {
  return apiFetch<Camera>(`/cameras/${id}`);
}

/* ------------------------------ Events ----------------------------- */
export function getEvents(limit = 50, offset = 0): Promise<EventList> {
  return apiFetch<EventList>(`/events?limit=${limit}&offset=${offset}`);
}
export function getEvent(id: number): Promise<Event> {
  return apiFetch<Event>(`/events/${id}`);
}
export function getEventReview(id: number): Promise<EventReview> {
  return apiFetch<EventReview>(`/events/${id}/review`);
}

/* ----------------------------- Evidence ---------------------------- */
export function getEvidence(eventId: number): Promise<Evidence[]> {
  return apiFetch<Evidence[]>(`/evidence/${eventId}`);
}
export function evidenceFileUrl(relPath: string): string {
  return `${BASE_URL}/evidence/file/${relPath}`;
}
export function cameraStreamUrl(cameraId: number): string {
  return `${BASE_URL}/cameras/${cameraId}/stream`;
}

/* --------------------------- Statistics ---------------------------- */
export function getStatistics(): Promise<Statistics> {
  return apiFetch<Statistics>("/statistics");
}

/* ----------------------------- Status ------------------------------ */
export function getStatus(): Promise<SystemStatus> {
  return apiFetch<SystemStatus>("/status");
}

/* ------------------------- Video Analysis -------------------------- */
export async function uploadVideoAnalysis(file: File): Promise<VideoAnalysisJob> {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch(`${BASE_URL}/analysis/upload`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${body || res.statusText}`);
  }
  return res.json() as Promise<VideoAnalysisJob>;
}

export function getAnalysisJobs(limit = 50, offset = 0): Promise<VideoAnalysisJobList> {
  return apiFetch<VideoAnalysisJobList>(`/analysis/jobs?limit=${limit}&offset=${offset}`);
}

export function getAnalysisJob(jobId: number): Promise<VideoAnalysisJob> {
  return apiFetch<VideoAnalysisJob>(`/analysis/jobs/${jobId}`);
}

export function getAnalysisManifest(jobId: number): Promise<AnalysisManifest> {
  return apiFetch<AnalysisManifest>(`/analysis/jobs/${jobId}/manifest`);
}

export function getAnalysisJobEvents(jobId: number): Promise<Event[]> {
  return apiFetch<Event[]>(`/analysis/jobs/${jobId}/events`);
}

export function analyzedVideoUrl(jobId: number): string {
  return `${BASE_URL}/analysis/jobs/${jobId}/analyzed-video`;
}

export function originalVideoUrl(jobId: number): string {
  return `${BASE_URL}/analysis/jobs/${jobId}/original-video`;
}

export function frameAnalysisUrl(jobId: number): string {
  return `${BASE_URL}/analysis/jobs/${jobId}/frame-analysis`;
}

export function stopAnalysisJob(jobId: number): Promise<{ status: string; job_id: number; message: string }> {
  return apiFetch<{ status: string; job_id: number; message: string }>(`/analysis/jobs/${jobId}/stop`, {
    method: "POST",
  });
}

export function deleteAnalysisJob(jobId: number, confirm = true): Promise<{ status: string; job_id: number; message: string }> {
  return apiFetch<{ status: string; job_id: number; message: string }>(`/analysis/jobs/${jobId}?confirm=${confirm}`, {
    method: "DELETE",
  });
}

export function deleteSourceVideo(jobId: number, confirm = true): Promise<{ status: string; job_id: number; message: string }> {
  return apiFetch<{ status: string; job_id: number; message: string }>(`/analysis/jobs/${jobId}/video?confirm=${confirm}`, {
    method: "DELETE",
  });
}
