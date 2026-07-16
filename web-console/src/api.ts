import type { JobRecord, LaunchPayload, Meta, Overview, RunDetail, SweepPayload } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error ?? `Request failed: ${response.status}`);
  }
  return payload as T;
}

export function fetchOverview(): Promise<Overview> {
  return request<Overview>("/api/overview");
}

export function fetchRunDetail(runId: string): Promise<RunDetail> {
  return request<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);
}

export async function fetchJobs(): Promise<JobRecord[]> {
  const payload = await request<{ jobs: JobRecord[] }>("/api/jobs");
  return payload.jobs;
}

export function startJob(payload: LaunchPayload): Promise<JobRecord> {
  return request<JobRecord>("/api/jobs", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function startSweep(payload: SweepPayload): Promise<JobRecord> {
  return request<JobRecord>("/api/sweeps", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function fetchMeta(): Promise<Meta> {
  return request<Meta>("/api/meta");
}

export function stopJob(jobId: string): Promise<JobRecord> {
  return request<JobRecord>(`/api/jobs/${encodeURIComponent(jobId)}/stop`, {
    method: "POST",
    body: "{}"
  });
}

