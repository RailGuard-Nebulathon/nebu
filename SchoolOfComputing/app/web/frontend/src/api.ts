import type { AnalysisMetadata, AssetMetadataSuggestion, HistoryFilters, HistoryRecord, PredictionResponse, RunMode, TaskDescriptor, TaskId } from "./types";

const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = "The analysis could not be completed. Check the selected file and try again.";
    if (response.status === 413) message = "The file is too large. Choose a file smaller than 30 MB.";
    else if (response.status === 400 || response.status === 422) message = "This file could not be analysed. Check that it matches the selected subsystem and try again.";
    else if (response.status >= 500) message = "The analysis service is temporarily unavailable. Try again in a few minutes or contact the RailGuard administrator.";
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function fetchTasks(): Promise<TaskDescriptor[]> {
  return checked<TaskDescriptor[]>(await fetch(`${API_URL}/api/tasks`));
}

export async function runPrediction(task: TaskId, file: File, mode: Exclude<RunMode, "auto"> = "real"): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/predict/${task}?mode=${mode}`, { method: "POST", body: form });
  } catch {
    throw new Error("The analysis service is temporarily unavailable. Try again in a few minutes or contact the RailGuard administrator.");
  }
  return checked<PredictionResponse>(response);
}

export async function extractAssetMetadata(task: TaskId, file: File): Promise<AssetMetadataSuggestion> {
  const form = new FormData();
  form.append("file", file);
  const query = new URLSearchParams({ file_modified_ms: String(file.lastModified) });
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/metadata/${task}?${query.toString()}`, { method: "POST", body: form });
  } catch {
    throw new Error("The analysis service is temporarily unavailable. Try again in a few minutes or contact the RailGuard administrator.");
  }
  return checked<AssetMetadataSuggestion>(response);
}

export async function saveAnalysis(metadata: AnalysisMetadata, result: PredictionResponse): Promise<HistoryRecord> {
  return checked<HistoryRecord>(await fetch(`${API_URL}/api/history`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...metadata, result }),
  }));
}

export async function fetchHistory(filters: HistoryFilters = {}): Promise<HistoryRecord[]> {
  const query = new URLSearchParams();
  if (filters.asset) query.set("asset", filters.asset);
  if (filters.task) query.set("task", filters.task);
  if (filters.date_from) query.set("date_from", new Date(`${filters.date_from}T00:00:00`).toISOString());
  if (filters.date_to) query.set("date_to", new Date(`${filters.date_to}T23:59:59.999`).toISOString());
  const suffix = query.size ? `?${query.toString()}` : "";
  return checked<HistoryRecord[]>(await fetch(`${API_URL}/api/history${suffix}`));
}

export async function updateHistoryMetadata(id: string, metadata: AnalysisMetadata): Promise<HistoryRecord> {
  return checked<HistoryRecord>(await fetch(`${API_URL}/api/history/${id}/metadata`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(metadata),
  }));
}
