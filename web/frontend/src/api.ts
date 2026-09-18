import type { PredictionResponse, RunMode, TaskDescriptor, TaskId } from "./types";

const API_URL = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the HTTP fallback when the server did not return JSON.
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export async function fetchTasks(): Promise<TaskDescriptor[]> {
  return checked<TaskDescriptor[]>(await fetch(`${API_URL}/api/tasks`));
}

export async function runPrediction(task: TaskId, file: File, mode: RunMode): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  return checked<PredictionResponse>(
    await fetch(`${API_URL}/api/predict/${task}?mode=${mode}`, { method: "POST", body: form }),
  );
}
