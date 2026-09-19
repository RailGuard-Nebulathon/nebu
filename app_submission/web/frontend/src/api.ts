import type { PredictionResponse, TaskDescriptor, TaskId } from "./types";

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

export async function runPrediction(task: TaskId, file: File): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);
  let response: Response;
  try {
    response = await fetch(`${API_URL}/api/predict/${task}?mode=real`, { method: "POST", body: form });
  } catch {
    throw new Error("The analysis service is temporarily unavailable. Try again in a few minutes or contact the RailGuard administrator.");
  }
  return checked<PredictionResponse>(response);
}
