export type TaskId = "door" | "acv" | "corrugation" | "shm";
export type RunMode = "auto" | "real" | "demo";

export interface TaskDescriptor {
  id: TaskId;
  name: string;
  short_name: string;
  description: string;
  accepted_extensions: string[];
  output_filename: string;
  bundle_available: boolean;
}

export interface PredictionResponse {
  task: TaskId;
  task_name: string;
  mode: "real" | "demo";
  source_file: string;
  output_filename: string;
  rows: Array<Record<string, string | number>>;
  summary: Record<string, string | number | null>;
  visual: {
    ranking?: Array<{ car: string; score: number }>;
    probabilities?: Record<string, number>;
    segments?: Array<Record<string, string>>;
    damage?: number;
    interval?: [number, number];
  };
  csv_text: string;
  notices: string[];
}
