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
  input_sha256: string;
  model_version: string;
  output_filename: string;
  rows: Array<Record<string, string | number>>;
  summary: Record<string, string | number | null>;
  visual: {
    ranking?: Array<{ car: string; score: number }>;
    probabilities?: Record<string, number>;
    segments?: Array<Record<string, string | number | null>>;
    damage?: number;
    interval?: [number, number];
  };
  csv_text: string;
  notices: string[];
}

export interface AnalysisMetadata {
  asset_id: string;
  component_info: string;
  measurement_time: string;
}

export type MetadataSource = "embedded" | "filename" | "file_modified";

export interface AssetMetadataSuggestion extends AnalysisMetadata {
  asset_source: MetadataSource;
  component_source: MetadataSource;
  measurement_time_source: MetadataSource;
  warnings: string[];
}

export interface HistoryRecord extends AnalysisMetadata {
  id: string;
  analysis_time: string;
  task: TaskId;
  mode: "real" | "demo";
  source_file: string;
  input_sha256: string;
  model_version: string;
  result: PredictionResponse;
}

export interface HistoryFilters {
  asset?: string;
  task?: TaskId | "";
  date_from?: string;
  date_to?: string;
}
