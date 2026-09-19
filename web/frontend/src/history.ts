import type { HistoryRecord, TaskId } from "./types";

export function chronological(records: HistoryRecord[]): HistoryRecord[] {
  return [...records].sort((left, right) => Date.parse(left.measurement_time) - Date.parse(right.measurement_time));
}

export function doorRate(record: HistoryRecord): number {
  const cycles = Number(record.result.summary.cycles || 0);
  return cycles > 0 ? Number(record.result.summary.abnormal_cycles || 0) / cycles : 0;
}

export function trendValue(record: HistoryRecord): number | null {
  if (record.task === "door") return doorRate(record) * 100;
  if (record.task === "shm") return Number(record.result.summary.predicted_damage);
  return null;
}

export function finding(record: HistoryRecord): string {
  if (record.task === "door") {
    return `${record.result.summary.abnormal_cycles ?? 0} of ${record.result.summary.cycles ?? 0} cycles abnormal`;
  }
  if (record.task === "acv") return `Car ${record.result.summary.top_car ?? "—"} ranked first`;
  if (record.task === "corrugation") return String(record.result.summary.prediction ?? "—");
  return `Damage ${Number(record.result.summary.predicted_damage).toFixed(4)}`;
}

export const taskLabels: Record<TaskId, string> = {
  door: "Door",
  acv: "ACV",
  corrugation: "Rail",
  shm: "SHM",
};
