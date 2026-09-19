import { describe, expect, it } from "vitest";
import { doorRate, finding, trendValue } from "./history";
import type { HistoryRecord } from "./types";

function record(task: HistoryRecord["task"], summary: HistoryRecord["result"]["summary"]): HistoryRecord {
  return {
    id: "one",
    asset_id: "Train-04",
    component_info: "Door 2L",
    measurement_time: "2026-09-19T01:00:00Z",
    analysis_time: "2026-09-19T01:01:00Z",
    task,
    mode: "real",
    source_file: "input.csv",
    input_sha256: "abc",
    model_version: "model:v1",
    result: {
      task,
      task_name: task,
      mode: "real",
      source_file: "input.csv",
      input_sha256: "abc",
      model_version: "model:v1",
      output_filename: "output.csv",
      rows: [],
      summary,
      visual: {},
      csv_text: "",
      notices: [],
    },
  };
}

describe("history trend derivation", () => {
  it("calculates Door abnormal percentage from immutable summary values", () => {
    const item = record("door", { cycles: 40, abnormal_cycles: 10 });
    expect(doorRate(item)).toBe(0.25);
    expect(trendValue(item)).toBe(25);
    expect(finding(item)).toBe("10 of 40 cycles abnormal");
  });

  it("uses SHM damage as the numeric trend", () => {
    expect(trendValue(record("shm", { predicted_damage: 0.1234 }))).toBe(0.1234);
  });
});
