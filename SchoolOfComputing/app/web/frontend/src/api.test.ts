import { afterEach, describe, expect, it, vi } from "vitest";
import { extractAssetMetadata, runPrediction } from "./api";
import type { PredictionResponse } from "./types";

const result: PredictionResponse = {
  task: "door",
  task_name: "Door diagnostics",
  mode: "real",
  source_file: "door.csv",
  input_sha256: "abc123",
  model_version: "door:v1:test",
  output_filename: "door_predictions.csv",
  rows: [],
  summary: {},
  visual: {},
  csv_text: "",
  notices: [],
};

afterEach(() => vi.unstubAllGlobals());

describe("production prediction requests", () => {
  it("always requests the approved real model", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(result), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);

    await runPrediction("door", new File(["sample"], "door.csv", { type: "text/csv" }));

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/predict\/door\?mode=real$/),
      expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
    );
  });

  it("returns actionable guidance for oversized requests", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 413 })));

    await expect(runPrediction("door", new File(["sample"], "door.csv"))).rejects.toThrow(
      "The file is too large. Choose a file smaller than 30 MB.",
    );
  });

  it("does not expose a browser network error to the user", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(runPrediction("door", new File(["sample"], "door.csv"))).rejects.toThrow(
      "The analysis service is temporarily unavailable. Try again in a few minutes or contact the RailGuard administrator.",
    );
  });

  it("sends the browser file-modified time for formats without embedded timestamps", async () => {
    const detected = {
      asset_id: "Rail sample Test1",
      component_info: "Bearing sensors",
      measurement_time: "2023-11-14T22:13:20Z",
      asset_source: "filename",
      component_source: "embedded",
      measurement_time_source: "file_modified",
      warnings: [],
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(detected), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["sample"], "Test1.csv", { lastModified: 1_700_000_000_000 });

    await extractAssetMetadata("corrugation", file);

    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringMatching(/\/api\/metadata\/corrugation\?file_modified_ms=1700000000000$/),
      expect.objectContaining({ method: "POST", body: expect.any(FormData) }),
    );
  });
});
