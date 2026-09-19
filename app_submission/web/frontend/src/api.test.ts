import { afterEach, describe, expect, it, vi } from "vitest";
import { runPrediction } from "./api";
import type { PredictionResponse } from "./types";

const result: PredictionResponse = {
  task: "door",
  task_name: "Door diagnostics",
  mode: "real",
  source_file: "door.csv",
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
});
