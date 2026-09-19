import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ResultPanel } from "./App";
import type { PredictionResponse } from "./types";
import { getUploadError } from "./upload";

const acvResult: PredictionResponse = {
  task: "acv",
  task_name: "ACV leak localisation",
  mode: "demo",
  source_file: "case.xlsx",
  output_filename: "acv_predictions.csv",
  rows: [{ file_id: "case.xlsx", ranked_cars: "03|06|01" }],
  summary: { top_car: "03", cars_ranked: 3 },
  visual: { ranking: [{ car: "03", score: 0.42 }, { car: "06", score: 0.17 }, { car: "01", score: 0.12 }] },
  csv_text: "file_id,ranked_cars\ncase.xlsx,03|06|01\n",
  notices: ["Demonstration result only."],
};

const doorResult: PredictionResponse = {
  task: "door",
  task_name: "Door diagnostics",
  mode: "real",
  source_file: "door.csv",
  output_filename: "door_predictions.csv",
  rows: [{ start_time: "start-1", end_time: "end-1", prediction: "Normal" }],
  summary: { cycles: 1, abnormal_cycles: 0, confidence: 0.913 },
  visual: { segments: [{ start_time: "start-1", end_time: "end-1", prediction: "Normal", confidence: 0.913 }] },
  csv_text: "start_time,end_time,prediction\nstart-1,end-1,Normal\n",
  notices: [],
};

describe("ResultPanel progressive disclosure", () => {
  it("defaults to a complete quick decision sourced from a playbook", () => {
    render(<ResultPanel result={acvResult} onReset={() => undefined} />);

    expect(screen.getByRole("heading", { name: /Car 03 is the first refrigerant-leak inspection candidate/i })).toBeInTheDocument();
    expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
    expect(screen.getByText(/Suggested checks come from the configured playbook/i)).toBeInTheDocument();
    expect(screen.getByLabelText("Train car inspection map")).toBeInTheDocument();
  });

  it("separates why, technical evidence and learning content", async () => {
    const user = userEvent.setup();
    render(<ResultPanel result={acvResult} onReset={() => undefined} />);

    await user.click(screen.getByRole("button", { name: "Why this result?" }));
    expect(screen.getByRole("heading", { name: "Why this result?" })).toBeInTheDocument();
    expect(screen.getByText(/do not establish physical cause/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Technical evidence" }));
    expect(screen.getByText("Fault ranking")).toBeInTheDocument();
    expect(screen.getByText("Top-two margin")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Learn" }));
    expect(screen.getByRole("heading", { name: "How to read this result" })).toBeInTheDocument();
    expect(screen.getByText("Use official limits only")).toBeInTheDocument();
  });

  it("shows Door confidence per cycle in technical evidence", async () => {
    const user = userEvent.setup();
    render(<ResultPanel result={doorResult} onReset={() => undefined} />);

    expect(screen.getByText(/lowest predicted-cycle confidence is 91.3%/i)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Technical evidence" }));
    expect(screen.getByRole("columnheader", { name: "Model confidence" })).toBeInTheDocument();
    expect(screen.getByText("91.3%")).toBeInTheDocument();
  });
});

describe("production upload safeguards", () => {
  it("accepts files at the documented limit", () => {
    expect(getUploadError({ size: 30 * 1024 * 1024 })).toBeNull();
  });

  it("explains how to fix an oversized upload", () => {
    expect(getUploadError({ size: 30 * 1024 * 1024 + 1 })).toBe(
      "The file is too large. Choose a file smaller than 30 MB.",
    );
  });
});
