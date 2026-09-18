import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ResultPanel } from "./App";
import type { PredictionResponse } from "./types";

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
});
