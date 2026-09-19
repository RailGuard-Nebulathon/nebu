import { describe, expect, it } from "vitest";
import { buildDecision, DEFAULT_DECISION_PLAYBOOK } from "./decision";
import type { DecisionPlaybook } from "./decision";
import type { PredictionResponse, TaskId } from "./types";

function result(task: TaskId, overrides: Partial<PredictionResponse> = {}): PredictionResponse {
  return {
    task,
    task_name: task,
    mode: "real",
    source_file: "sample.csv",
    output_filename: "prediction.csv",
    rows: [],
    summary: {},
    visual: {},
    csv_text: "file_id,prediction\n",
    notices: [],
    ...overrides,
  };
}

describe("buildDecision", () => {
  it("provides status, exact reported location, finding, urgency and playbook checks", () => {
    const decision = buildDecision(result("door", { summary: { cycles: 5, abnormal_cycles: 2, door_id: "Door 04" } }));

    expect(decision.status).toBe("Inspect Soon");
    expect(decision.location).toBe("Door 04");
    expect(decision.finding).toContain("2 of 5");
    expect(decision.urgency).toContain("scheduled maintenance release");
    expect(decision.nextChecks).toEqual(DEFAULT_DECISION_PLAYBOOK.door.abnormal.checks);
  });

  it("supports an approved custom playbook including High Priority", () => {
    const playbook: DecisionPlaybook = structuredClone(DEFAULT_DECISION_PLAYBOOK);
    playbook.door.abnormal = { status: "High Priority", urgency: "Apply depot procedure RG-4 now", checks: ["Use approved checklist RG-4."] };

    const decision = buildDecision(result("door", { summary: { cycles: 1, abnormal_cycles: 1 } }), playbook);

    expect(decision.status).toBe("High Priority");
    expect(decision.urgency).toBe("Apply depot procedure RG-4 now");
    expect(decision.nextChecks).toEqual(["Use approved checklist RG-4."]);
  });

  it("uses the lowest Door cycle confidence as a conservative reliability summary", () => {
    const decision = buildDecision(result("door", { summary: { cycles: 2, abnormal_cycles: 1, confidence: 0.81 } }));

    expect(decision.reliability).toBe("Reliable");
    expect(decision.reliabilityReason).toContain("lowest predicted-cycle confidence is 81.0%");
    expect(decision.reliabilityReason).toContain("not a safety probability");
  });

  it("maps ACV separation to Reliable and close candidates to Review advised", () => {
    const clear = buildDecision(result("acv", { summary: { top_car: "03", cars_ranked: 2 }, visual: { ranking: [{ car: "03", score: 0.7 }, { car: "06", score: 0.2 }] } }));
    const close = buildDecision(result("acv", { summary: { top_car: "03", cars_ranked: 2 }, visual: { ranking: [{ car: "03", score: 0.42 }, { car: "06", score: 0.37 }] } }));

    expect(clear.reliability).toBe("Reliable");
    expect(close.reliability).toBe("Review advised");
    expect(close.nextChecks[1]).toContain("Car 06");
  });

  it("always labels demo output as Insufficient evidence", () => {
    const decision = buildDecision(result("corrugation", { mode: "demo", summary: { prediction: "Side II", confidence: 0.99, side: "Side II" } }));

    expect(decision.reliability).toBe("Insufficient evidence");
    expect(decision.reliabilityReason).toContain("simulated");
    expect(decision.location).toBe("Side II");
  });

  it("does not invent an SHM threshold or percentage", () => {
    const decision = buildDecision(result("shm", { summary: { predicted_damage: 0.91, uncertainty: 0.04, component: "Bogie A" }, visual: { interval: [0.87, 0.95] } }));

    expect(decision.status).toBe("Monitor");
    expect(decision.location).toBe("Bogie A");
    expect(decision.urgency).toContain("approved engineering limits");
    expect(decision.nextChecks.join(" ")).toContain("Do not interpret");
  });
});
