import type { PredictionResponse, TaskId } from "./types";

export type DecisionStatus = "Healthy" | "Monitor" | "Inspect Soon" | "High Priority";
export type Reliability = "Reliable" | "Review advised" | "Insufficient evidence";

export interface PlaybookEntry {
  status: DecisionStatus;
  urgency: string;
  checks: string[];
}

export interface DecisionPlaybook {
  door: { normal: PlaybookEntry; abnormal: PlaybookEntry };
  acv: { candidate: PlaybookEntry };
  corrugation: { normal: PlaybookEntry; detected: PlaybookEntry };
  shm: { estimate: PlaybookEntry };
}

export interface DecisionViewModel {
  status: DecisionStatus;
  location: string;
  finding: string;
  urgency: string;
  nextChecks: string[];
  reliability: Reliability;
  reliabilityReason: string;
  evidence: string[];
  playbookLabel: string;
}

export const DEFAULT_DECISION_PLAYBOOK: DecisionPlaybook = {
  door: {
    normal: {
      status: "Healthy",
      urgency: "Continue routine monitoring",
      checks: ["Confirm that all expected door cycles were captured.", "Continue the locally approved inspection schedule."],
    },
    abnormal: {
      status: "Inspect Soon",
      urgency: "Review before the next scheduled maintenance release",
      checks: ["Review the flagged cycle windows.", "Compare current and position traces with a normal cycle.", "Escalate under the local procedure if the pattern persists."],
    },
  },
  acv: {
    candidate: {
      status: "Inspect Soon",
      urgency: "Confirm during the next approved ACV inspection opportunity",
      checks: ["Inspect the first-ranked car.", "Inspect the second-ranked car if the first check is inconclusive.", "Compare the candidate trace with the peer median when available."],
    },
  },
  corrugation: {
    normal: {
      status: "Healthy",
      urgency: "Continue routine monitoring",
      checks: ["Confirm the uploaded segment and operating context.", "Continue the locally approved inspection schedule."],
    },
    detected: {
      status: "Inspect Soon",
      urgency: "Engineering review advised",
      checks: ["Confirm the source segment and side metadata.", "Review vibration and frequency evidence.", "Follow the locally approved track-inspection procedure."],
    },
  },
  shm: {
    estimate: {
      status: "Monitor",
      urgency: "Interpret against approved engineering limits",
      checks: ["Compare the estimate and interval with approved limits.", "Review the stress history and influential intervals.", "Do not interpret the raw value as a percentage or an automatic maintenance threshold."],
    },
  },
};

const CONFIDENCE_REVIEW_THRESHOLD = 0.8;
const ACV_CLOSE_MARGIN = 0.1;

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function reliabilityFor(result: PredictionResponse): Pick<DecisionViewModel, "reliability" | "reliabilityReason"> {
  if (result.mode === "demo") {
    return { reliability: "Insufficient evidence", reliabilityReason: "This result is simulated and cannot support an operational decision." };
  }
  if (result.task === "acv") {
    const ranking = result.visual.ranking ?? [];
    if (ranking.length < 2) return { reliability: "Insufficient evidence", reliabilityReason: "A top-two ranking margin was not available." };
    const margin = ranking[0].score - ranking[1].score;
    return margin < ACV_CLOSE_MARGIN
      ? { reliability: "Review advised", reliabilityReason: `The two leading cars have a close model-score margin of ${margin.toFixed(2)}.` }
      : { reliability: "Reliable", reliabilityReason: `The leading car is separated by a model-score margin of ${margin.toFixed(2)}.` };
  }
  const confidence = numeric(result.summary.confidence);
  if (confidence !== null) {
    return confidence >= CONFIDENCE_REVIEW_THRESHOLD
      ? { reliability: "Reliable", reliabilityReason: `${(confidence * 100).toFixed(1)}% model confidence; this is not a safety probability.` }
      : { reliability: "Review advised", reliabilityReason: `${(confidence * 100).toFixed(1)}% model confidence is below the display policy's 80% review threshold.` };
  }
  const uncertainty = numeric(result.summary.uncertainty);
  if (uncertainty !== null) return { reliability: "Review advised", reliabilityReason: `Model uncertainty is ${uncertainty.toFixed(4)}; compare the interval with approved limits.` };
  return { reliability: "Insufficient evidence", reliabilityReason: "Confidence or uncertainty was not reported; review the technical evidence." };
}

function doorDecision(result: PredictionResponse, playbook: DecisionPlaybook): Omit<DecisionViewModel, "reliability" | "reliabilityReason"> {
  const abnormal = Number(result.summary.abnormal_cycles ?? 0);
  const cycles = Number(result.summary.cycles ?? 0);
  const entry = abnormal > 0 ? playbook.door.abnormal : playbook.door.normal;
  return {
    ...entry,
    location: String(result.summary.door_id ?? result.summary.component ?? "Uploaded door system"),
    finding: abnormal > 0 ? `Abnormal resistance detected in ${abnormal} of ${cycles} door cycles.` : `No abnormal resistance detected across ${cycles} door cycles.`,
    nextChecks: entry.checks,
    evidence: abnormal > 0 ? [`${abnormal} cycle ${abnormal === 1 ? "was" : "were"} classified as abnormal resistance.`, "The engineering view lists each flagged start and end time."] : [`All ${cycles} detected cycles were classified as normal.`],
    playbookLabel: "Door response playbook",
  };
}

function acvDecision(result: PredictionResponse, playbook: DecisionPlaybook): Omit<DecisionViewModel, "reliability" | "reliabilityReason"> {
  const ranking = result.visual.ranking ?? [];
  const first = String(result.summary.top_car ?? ranking[0]?.car ?? "—").padStart(2, "0");
  const second = ranking[1]?.car ? String(ranking[1].car).padStart(2, "0") : null;
  const entry = playbook.acv.candidate;
  return {
    ...entry,
    location: `ACV system · Car ${first}`,
    finding: `Car ${first} is the first refrigerant-leak inspection candidate.`,
    nextChecks: entry.checks.map((check, index) => index === 0 ? `Inspect the ACV system on Car ${first}.` : index === 1 && second ? `Inspect Car ${second} if the first check is inconclusive.` : check),
    evidence: [`Car ${first} ranked first among ${result.summary.cars_ranked ?? ranking.length} cars.`, ranking.length > 1 ? `Its model-score lead over Car ${second} is ${(ranking[0].score - ranking[1].score).toFixed(2)}.` : "A comparison with a second candidate was unavailable."],
    playbookLabel: "ACV inspection playbook",
  };
}

function corrugationDecision(result: PredictionResponse, playbook: DecisionPlaybook): Omit<DecisionViewModel, "reliability" | "reliabilityReason"> {
  const prediction = String(result.summary.prediction ?? "Unknown");
  const normal = prediction === "Normal";
  const entry = normal ? playbook.corrugation.normal : playbook.corrugation.detected;
  return {
    ...entry,
    location: String(result.summary.side ?? result.summary.segment ?? "Uploaded rail-vibration segment"),
    finding: normal ? "No corrugation pattern was detected." : `${prediction} corrugation pattern detected.`,
    nextChecks: entry.checks,
    evidence: [`${prediction} received the highest model probability.`, ...(normal ? [] : ["Frequency and channel evidence should be reviewed before action."])],
    playbookLabel: "Rail inspection playbook",
  };
}

function shmDecision(result: PredictionResponse, playbook: DecisionPlaybook): Omit<DecisionViewModel, "reliability" | "reliabilityReason"> {
  const damage = numeric(result.summary.predicted_damage);
  const interval = result.visual.interval;
  const entry = playbook.shm.estimate;
  return {
    ...entry,
    location: String(result.summary.component ?? "Uploaded structural component"),
    finding: damage === null ? "A cumulative damage estimate was not available." : `Estimated cumulative damage is ${damage.toFixed(4)}.`,
    nextChecks: entry.checks,
    evidence: [damage === null ? "No numeric estimate was returned." : `The model estimate is ${damage.toFixed(4)}.`, interval ? `The indicative interval is ${interval[0].toFixed(4)}–${interval[1].toFixed(4)}.` : "No prediction interval was supplied."],
    playbookLabel: "Structural review playbook",
  };
}

const BUILDERS: Record<TaskId, (result: PredictionResponse, playbook: DecisionPlaybook) => Omit<DecisionViewModel, "reliability" | "reliabilityReason">> = {
  door: doorDecision,
  acv: acvDecision,
  corrugation: corrugationDecision,
  shm: shmDecision,
};

export function buildDecision(result: PredictionResponse, playbook: DecisionPlaybook = DEFAULT_DECISION_PLAYBOOK): DecisionViewModel {
  return { ...BUILDERS[result.task](result, playbook), ...reliabilityFor(result) };
}
