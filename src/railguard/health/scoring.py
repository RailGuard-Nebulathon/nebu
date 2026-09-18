"""Transparent decision-support health scoring (not an approved maintenance rule)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HealthWeights:
    fault: float = 0.5
    damage: float = 0.3
    anomaly: float = 0.2
    conservative_prior: float = 0.5


def health_score(fault_risk: float = 0.0, damage_risk: float = 0.0, anomaly_risk: float = 0.0, confidence: float = 1.0, ood: bool = False, weights: HealthWeights | None = None) -> float:
    weights = weights or HealthWeights()
    total = weights.fault + weights.damage + weights.anomaly
    risk = (weights.fault * fault_risk + weights.damage * damage_risk + weights.anomaly * anomaly_risk) / max(total, 1e-9)
    adjusted = risk * confidence + weights.conservative_prior * (1 - confidence)
    if ood:
        adjusted = max(adjusted, weights.conservative_prior)
    return float(100 * (1 - np.clip(adjusted, 0, 1)))

