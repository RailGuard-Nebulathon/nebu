"""Shared typed records used at adapter and inference boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SequenceSample:
    sample_id: str
    values: np.ndarray
    timestamps: np.ndarray | None
    channel_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
    target: Any | None = None


@dataclass(slots=True)
class Prediction:
    sample_id: str
    task: str
    prediction: Any
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    uncertainty: float | None = None
    ood_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Explanation:
    sample_id: str
    feature_importance: dict[str, float] | None = None
    channel_importance: dict[str, float] | None = None
    temporal_importance: np.ndarray | None = None
    spectral_importance: dict[str, float] | None = None
    neighbours: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

