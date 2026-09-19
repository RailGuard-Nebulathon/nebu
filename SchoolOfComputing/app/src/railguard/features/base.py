"""Deterministic feature extractor protocol and composition."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from railguard.features.spectral import spectral_features
from railguard.features.statistical import statistical_features
from railguard.features.temporal import temporal_features
from railguard.types import SequenceSample


class FeatureExtractor(Protocol):
    def fit(self, samples: Sequence[SequenceSample], y: object = None) -> "FeatureExtractor": ...
    def transform(self, samples: Sequence[SequenceSample]) -> pd.DataFrame: ...


class CombinedFeatureExtractor:
    def __init__(self, sampling_rate: float = 1.0) -> None:
        self.sampling_rate = sampling_rate
        self.feature_names_: list[str] = []

    def fit(self, samples: Sequence[SequenceSample], y: object = None) -> "CombinedFeatureExtractor":
        if samples:
            self.feature_names_ = list(self._one(samples[0]))
        return self

    def _one(self, sample: SequenceSample) -> dict[str, float]:
        output: dict[str, float] = {}
        for index, name in enumerate(sample.channel_names):
            values = sample.values[:, index]
            for family in (statistical_features(values), temporal_features(values), spectral_features(values, self.sampling_rate)):
                output.update({f"{name}__{key}": float(value) for key, value in family.items()})
        return output

    def transform(self, samples: Sequence[SequenceSample]) -> pd.DataFrame:
        rows = [self._one(sample) for sample in samples]
        frame = pd.DataFrame(rows, index=[sample.sample_id for sample in samples])
        ordered = sorted(frame.columns)
        frame = frame.reindex(columns=ordered).replace([np.inf, -np.inf], np.nan)
        if frame.isna().any().any():
            raise ValueError("Feature extraction produced NaN/Inf; clean input or configure handling")
        self.feature_names_ = ordered
        return frame

    def fit_transform(self, samples: Sequence[SequenceSample], y: object = None) -> pd.DataFrame:
        self.fit(samples, y)
        return self.transform(samples)

