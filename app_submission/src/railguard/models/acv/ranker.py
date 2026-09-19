"""Small-data ACV ranker combining supervised and within-case anomaly evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from railguard.models.classical import ClassicalClassifier


class ACVRanker:
    """Rank cars without trusting six labelled cases to fit a large network.

    Inference is performed one case at a time.  Both the supervised probability and
    the robust peer-deviation score are converted to within-case percentile ranks so
    their scales remain comparable across the two vendor schemas.
    """

    def __init__(
        self, model: str = "logreg", supervised_weight: float = 0.5, seed: int = 42
    ) -> None:
        if not 0.0 <= supervised_weight <= 1.0:
            raise ValueError("supervised_weight must be between zero and one")
        self.model_name = model
        self.supervised_weight = supervised_weight
        self.seed = seed
        self.classifier = ClassicalClassifier(model, seed=seed, class_weight="balanced")
        self.feature_names_: list[str] = []

    @property
    def pipeline(self):
        return self.classifier.pipeline

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray([False, True])

    def fit(self, x: pd.DataFrame, y: Any) -> ACVRanker:
        self.feature_names_ = list(x.columns)
        self.classifier.fit(x, y)
        return self

    @staticmethod
    def _percentile(values: np.ndarray) -> np.ndarray:
        return (
            pd.Series(np.asarray(values, dtype=float)).rank(method="average", pct=True).to_numpy()
        )

    def _rule_score(self, x: pd.DataFrame) -> np.ndarray:
        columns = [
            column
            for column in x
            if column.startswith("peer_residual__")
            and not column.endswith(("parameter_count", "missing_fraction"))
        ]
        if not columns:
            return np.zeros(len(x), dtype=float)
        matrix = x[columns].to_numpy(dtype=float)
        scale = np.nanmedian(np.abs(matrix), axis=0) * 1.4826
        usable = np.isfinite(scale) & (scale > 1e-9)
        if not usable.any():
            return np.zeros(len(x), dtype=float)
        return np.nanmedian(np.abs(matrix[:, usable]) / scale[usable], axis=1)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        if list(x.columns) != self.feature_names_:
            raise ValueError(
                "Inference feature schema/order is incompatible with the fitted ACV ranker"
            )
        base = self.classifier.predict_proba(x)
        positive_index = list(self.classifier.pipeline.named_steps["model"].classes_).index(True)
        supervised = self._percentile(base[:, positive_index])
        anomaly = self._percentile(self._rule_score(x))
        score = self.supervised_weight * supervised + (1.0 - self.supervised_weight) * anomaly
        score = np.clip(score, 0.0, 1.0)
        return np.column_stack([1.0 - score, score])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(x)[:, 1] >= 0.5

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "acv_ranking",
            "model": self.model_name,
            "seed": self.seed,
            "supervised_weight": self.supervised_weight,
            "feature_names": self.feature_names_,
            "classes": [False, True],
            "checkpoint_version": 1,
        }
