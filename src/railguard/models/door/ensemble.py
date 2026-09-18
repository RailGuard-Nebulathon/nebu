"""Serializable probability ensemble for Door operation classification."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from railguard.models.classical import ClassicalClassifier


class DoorEnsemble:
    """Average three independently fitted classifiers with aligned class columns."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.models = [
            ClassicalClassifier(
                "extra_trees",
                seed=seed,
                class_weight="balanced",
                estimator_params={"n_estimators": 750, "max_features": "sqrt", "n_jobs": -1},
            ),
            ClassicalClassifier(
                "random_forest",
                seed=seed + 1,
                class_weight="balanced",
                estimator_params={"n_estimators": 750, "max_features": "sqrt", "n_jobs": -1},
            ),
            ClassicalClassifier(
                "hist_gradient_boosting",
                seed=seed + 2,
                class_weight="balanced",
                estimator_params={
                    "max_iter": 300,
                    "learning_rate": 0.05,
                    "max_leaf_nodes": 15,
                    "l2_regularization": 1.0,
                },
            ),
        ]
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray = np.asarray([], dtype=object)

    def fit(self, x: pd.DataFrame, y: Any) -> "DoorEnsemble":
        self.feature_names_ = list(x.columns)
        for model in self.models:
            model.fit(x, y)
        class_sets = [tuple(model.classes_) for model in self.models]
        if len(set(class_sets)) != 1:
            raise ValueError(f"Door ensemble class mismatch: {class_sets}")
        self.classes_ = np.asarray(class_sets[0])
        return self

    def _validate(self, x: pd.DataFrame) -> None:
        if list(x.columns) != self.feature_names_:
            raise ValueError("Inference feature schema/order is incompatible with Door ensemble")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        self._validate(x)
        return np.mean([model.predict_proba(x) for model in self.models], axis=0)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(x)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "door_probability_ensemble",
            "model": "extra_trees_random_forest_hist_gradient_boosting",
            "seed": self.seed,
            "members": [model.metadata() for model in self.models],
            "feature_names": self.feature_names_,
            "classes": self.classes_.tolist(),
            "checkpoint_version": 1,
        }


__all__ = ["DoorEnsemble"]
