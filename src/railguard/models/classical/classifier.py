"""Leakage-safe scikit-learn classification pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _estimator(
    name: str,
    seed: int,
    class_weight: str | dict[str, float] | None,
    estimator_params: dict[str, Any],
) -> Any:
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000, class_weight=class_weight, random_state=seed, **estimator_params
        )
    if name == "random_forest":
        return RandomForestClassifier(
            **({"n_estimators": 200, "n_jobs": 1} | estimator_params),
            class_weight=class_weight,
            random_state=seed,
        )
    if name == "extra_trees":
        return ExtraTreesClassifier(
            **({"n_estimators": 200, "n_jobs": 1} | estimator_params),
            class_weight=class_weight,
            random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            class_weight=class_weight, random_state=seed, **estimator_params
        )
    raise ValueError(f"Unknown classifier: {name}")


class ClassicalClassifier:
    def __init__(
        self,
        name: str = "extra_trees",
        seed: int = 42,
        class_weight: str | dict[str, float] | None = "balanced",
        estimator_params: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.seed = seed
        self.class_weight = class_weight
        self.estimator_params = estimator_params or {}
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", _estimator(name, seed, class_weight, self.estimator_params)),
            ]
        )
        self.feature_names_: list[str] = []

    @property
    def classes_(self) -> np.ndarray:
        return np.asarray(getattr(self.pipeline.named_steps["model"], "classes_", []))

    def fit(self, x: pd.DataFrame | np.ndarray, y: Any) -> ClassicalClassifier:
        self.feature_names_ = (
            list(x.columns)
            if isinstance(x, pd.DataFrame)
            else [f"feature_{i}" for i in range(np.asarray(x).shape[1])]
        )
        self.pipeline.fit(x, y)
        return self

    def _validate(self, x: pd.DataFrame | np.ndarray) -> None:
        if isinstance(x, pd.DataFrame) and list(x.columns) != self.feature_names_:
            raise ValueError("Inference feature schema/order is incompatible with the fitted model")

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._validate(x)
        return self.pipeline.predict(x)

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._validate(x)
        model = self.pipeline.named_steps["model"]
        if hasattr(model, "predict_proba"):
            return self.pipeline.predict_proba(x)
        decision = self.pipeline.decision_function(x)
        decision = np.column_stack([-decision, decision]) if decision.ndim == 1 else decision
        shifted = decision - decision.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def metadata(self) -> dict[str, Any]:
        model = self.pipeline.named_steps["model"]
        return {
            "kind": "classification",
            "model": self.name,
            "seed": self.seed,
            "class_weight": self.class_weight,
            "estimator_params": self.estimator_params,
            "feature_names": self.feature_names_,
            "classes": list(getattr(model, "classes_", [])),
            "checkpoint_version": 1,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ClassicalClassifier:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("Bundle does not contain a ClassicalClassifier")
        return loaded
