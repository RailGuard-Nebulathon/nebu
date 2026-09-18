"""Classical regression pipelines with exact persisted target transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _estimator(name: str, seed: int) -> Any:
    choices = {
        "ridge": Ridge(alpha=1.0),
        "elastic_net": ElasticNet(alpha=0.01, random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
        "extra_trees": ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
    }
    try:
        return choices[name]
    except KeyError as exc:
        raise ValueError(f"Unknown regressor: {name}") from exc


class ClassicalRegressor:
    def __init__(
        self, name: str = "extra_trees", seed: int = 42, target_transform: str = "raw"
    ) -> None:
        if target_transform not in {"raw", "log", "log1p"}:
            raise ValueError("target_transform must be raw, log, or log1p")
        self.name, self.seed, self.target_transform = name, seed, target_transform
        self.pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", _estimator(name, seed)),
            ]
        )
        self.feature_names_: list[str] = []
        self.residual_quantile_: float | None = None
        self.prediction_scale_: float = 1.0

    def fit(self, x: pd.DataFrame | np.ndarray, y: Any) -> ClassicalRegressor:
        self.feature_names_ = (
            list(x.columns)
            if isinstance(x, pd.DataFrame)
            else [f"feature_{i}" for i in range(np.asarray(x).shape[1])]
        )
        target = np.asarray(y, dtype=float)
        if self.target_transform == "log":
            if np.any(target <= 0):
                raise ValueError("log transform requires positive targets")
            target = np.log(target)
        elif self.target_transform == "log1p":
            if np.any(target < 0):
                raise ValueError("log1p transform requires non-negative targets")
            target = np.log1p(target)
        self.pipeline.fit(x, target)
        return self

    def _validate(self, x: pd.DataFrame | np.ndarray) -> None:
        if isinstance(x, pd.DataFrame) and list(x.columns) != self.feature_names_:
            raise ValueError("Inference feature schema/order is incompatible with the fitted model")

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._validate(x)
        prediction = self.pipeline.predict(x)
        if self.target_transform == "log":
            prediction = np.exp(prediction)
        elif self.target_transform == "log1p":
            prediction = np.expm1(prediction)
        return prediction * self.prediction_scale_

    def calibrate_intervals(
        self, x: pd.DataFrame | np.ndarray, y: Any, coverage: float = 0.9
    ) -> None:
        self.residual_quantile_ = float(
            np.quantile(np.abs(np.asarray(y) - self.predict(x)), coverage)
        )

    def predict_interval(self, x: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.residual_quantile_ is None:
            raise RuntimeError("Intervals must be calibrated on held-out residuals")
        prediction = self.predict(x)
        return prediction - self.residual_quantile_, prediction + self.residual_quantile_

    def metadata(self) -> dict[str, Any]:
        return {
            "kind": "regression",
            "model": self.name,
            "seed": self.seed,
            "target_transform": self.target_transform,
            "prediction_scale": self.prediction_scale_,
            "feature_names": self.feature_names_,
            "checkpoint_version": 1,
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ClassicalRegressor:
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("Bundle does not contain a ClassicalRegressor")
        return loaded
