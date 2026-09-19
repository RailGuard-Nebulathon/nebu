"""Tree and model-agnostic feature evidence."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance


def feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    estimator = getattr(model, "pipeline", model)
    estimator = getattr(estimator, "named_steps", {}).get("model", estimator)
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = np.abs(getattr(estimator, "coef_", np.zeros((1, len(feature_names))))).mean(axis=0)
    return dict(sorted(zip(feature_names, map(float, values), strict=True), key=lambda item: abs(item[1]), reverse=True))


def permutation_feature_importance(model: Any, x: Any, y: Any, feature_names: list[str], scoring: str | None = None, repeats: int = 5, seed: int = 42) -> dict[str, float]:
    result = permutation_importance(model, x, y, scoring=scoring, n_repeats=repeats, random_state=seed, n_jobs=1)
    return dict(sorted(zip(feature_names, map(float, result.importances_mean), strict=True), key=lambda item: abs(item[1]), reverse=True))

