"""Generic averaging and OOF-only stacking guard."""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from railguard.uncertainty.ensembles import classification_ensemble, regression_ensemble


class OOFStacker:
    def __init__(self, regression: bool = False) -> None:
        self.model = Ridge() if regression else LogisticRegression(max_iter=1000)

    def fit(self, oof_predictions: np.ndarray, targets: np.ndarray, fold_ids: np.ndarray) -> "OOFStacker":
        if len(np.unique(fold_ids)) < 2:
            raise ValueError("Stacking requires out-of-fold predictions from at least two folds")
        self.model.fit(oof_predictions, targets)
        return self


__all__ = ["OOFStacker", "classification_ensemble", "regression_ensemble"]

