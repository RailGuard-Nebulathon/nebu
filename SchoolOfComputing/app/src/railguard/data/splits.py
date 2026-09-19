"""Sample-level stratified/group-aware split construction."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold


def classification_splits(y: np.ndarray, groups: np.ndarray | None = None, folds: int = 5, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        return StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, groups)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def regression_splits(y: np.ndarray, groups: np.ndarray | None = None, folds: int = 5, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if groups is not None:
        return GroupKFold(n_splits=folds).split(np.zeros(len(y)), y, groups)
    return KFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)

