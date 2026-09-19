"""Ensemble predictions and disagreement."""

import numpy as np


def classification_ensemble(probabilities: list[np.ndarray], weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(probabilities)
    weights = np.ones(len(stack)) / len(stack) if weights is None else np.asarray(weights) / np.sum(weights)
    mean = np.average(stack, axis=0, weights=weights)
    uncertainty = np.mean(np.std(stack, axis=0), axis=-1)
    return mean, uncertainty


def regression_ensemble(predictions: list[np.ndarray], validation_errors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(predictions)
    weights = None if validation_errors is None else 1 / np.maximum(np.asarray(validation_errors), 1e-9)
    if weights is not None:
        weights /= weights.sum()
    return np.average(stack, axis=0, weights=weights), np.std(stack, axis=0)

