"""Imbalance-aware corrugation baseline helpers."""

import numpy as np
import pandas as pd

from railguard.evaluation.classification import classification_metrics
from railguard.models.classical import ClassicalClassifier


def fit_evaluate_corrugation(train_x: pd.DataFrame, train_y: np.ndarray, validation_x: pd.DataFrame, validation_y: np.ndarray):
    model = ClassicalClassifier("extra_trees", class_weight="balanced").fit(train_x, train_y)
    prediction = model.predict(validation_x)
    probabilities = model.predict_proba(validation_x)
    labels = list(model.pipeline.named_steps["model"].classes_)
    report = classification_metrics(validation_y, prediction, probabilities, labels)
    report["primary_metric"] = report["macro_f1"]
    return model, report

