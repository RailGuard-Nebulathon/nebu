"""JSON-safe classification reports."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import label_binarize


def classification_metrics(y_true: Any, y_pred: Any, probabilities: np.ndarray | None = None, labels: list[Any] | None = None) -> dict[str, Any]:
    labels = labels or sorted(set(y_true) | set(y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    report: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {str(label): {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if probabilities is not None and len(set(y_true)) > 1:
        binary = label_binarize(y_true, classes=labels)
        if len(labels) == 2:
            binary = binary.reshape(-1)
            report["auroc"] = float(roc_auc_score(binary, probabilities[:, 1]))
            report["auprc"] = float(average_precision_score(binary, probabilities[:, 1]))
        else:
            report["auroc_ovr_macro"] = float(roc_auc_score(binary, probabilities, average="macro", multi_class="ovr"))
            report["auprc_macro"] = float(average_precision_score(binary, probabilities, average="macro"))
    return report

