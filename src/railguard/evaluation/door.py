"""Official one-to-one greedy IoU-weighted Door F1."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _iou(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> float:
    intersection = max(pd.Timedelta(0), min(a_end, b_end) - max(a_start, b_start)).total_seconds()
    union = (a_end - a_start).total_seconds() + (b_end - b_start).total_seconds() - intersection
    return intersection / union if union > 0 else 0.0


def door_iou_weighted_f1(truth: pd.DataFrame, prediction: pd.DataFrame) -> dict[str, Any]:
    candidates = []
    for true_index, true in truth.iterrows():
        for predicted_index, predicted in prediction.iterrows():
            if true["prediction"] != predicted["prediction"]:
                continue
            iou = _iou(pd.Timestamp(true["start_time"]), pd.Timestamp(true["end_time"]), pd.Timestamp(predicted["start_time"]), pd.Timestamp(predicted["end_time"]))
            if iou > 0:
                candidates.append((iou, true_index, predicted_index))
    used_truth, used_prediction, matches = set(), set(), []
    for iou, true_index, predicted_index in sorted(candidates, reverse=True):
        if true_index not in used_truth and predicted_index not in used_prediction:
            used_truth.add(true_index); used_prediction.add(predicted_index); matches.append(iou)
    total = sum(matches)
    recall = total / len(truth) if len(truth) else 0.0
    precision = total / len(prediction) if len(prediction) else 0.0
    score = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"score": score, "soft_recall": recall, "soft_precision": precision, "matches": len(matches), "iou_sum": total}

