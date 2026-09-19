"""Grouped ACV candidate classifier and ranking evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from railguard.models.classical import ClassicalClassifier


def rank_candidates(candidates: pd.DataFrame, scores: np.ndarray) -> list[str]:
    order = np.argsort(-np.asarray(scores), kind="stable")
    return candidates.iloc[order]["car_id"].astype(str).str.zfill(2).tolist()


def evaluate_leave_one_case_out(candidates: pd.DataFrame) -> list[dict[str, object]]:
    feature_columns = [c for c in candidates if c not in {"case_id", "car_id", "is_faulty"}]
    reports = []
    for case_id in candidates["case_id"].unique():
        train = candidates[candidates["case_id"] != case_id]
        validation = candidates[candidates["case_id"] == case_id]
        model = ClassicalClassifier("logreg", class_weight="balanced").fit(train[feature_columns], train["is_faulty"])
        scores = model.predict_proba(validation[feature_columns])[:, list(model.pipeline.named_steps["model"].classes_).index(True)]
        ranking = rank_candidates(validation, scores)
        truth = validation.loc[validation["is_faulty"], "car_id"].iloc[0]
        rank = ranking.index(truth) + 1
        reports.append({"case_id": case_id, "ranking": ranking, "true_rank": rank, "rank_score": (len(ranking) - (rank - 1)) / len(ranking), "confidence_margin": float(np.sort(scores)[-1] - np.sort(scores)[-2])})
    return reports

