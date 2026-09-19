"""Metric-aligned training and cross-validation for all four PS3 tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from railguard.data.adapters.acv import ACVAdapter
from railguard.data.adapters.corrugation import CorrugationAdapter
from railguard.data.adapters.door import DoorAdapter
from railguard.data.adapters.shm import SHMAdapter
from railguard.evaluation import classification_metrics, regression_metrics
from railguard.features.acv import acv_candidate_features
from railguard.features.corrugation import corrugation_features
from railguard.features.door import door_feature_table
from railguard.features.shm import shm_features
from railguard.models.acv import ACVRanker, rank_candidates
from railguard.models.classical import ClassicalClassifier, ClassicalRegressor
from railguard.models.door import DoorEnsemble
from railguard.models.serialization import save_classical_bundle

IDENTIFIER_COLUMNS = {"sample_id", "case_id", "car_id", "target"}


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _cache_frame(path: Path | None, builder) -> pd.DataFrame:
    if path is not None and path.is_file():
        return pd.read_parquet(path)
    frame = builder()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return frame


def build_acv_candidates(raw_root: Path) -> pd.DataFrame:
    adapter = ACVAdapter(raw_root / "ACV")
    labels = (
        pd.read_csv(adapter.root / "Train_Labels.csv", dtype={"faulty_car": str})
        .set_index("filename")["faulty_car"]
        .str.zfill(2)
    )
    return pd.concat(
        [
            acv_candidate_features(pd.read_excel(path), path.name, labels[path.name])
            for path in adapter.files("train")
        ],
        ignore_index=True,
    )


def build_file_features(task: str, raw_root: Path) -> pd.DataFrame:
    adapter = (
        CorrugationAdapter(raw_root / "Rail_Corrugation")
        if task == "corrugation"
        else SHMAdapter(raw_root / "SHM")
    )
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(adapter.samples("train"), 1):
        features = corrugation_features(sample) if task == "corrugation" else shm_features(sample)
        rows.append({"sample_id": sample.sample_id, "target": sample.target, **features})
        if index % 20 == 0:
            print(f"{task}: extracted {index} files", flush=True)
    return pd.DataFrame(rows)


def build_door_features(raw_root: Path) -> pd.DataFrame:
    samples = list(DoorAdapter(raw_root / "Door").samples("train"))
    frame = door_feature_table(samples)
    frame.insert(0, "sample_id", [sample.sample_id for sample in samples])
    frame.insert(1, "operation", [str(sample.metadata["operation"]) for sample in samples])
    frame.insert(2, "target", [str(sample.target) for sample in samples])
    return frame


def train_door(
    raw_root: Path, output_dir: Path, cache_dir: Path | None, folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    """Cross-validate and persist the exact ensemble used for Door inference."""
    cache = cache_dir / "door_features.parquet" if cache_dir else None
    frame = _cache_frame(cache, lambda: build_door_features(raw_root))
    feature_columns = [
        column for column in frame if column not in IDENTIFIER_COLUMNS | {"operation"}
    ]
    x = frame[feature_columns]
    y = frame["target"].astype(str).to_numpy()
    operations = frame["operation"].astype(str).to_numpy()
    strata = np.char.add(y, np.char.add("__", operations))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    probabilities = np.zeros((len(y), len(np.unique(y))), dtype=float)
    fold_scores = []
    classes: list[str] | None = None
    for train_index, validation_index in splitter.split(x, strata):
        model = DoorEnsemble(seed=seed).fit(x.iloc[train_index], y[train_index])
        prediction = model.predict(x.iloc[validation_index])
        fold_probability = model.predict_proba(x.iloc[validation_index])
        oof[validation_index] = prediction
        probabilities[validation_index] = fold_probability
        classes = [str(label) for label in model.classes_]
        fold_scores.append(classification_metrics(y[validation_index], prediction)["macro_f1"])
    metrics = classification_metrics(y, oof, probabilities, classes)
    final = DoorEnsemble(seed=seed).fit(x, y)
    save_classical_bundle(
        output_dir,
        final,
        {
            "selection": "stratified_operation_status_cv_macro_f1",
            "folds": folds,
            "probability_ensemble": "uniform_mean",
        },
        {"feature_names": feature_columns},
        {"task": "door", "cv_primary_metric": metrics["macro_f1"]},
    )
    report = {
        "task": "door",
        "selected": {
            "model": "door_probability_ensemble",
            "members": [model.name for model in final.models],
            "macro_f1": metrics["macro_f1"],
            "fold_macro_f1": fold_scores,
        },
        "metrics": metrics,
        "class_counts": frame["target"].value_counts().to_dict(),
        "operation_counts": frame["operation"].value_counts().to_dict(),
    }
    _write_report(output_dir / "cv_report.json", report)
    return report


def train_acv(
    raw_root: Path, output_dir: Path, cache_dir: Path | None, seed: int = 42
) -> dict[str, Any]:
    cache = cache_dir / "acv_candidates.parquet" if cache_dir else None
    candidates = _cache_frame(cache, lambda: build_acv_candidates(raw_root))
    feature_columns = [
        column for column in candidates if column not in {"case_id", "car_id", "is_faulty"}
    ]
    cases = list(candidates["case_id"].unique())
    trials: list[dict[str, Any]] = []
    for model_name in ("logreg", "extra_trees", "random_forest"):
        for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            fold_rows = []
            for case_id in cases:
                train = candidates[candidates["case_id"] != case_id]
                validation = candidates[candidates["case_id"] == case_id]
                model = ACVRanker(model_name, supervised_weight=weight, seed=seed).fit(
                    train[feature_columns], train["is_faulty"]
                )
                scores = model.predict_proba(validation[feature_columns])[:, 1]
                ranking = rank_candidates(validation, scores)
                truth = str(validation.loc[validation["is_faulty"], "car_id"].iloc[0]).zfill(2)
                true_rank = ranking.index(truth) + 1
                fold_rows.append(
                    {
                        "case_id": case_id,
                        "truth": truth,
                        "true_rank": true_rank,
                        "rank_score": (len(ranking) - true_rank + 1) / len(ranking),
                        "ranking": ranking,
                    }
                )
            trials.append(
                {
                    "model": model_name,
                    "supervised_weight": weight,
                    "mean_rank_score": float(np.mean([row["rank_score"] for row in fold_rows])),
                    "folds": fold_rows,
                }
            )
    best = max(trials, key=lambda item: (item["mean_rank_score"], -item["supervised_weight"]))
    final = ACVRanker(str(best["model"]), float(best["supervised_weight"]), seed=seed).fit(
        candidates[feature_columns], candidates["is_faulty"]
    )
    save_classical_bundle(
        output_dir,
        final,
        {"selection": "leave_one_case_out_rank_decay"},
        {"feature_names": feature_columns},
        {"task": "acv", "cv_primary_metric": best["mean_rank_score"]},
    )
    report = {"task": "acv", "selected": best, "trials": trials, "training_cases": len(cases)}
    _write_report(output_dir / "cv_report.json", report)
    return report


def train_corrugation(
    raw_root: Path, output_dir: Path, cache_dir: Path | None, folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    cache = cache_dir / "corrugation_features.parquet" if cache_dir else None
    frame = _cache_frame(cache, lambda: build_file_features("corrugation", raw_root))
    feature_columns = [column for column in frame if column not in IDENTIFIER_COLUMNS]
    x, y = frame[feature_columns], frame["target"].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    trials = []
    specifications: list[tuple[str, str | dict[str, float] | None, dict[str, Any]]] = [
        ("logreg", "balanced", {"C": value}) for value in (0.1, 1.0, 10.0)
    ]
    specifications += [
        (forest, weight, {"n_estimators": 600, "min_samples_leaf": leaf, "max_features": maximum})
        for forest in ("extra_trees", "random_forest")
        for weight in ("balanced", "balanced_subsample")
        for leaf in (1, 2)
        for maximum in ("sqrt", 0.7)
    ]
    specifications += [
        (
            "hist_gradient_boosting",
            weight,
            {
                "learning_rate": learning_rate,
                "max_leaf_nodes": leaves,
                "min_samples_leaf": minimum_leaf,
                "l2_regularization": regularization,
                "max_iter": 250,
            },
        )
        for weight in (None, "balanced")
        for learning_rate in (0.05, 0.1)
        for leaves in (7, 15)
        for minimum_leaf in (10, 20)
        for regularization in (0.0, 1.0)
    ]
    for model_name, class_weight, estimator_params in specifications:
        oof = np.empty(len(y), dtype=object)
        fold_scores = []
        for train_index, validation_index in splitter.split(x, y):
            model = ClassicalClassifier(
                model_name,
                seed=seed,
                class_weight=class_weight,
                estimator_params=estimator_params,
            ).fit(x.iloc[train_index], y[train_index])
            prediction = model.predict(x.iloc[validation_index])
            oof[validation_index] = prediction
            fold_scores.append(classification_metrics(y[validation_index], prediction)["macro_f1"])
        report = classification_metrics(y, oof)
        trials.append(
            {
                "model": model_name,
                "class_weight": class_weight,
                "estimator_params": estimator_params,
                "macro_f1": report["macro_f1"],
                "balanced_accuracy": report["balanced_accuracy"],
                "fold_macro_f1": fold_scores,
                "per_class": report["per_class"],
                "confusion_matrix": report["confusion_matrix"],
            }
        )
    best = max(trials, key=lambda item: item["macro_f1"])
    final = ClassicalClassifier(
        str(best["model"]),
        seed=seed,
        class_weight=best["class_weight"],
        estimator_params=best["estimator_params"],
    ).fit(x, y)
    save_classical_bundle(
        output_dir,
        final,
        {"selection": "stratified_cv_macro_f1", "folds": folds},
        {"feature_names": feature_columns},
        {"task": "corrugation", "cv_primary_metric": best["macro_f1"]},
    )
    report = {
        "task": "corrugation",
        "selected": best,
        "trials": trials,
        "class_counts": frame["target"].value_counts().to_dict(),
    }
    _write_report(output_dir / "cv_report.json", report)
    return report


def _mape_scale(truth: np.ndarray, prediction: np.ndarray) -> float:
    prediction = np.maximum(np.asarray(prediction, dtype=float), 1e-12)
    truth = np.asarray(truth, dtype=float)
    ratios = truth / prediction
    weights = prediction / np.maximum(truth, 1e-12)
    order = np.argsort(ratios)
    cumulative = np.cumsum(weights[order])
    return float(
        ratios[order][min(int(np.searchsorted(cumulative, cumulative[-1] / 2)), len(ratios) - 1)]
    )


def train_shm(
    raw_root: Path, output_dir: Path, cache_dir: Path | None, folds: int = 5, seed: int = 42
) -> dict[str, Any]:
    cache = cache_dir / "shm_features.parquet" if cache_dir else None
    frame = _cache_frame(cache, lambda: build_file_features("shm", raw_root))
    feature_columns = [column for column in frame if column not in IDENTIFIER_COLUMNS]
    x, y = frame[feature_columns], frame["target"].astype(float).to_numpy()
    bins = pd.qcut(pd.Series(y).rank(method="first"), q=min(folds, 5), labels=False).to_numpy()
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    trials = []
    for model_name in ("ridge", "extra_trees", "random_forest", "hist_gradient_boosting"):
        for transform in ("log", "log1p", "raw"):
            oof = np.zeros(len(y), dtype=float)
            fold_scores = []
            for train_index, validation_index in splitter.split(x, bins):
                model = ClassicalRegressor(model_name, seed=seed, target_transform=transform).fit(
                    x.iloc[train_index], y[train_index]
                )
                prediction = model.predict(x.iloc[validation_index])
                oof[validation_index] = prediction
                fold_scores.append(regression_metrics(y[validation_index], prediction)["shm_score"])
            scale = _mape_scale(y, oof)
            calibrated = oof * scale
            metrics = regression_metrics(y, calibrated)
            trials.append(
                {
                    "model": model_name,
                    "target_transform": transform,
                    "prediction_scale": scale,
                    "shm_score": metrics["shm_score"],
                    "mape": metrics["mape"],
                    "mae": metrics["mae"],
                    "spearman": metrics["spearman"],
                    "uncalibrated_shm_score": regression_metrics(y, oof)["shm_score"],
                    "fold_uncalibrated_score": fold_scores,
                }
            )
    best = max(trials, key=lambda item: (item["shm_score"], item["spearman"]))
    final = ClassicalRegressor(
        str(best["model"]), seed=seed, target_transform=str(best["target_transform"])
    ).fit(x, y)
    final.prediction_scale_ = float(best["prediction_scale"])
    save_classical_bundle(
        output_dir,
        final,
        {"selection": "stratified_cv_mape", "folds": folds},
        {"feature_names": feature_columns},
        {"task": "shm", "cv_primary_metric": best["shm_score"]},
    )
    baseline = np.repeat(np.median(y), len(y))
    report = {
        "task": "shm",
        "selected": best,
        "trials": trials,
        "target_summary": pd.Series(y).describe().to_dict(),
        "median_baseline": regression_metrics(y, baseline),
    }
    _write_report(output_dir / "cv_report.json", report)
    return report
