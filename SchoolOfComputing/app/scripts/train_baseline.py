"""Explicit baseline training entry point; never runs on import."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from railguard.config import load_config
from railguard.data.adapters.acv import ACVAdapter
from railguard.data.adapters.corrugation import CorrugationAdapter
from railguard.data.adapters.door import DoorAdapter
from railguard.data.adapters.shm import SHMAdapter
from railguard.data.synthetic import synthetic_features
from railguard.evaluation import classification_metrics, regression_metrics
from railguard.features.acv import acv_candidate_features
from railguard.features.corrugation import corrugation_features
from railguard.features.door import door_feature_table
from railguard.features.shm import shm_features
from railguard.models import create_model
from railguard.models.serialization import save_classical_bundle


def real_features(task: str, raw_root: Path, max_samples: int | None) -> tuple[pd.DataFrame, np.ndarray]:
    if task == "door":
        samples = list(DoorAdapter(raw_root / "Door").samples("train"))
        if max_samples: samples = samples[:max_samples]
        return door_feature_table(samples), np.asarray([sample.target for sample in samples])
    if task == "acv":
        adapter = ACVAdapter(raw_root / "ACV")
        labels = pd.read_csv(adapter.root / "Train_Labels.csv", dtype={"faulty_car": str}).set_index("filename")["faulty_car"].str.zfill(2)
        files = adapter.files("train")[:max_samples] if max_samples else adapter.files("train")
        candidates = pd.concat([acv_candidate_features(pd.read_excel(path), path.name, labels[path.name]) for path in files], ignore_index=True)
        target = candidates.pop("is_faulty").to_numpy()
        return candidates.drop(columns=["case_id", "car_id"]), target
    adapter = CorrugationAdapter(raw_root / "Rail_Corrugation") if task == "corrugation" else SHMAdapter(raw_root / "SHM")
    samples = []
    for sample in adapter.samples("train"):
        samples.append(sample)
        if max_samples and len(samples) >= max_samples: break
    frame = pd.DataFrame([corrugation_features(sample) if task == "corrugation" else shm_features(sample) for sample in samples], index=[sample.sample_id for sample in samples])
    return frame, np.asarray([sample.target for sample in samples])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "acv", "corrugation", "shm"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Use deterministic smoke data instead of official inputs")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints/baseline_smoke"))
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    effective_max = args.max_samples or (8 if args.dry_run else None)
    x, y = synthetic_features(args.task, effective_max or 48, args.seed) if args.synthetic else real_features(args.task, args.raw_root, effective_max)
    name = config.model if config.model in {"logreg", "ridge", "elastic_net", "random_forest", "extra_trees", "hist_gradient_boosting"} else ("extra_trees")
    kwargs = {"seed": args.seed}
    if args.task == "shm":
        kwargs["target_transform"] = "log1p"
    model = create_model(args.task, name, **kwargs).fit(x, y)
    prediction = model.predict(x)
    metrics = regression_metrics(y, prediction) if args.task == "shm" else classification_metrics(y, prediction, model.predict_proba(x))
    print(json.dumps({"fit_smoke_metrics": metrics, "warning": "In-sample smoke diagnostics; use leakage-safe evaluation for reported performance."}, indent=2))
    if not args.dry_run:
        save_classical_bundle(args.output_dir, model, config.model_dump(mode="json"), {"feature_names": list(x.columns)}, {"task": args.task, "synthetic": args.synthetic})


if __name__ == "__main__":
    main()
