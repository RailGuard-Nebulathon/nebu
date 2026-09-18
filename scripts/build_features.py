"""Explicit per-task feature generation."""

import argparse
from pathlib import Path
import pandas as pd

from railguard.data.registry import get_adapter
from railguard.features.corrugation import corrugation_features
from railguard.features.door import door_feature_table
from railguard.features.shm import shm_features


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--task", required=True, choices=("door", "corrugation", "shm")); parser.add_argument("--config", type=Path); parser.add_argument("--raw-root", type=Path, default=Path("data/raw")); parser.add_argument("--max-samples", type=int); parser.add_argument("--output", type=Path, default=Path("data/processed/features.csv")); args = parser.parse_args()
    folder = {"door": "Door", "corrugation": "Rail_Corrugation", "shm": "SHM"}[args.task]
    samples = list(get_adapter(args.task, args.raw_root / folder).samples("train"))
    if args.max_samples: samples = samples[: args.max_samples]
    if args.task == "door": frame = door_feature_table(samples)
    else: frame = pd.DataFrame([corrugation_features(s) if args.task == "corrugation" else shm_features(s) for s in samples], index=[s.sample_id for s in samples])
    args.output.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(args.output); print(f"{frame.shape} -> {args.output}")


if __name__ == "__main__": main()

