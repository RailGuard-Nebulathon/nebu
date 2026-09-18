"""Explicit metric-aligned PS3 model selection and bundle training CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from railguard.training.competition import train_acv, train_corrugation, train_shm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("acv", "corrugation", "shm", "all"))
    parser.add_argument("--raw-root", type=Path, required=True, help="Path to PS3/02_Datasets")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints/competition"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/competition"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    selected = ("acv", "corrugation", "shm") if args.task == "all" else (args.task,)
    reports = {}
    for task in selected:
        print(f"Selecting {task} model with official metric...", flush=True)
        if task == "acv":
            report = train_acv(args.raw_root, args.output_root / task, args.cache_dir, args.seed)
        elif task == "corrugation":
            report = train_corrugation(
                args.raw_root, args.output_root / task, args.cache_dir, args.folds, args.seed
            )
        else:
            report = train_shm(
                args.raw_root, args.output_root / task, args.cache_dir, args.folds, args.seed
            )
        reports[task] = report["selected"]
        print(json.dumps({task: report["selected"]}, indent=2), flush=True)
    print(json.dumps({"selected_models": reports}, indent=2), flush=True)


if __name__ == "__main__":
    main()
