"""Validate existing task prediction CSVs and build predictions.zip."""

import argparse
from pathlib import Path

from railguard.submission.packager import package_predictions
from railguard.submission.schemas import OFFICIAL_FILES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("all", "door", "acv", "corrugation", "shm"), default="all")
    parser.add_argument("--predictions-root", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submissions/predictions.zip"))
    args = parser.parse_args()
    tasks = OFFICIAL_FILES if args.task == "all" else {args.task: OFFICIAL_FILES[args.task]}
    files = [args.predictions_root / filename for filename in tasks.values() if (args.predictions_root / filename).exists()]
    if not files:
        raise SystemExit("No official prediction CSVs found; inference must run first")
    print(package_predictions(files, args.output))


if __name__ == "__main__":
    main()

