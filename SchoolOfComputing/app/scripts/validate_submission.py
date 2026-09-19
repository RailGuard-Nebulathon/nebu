"""Validate official CSV or predictions.zip structure."""

import argparse
from pathlib import Path

from railguard.submission.validator import validate_prediction_file, validate_zip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    result = validate_zip(args.path) if args.path.suffix.lower() == ".zip" else validate_prediction_file(args.path)
    print(f"Valid: {result}")


if __name__ == "__main__":
    main()

