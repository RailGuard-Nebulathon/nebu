"""Robustness report CLI."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "acv", "corrugation", "shm"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/robustness.json"))
    parser.parse_args()
    print("Load an inspected evaluation split to generate a corruption report; labels are never corrupted.")


if __name__ == "__main__":
    main()

