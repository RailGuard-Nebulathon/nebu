"""Opt-in SSL entry point. No pretraining occurs without explicit invocation."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("door", "acv", "corrugation", "shm"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.parse_args()
    print("SSL pipeline configured. Connect inspected task windows before explicit pretraining.")


if __name__ == "__main__":
    main()

