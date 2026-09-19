"""Build per-task manifests from official folder layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from railguard.constants import TASKS
from railguard.data.manifest import build_manifest

OFFICIAL_DIR = {"door": "Door", "acv": "ACV", "corrugation": "Rail_Corrugation", "shm": "SHM"}


def _labels(root: Path) -> dict[str, object]:
    candidates = [root / "Train_Labels.csv"]
    for path in candidates:
        if path.exists():
            frame = pd.read_csv(path, dtype={"faulty_car": str})
            key, value = frame.columns[:2]
            return dict(zip(frame[key].astype(str), frame[value], strict=True))
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--max-files", type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        root = args.raw_root / OFFICIAL_DIR[task]
        if not root.exists():
            print(f"SKIP {task}: {root} does not exist")
            continue
        parts = []
        for split_name, folder_name in (("train", "Train"), ("test", "Test")):
            folder = root / folder_name
            if task == "door":
                paths = [root / f"{folder_name}.csv"]
            else:
                paths = sorted(p for p in folder.glob("*") if p.suffix.lower() in {".csv", ".xlsx", ".txt"})
            paths = [p for p in paths if p.exists()]
            if args.max_files:
                paths = paths[: args.max_files]
            parts.append(build_manifest(paths, task, split_name, _labels(root) if split_name == "train" else None))
        frame = pd.concat(parts, ignore_index=True)
        frame.to_csv(args.output_root / f"{task}_manifest.csv", index=False)
        print(f"{task}: {len(frame)} records")


if __name__ == "__main__":
    main()

