"""Inspect PS3 source schemas without reading entire large files into memory."""

from __future__ import annotations

import argparse
from pathlib import Path

from railguard.constants import TASKS
from railguard.data.schema_inspector import inspect_file, reports_to_markdown
from railguard.utils.io import write_json

OFFICIAL_DIR = {"door": "Door", "acv": "ACV", "corrugation": "Rail_Corrugation", "shm": "SHM"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/manifests"))
    parser.add_argument("--max-files", type=int, default=3, help="Representative files per split")
    args = parser.parse_args()
    tasks = TASKS if args.task == "all" else (args.task,)
    reports = []
    for task in tasks:
        root = args.raw_root / OFFICIAL_DIR[task]
        if not root.exists():
            print(f"SKIP {task}: {root} does not exist")
            continue
        files = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".xlsx", ".txt"})[: args.max_files]
        reports.extend(inspect_file(path) for path in files)
    write_json(args.output_root / "schema_report.json", reports)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "schema_report.md").write_text(reports_to_markdown(reports), encoding="utf-8")
    print(f"Inspected {len(reports)} files")


if __name__ == "__main__":
    main()

