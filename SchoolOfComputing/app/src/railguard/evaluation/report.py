"""JSON report writer."""

import json
from pathlib import Path


def write_report(report: dict[str, object], path: str | Path) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8"); return target

