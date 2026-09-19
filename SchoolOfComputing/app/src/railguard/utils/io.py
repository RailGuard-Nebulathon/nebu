"""Safe local serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    return target


def write_yaml(path: str | Path, value: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return target

