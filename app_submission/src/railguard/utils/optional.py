"""Optional dependency checks with actionable errors."""

import importlib
from types import ModuleType


def optional_import(name: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(f"Optional dependency '{name}' is required; install railguard-ai[{extra}]") from exc

