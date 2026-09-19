"""Reproducibility metadata capture."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def environment_metadata(packages: tuple[str, ...] = ("numpy", "pandas", "scikit-learn", "scipy", "torch")) -> dict[str, object]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {"timestamp": datetime.now(UTC).isoformat(), "python": platform.python_version(), "platform": platform.platform(), "packages": versions, "git_commit": git_commit}

