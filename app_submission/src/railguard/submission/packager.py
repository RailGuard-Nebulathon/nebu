"""Flat deterministic predictions.zip packager."""

from pathlib import Path
import zipfile

from railguard.submission.schemas import OFFICIAL_FILES
from railguard.submission.validator import validate_prediction_file, validate_zip


def package_predictions(files: list[str | Path], output: str | Path) -> Path:
    sources = [Path(path) for path in files]
    allowed = set(OFFICIAL_FILES.values())
    if len({path.name for path in sources}) != len(sources):
        raise ValueError("Duplicate prediction filenames")
    for source in sources:
        if source.name not in allowed:
            raise ValueError(f"Unexpected prediction filename: {source.name}")
        validate_prediction_file(source)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(sources, key=lambda path: path.name):
            archive.write(source, arcname=source.name)
    validate_zip(target)
    return target

