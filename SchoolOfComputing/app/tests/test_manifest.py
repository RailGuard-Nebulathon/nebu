from pathlib import Path

from railguard.data.manifest import build_manifest, manifest_hash
from railguard.data.schema_inspector import inspect_file


def test_csv_inspection_and_manifest() -> None:
    source = Path("tests/fixtures/sample.csv")
    report = inspect_file(source)
    assert report["n_rows"] == 2
    manifest = build_manifest([source], "shm", "train", {source.name: 0.2})
    assert len(manifest.loc[0, "file_hash"]) == 64
    assert manifest_hash(manifest) == manifest_hash(manifest.copy())


def test_xlsx_inspection() -> None:
    source = Path("NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/ACV/Train/acv_case_01.xlsx")
    report = inspect_file(source)
    assert report["sheet_names"]
    assert report["sheets"][0]["n_columns"] > 3
    assert any(column.startswith("Car 01 -") for column in report["sheets"][0]["columns"])
