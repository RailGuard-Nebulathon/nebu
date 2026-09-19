from __future__ import annotations

import io

import pandas as pd

from railguard_api.metadata import extract_metadata


def test_door_uses_embedded_recording_time_without_inventing_train_id() -> None:
    payload = b"Datetime,Motor Current\n2023-7-5-0-0-0-0,1.2\n"

    result = extract_metadata("door", payload, "Test (1).csv", 1_700_000_000_000)

    assert result.asset_id == "Door dataset Test (1)"
    assert result.asset_source == "filename"
    assert result.measurement_time.isoformat() == "2023-07-05T00:00:00"
    assert result.measurement_time_source == "embedded"


def test_rail_derives_sensor_coverage_and_labels_fallbacks() -> None:
    payload = (
        b"Rotating speed,Vibration/Shock of bearing in position 1 of car 1,"
        b"Vibration/Shock of bearing in position 8 of car 8\n1,2,3\n"
    )

    result = extract_metadata("corrugation", payload, "Test1.csv", 1_700_000_000_000)

    assert result.asset_id == "Rail sample Test1"
    assert result.component_info == "Bearing sensors · Cars 1–8 · Positions 1–8"
    assert result.measurement_time_source == "file_modified"
    assert len(result.warnings) == 2


def test_shm_recognises_headerless_stress_channels() -> None:
    result = extract_metadata("shm", b"12.5\n13.1\n", "test14.csv", 1_700_000_000_000)

    assert result.asset_id == "SHM sample test14"
    assert result.component_info == "Structural stress · 1 channel"


def test_acv_uses_embedded_train_model_and_time() -> None:
    workbook = io.BytesIO()
    pd.DataFrame(
        [{
            "Car model": "A",
            "Train number": 620,
            "Time": pd.Timestamp("2023-05-18 00:00:00"),
            "Car 01_Temperature": 22.0,
            "Car 08_Temperature": 23.0,
        }]
    ).to_excel(workbook, index=False)

    result = extract_metadata("acv", workbook.getvalue(), "acv_case_01.xlsx", None)

    assert result.asset_id == "Train 620"
    assert result.component_info == "ACV · Cars 01–08 · Model A"
    assert result.measurement_time.isoformat() == "2023-05-18T00:00:00"
    assert result.warnings == []
