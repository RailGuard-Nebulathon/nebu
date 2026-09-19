from pathlib import Path

import pandas as pd

from railguard.data.adapters.door import DoorAdapter, infer_cycle_boundaries


def test_fallback_segmentation_boundaries() -> None:
    rows = 50
    frame = pd.DataFrame({column: [0] * rows for column in DoorAdapter.__module__ and [
        "Motor current(mA)", "Motor Voltage(10mV)", "Motor electrodynamic force",
        "Door opening time(.1s)", "Door closing time(.1s)", "Door leaf position",
        "Close command", "Open command", "DCSR", "DCSL", "DLSR", "DLSL",
        "Door Opened", "Door Locked", "Door is opening", "Door is closing",
    ]})
    frame.insert(0, "Datetime", [f"2023-7-5-0-0-0-{i}" for i in range(rows)])
    frame.loc[10:39, "Door is opening"] = 1
    assert infer_cycle_boundaries(frame, min_rows=20) == [(10, 39, "Open")]


def test_official_train_segments_match_exactly() -> None:
    root = Path("NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Door")
    samples = list(DoorAdapter(root).samples("train"))
    assert len(samples) == 110
    assert all(sample.metadata["row_count_diagnostic"]["exact"] for sample in samples)
    assert {sample.target for sample in samples} == {"Normal", "Abnormal resistance"}


def test_official_test_gap_segmentation() -> None:
    root = Path("NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Door")
    samples = list(DoorAdapter(root).samples("test"))
    assert len(samples) == 38
    assert all(130 <= len(sample.values) <= 190 for sample in samples)
