from __future__ import annotations

from railguard_api.history import create_analysis, list_analyses, update_metadata


def _result() -> dict:
    return {
        "task": "door",
        "task_name": "Door diagnostics",
        "mode": "real",
        "source_file": "Test.csv",
        "input_sha256": "a" * 64,
        "model_version": "door:v1:test",
        "output_filename": "door_predictions.csv",
        "rows": [
            {
                "start_time": "2026-1-1-0-0-0-0",
                "end_time": "2026-1-1-0-0-3-0",
                "prediction": "Normal",
            }
        ],
        "summary": {"cycles": 1, "abnormal_cycles": 0},
        "visual": {"segments": []},
        "csv_text": "start_time,end_time,prediction\n",
        "notices": [],
    }


def test_history_round_trip_and_metadata_only_update(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RAILGUARD_HISTORY_DB", str(tmp_path / "history.db"))
    created = create_analysis(
        asset_id="Train-04",
        component_info="Door 2L",
        measurement_time="2026-09-19T01:00:00+00:00",
        result=_result(),
    )

    assert list_analyses(asset="train-04", task="door") == [created]
    updated = update_metadata(
        created["id"],
        asset_id="Train-05",
        component_info="Door 3R",
        measurement_time="2026-09-19T02:00:00+00:00",
    )

    assert updated["asset_id"] == "Train-05"
    assert updated["result"] == created["result"]
    assert updated["input_sha256"] == created["input_sha256"]
