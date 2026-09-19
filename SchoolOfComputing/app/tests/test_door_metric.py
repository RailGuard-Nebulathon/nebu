import pandas as pd

from railguard.evaluation.door import door_iou_weighted_f1


def test_perfect_door_iou_score() -> None:
    frame = pd.DataFrame({"start_time": pd.to_datetime(["2024-01-01"]), "end_time": pd.to_datetime(["2024-01-01 00:00:01"]), "prediction": ["Normal"]})
    assert door_iou_weighted_f1(frame, frame)["score"] == 1.0

