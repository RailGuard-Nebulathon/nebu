from pathlib import Path

import pandas as pd
import pytest

from railguard.submission.validator import validate_prediction_file, validate_prediction_frame


EXAMPLES = Path("NebulaX-Hackathon-ProblemStatement/PS3/04_Example_Submission")


@pytest.mark.parametrize("name", ["door_predictions.csv", "acv_predictions.csv", "rail_predictions.csv", "shm_predictions.csv"])
def test_official_examples_validate(name: str) -> None:
    assert not validate_prediction_file(EXAMPLES / name).empty


def test_missing_duplicate_and_label_checks() -> None:
    frame = pd.DataFrame({"file_id": ["a.csv", "a.csv"], "prediction": ["Normal", "Bad"]})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_prediction_frame("corrugation", frame)
    frame = pd.DataFrame({"file_id": ["a.csv"], "prediction": ["Normal"]})
    with pytest.raises(ValueError, match="Missing IDs"):
        validate_prediction_frame("corrugation", frame, expected_ids=["a.csv", "b.csv"])

