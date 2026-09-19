from pathlib import Path

import pandas as pd

from railguard.data.adapters.acv import ACVAdapter, discover_acv_mapping
from railguard.features.acv import acv_candidate_features, robust_rule_score
from railguard.models.acv.localizer import rank_candidates


def test_real_acv_schema_and_candidates() -> None:
    root = Path("NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/ACV")
    path = root / "Train/acv_case_01.xlsx"
    mapping = discover_acv_mapping(path)
    cars = next(iter(mapping["sheets"].values()))["cars"]
    assert sorted(cars) == [f"{i:02d}" for i in range(1, 9)]
    frame = pd.read_excel(path, nrows=256)
    candidates = acv_candidate_features(frame, path.name, "01")
    assert len(candidates) == 8 and candidates["is_faulty"].sum() == 1
    ranking = rank_candidates(candidates, robust_rule_score(candidates))
    assert sorted(ranking) == sorted(candidates["car_id"])
