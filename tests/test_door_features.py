from pathlib import Path

import numpy as np

from railguard.data.adapters.door import DoorAdapter
from railguard.features.door import door_feature_table
from railguard.models.door.normality import DoorNormalityModel


def test_door_features_and_normality() -> None:
    root = Path("NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Door")
    samples = list(DoorAdapter(root).samples("train"))[:8]
    features = door_feature_table(samples)
    assert np.isfinite(features.to_numpy()).all()
    model = DoorNormalityModel(components=2).fit(features.to_numpy()[:5])
    assert model.score(features.to_numpy()).shape == (8,)

