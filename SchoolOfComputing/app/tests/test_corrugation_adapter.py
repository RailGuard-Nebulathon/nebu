from pathlib import Path

import numpy as np

from railguard.data.adapters.corrugation import CorrugationAdapter
from railguard.features.corrugation import corrugation_features


def test_real_corrugation_file_and_features() -> None:
    root = Path("NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Rail_Corrugation")
    sample = next(iter(CorrugationAdapter(root).samples("train")))
    assert sample.values.shape == (10000, 128)
    features = corrugation_features(sample)
    assert "side_energy_ratio__vibration" in features
    assert np.isfinite(list(features.values())).all()

