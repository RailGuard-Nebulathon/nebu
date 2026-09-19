from pathlib import Path

import numpy as np
import pytest

from railguard.data.adapters.shm import SHMAdapter
from railguard.features.shm import miner_damage_proxy, rainflow_ranges, shm_features


def test_real_shm_file_and_fatigue_proxies() -> None:
    root = Path("NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/SHM")
    sample = next(iter(SHMAdapter(root).samples("train")))
    assert sample.values.shape == (581120, 1)
    features = shm_features(sample)
    assert features["stress_0__rainflow_cycle_count"] > 0
    assert np.isfinite(list(features.values())).all()


def test_miner_requires_material_constants() -> None:
    ranges, counts = rainflow_ranges(np.array([0, 1, 0, -1, 0]))
    with pytest.raises(ValueError, match="S-N"):
        miner_damage_proxy(ranges, counts)

