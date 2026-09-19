"""Fast deterministic synthetic tabular tasks for smoke tests."""

import numpy as np
import pandas as pd


def synthetic_features(task: str, n: int = 48, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    if task == "door":
        y = np.where(x[:, 0] + x[:, 1] > 0, "abnormal_resistance", "normal")
    elif task == "acv":
        y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    elif task == "corrugation":
        score = x[:, 0] - x[:, 1]
        y = np.where(score < -0.5, "side_i", np.where(score > 0.5, "side_ii", "normal"))
    elif task == "shm":
        y = np.maximum(0, 0.1 + np.abs(x[:, 0]) * 0.2 + np.square(x[:, 1]) * 0.05)
    else:
        raise ValueError(task)
    return pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])]), y

