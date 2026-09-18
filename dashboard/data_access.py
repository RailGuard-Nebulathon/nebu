"""Dashboard data sources with checkpoint-free deterministic demo data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def demo_data(seed: int = 42) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 4, 200)
    door = pd.DataFrame({"time": time, "motor_current": 120 + 25 * np.sin(time * 4) + rng.normal(0, 3, len(time)), "voltage": 400 + 40 * np.cos(time * 3), "position": np.linspace(700, 0, len(time))})
    acv_time = np.arange(120) * 30 / 60
    acv = pd.DataFrame({"minutes": acv_time, **{f"Car {i:02d}": 24 - 3 * (1 - np.exp(-acv_time / (10 if i == 3 else 4))) + rng.normal(0, 0.1, len(acv_time)) for i in range(1, 9)}})
    vibration_time = np.linspace(0, 1, 1000, endpoint=False)
    vibration = np.sin(2 * np.pi * 120 * vibration_time) + 0.35 * np.sin(2 * np.pi * 310 * vibration_time) + rng.normal(0, 0.15, 1000)
    stress_time = np.linspace(0, 20, 1000)
    stress = (1 + 0.3 * np.sin(stress_time / 3)) * np.sin(stress_time * 4) + rng.normal(0, 0.08, 1000)
    predictions = pd.DataFrame([
        {"component": "Door 01", "task": "door", "prediction": "Normal", "confidence": 0.91, "ood": False, "health": 89},
        {"component": "ACV Car 03", "task": "acv", "prediction": "Inspect", "confidence": 0.78, "ood": False, "health": 52},
        {"component": "Rail Side II", "task": "corrugation", "prediction": "Side II", "confidence": 0.83, "ood": False, "health": 43},
        {"component": "Bogie stress", "task": "shm", "prediction": "0.23", "confidence": 0.74, "ood": True, "health": 67},
    ])
    return {"door": door, "acv": acv, "vibration_time": vibration_time, "vibration": vibration, "stress_time": stress_time, "stress": stress, "predictions": predictions}


def real_predictions(root: str | Path = "outputs/predictions") -> pd.DataFrame:
    root = Path(root)
    frames = []
    for path in root.glob("*_predictions.csv"):
        frame = pd.read_csv(path)
        frame.insert(0, "source", path.name)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

