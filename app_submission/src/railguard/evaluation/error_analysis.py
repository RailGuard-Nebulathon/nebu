"""High-confidence classification error extraction."""

import numpy as np
import pandas as pd


def classification_errors(ids, truth, prediction, probabilities) -> pd.DataFrame:
    confidence = np.max(probabilities, axis=1)
    frame = pd.DataFrame({"sample_id": ids, "truth": truth, "prediction": prediction, "confidence": confidence})
    return frame[frame["truth"] != frame["prediction"]].sort_values("confidence", ascending=False)

