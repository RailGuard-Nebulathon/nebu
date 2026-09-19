"""Normal-only anomaly models."""

from sklearn.ensemble import IsolationForest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def isolation_forest(seed: int = 42):
    return make_pipeline(StandardScaler(), IsolationForest(contamination="auto", random_state=seed))

