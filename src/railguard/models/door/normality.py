"""Normal-cycle reconstruction and isolation evidence."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DoorNormalityModel:
    def __init__(self, components: int = 8, seed: int = 42) -> None:
        self.components = components
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        self.isolation = IsolationForest(random_state=seed, contamination="auto", n_jobs=1)

    def fit(self, normal_features: np.ndarray) -> "DoorNormalityModel":
        scaled = self.scaler.fit_transform(normal_features)
        self.pca = PCA(n_components=min(self.components, scaled.shape[0], scaled.shape[1])).fit(scaled)
        self.isolation.fit(scaled)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Normality model is not fitted")
        scaled = self.scaler.transform(features)
        reconstructed = self.pca.inverse_transform(self.pca.transform(scaled))
        reconstruction = np.mean(np.square(scaled - reconstructed), axis=1)
        isolation = -self.isolation.score_samples(scaled)
        return reconstruction + isolation - isolation.min()


def combine_scores(classifier_probability: np.ndarray, anomaly_score: np.ndarray, supervised_weight: float = 0.7) -> np.ndarray:
    normalized = (anomaly_score - anomaly_score.min()) / max(float(np.ptp(anomaly_score)), 1e-9)
    return supervised_weight * classifier_probability + (1 - supervised_weight) * normalized

