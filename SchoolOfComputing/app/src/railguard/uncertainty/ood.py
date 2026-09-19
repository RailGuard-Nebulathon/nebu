"""Training-only embedding references for Mahalanobis and kNN OOD."""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class EmbeddingOOD:
    def __init__(self, neighbours: int = 5, threshold_quantile: float = 0.99) -> None:
        self.neighbours, self.threshold_quantile = neighbours, threshold_quantile
        self.scaler, self.covariance = StandardScaler(), LedoitWolf()
        self.knn: NearestNeighbors | None = None
        self.thresholds: dict[str, float] = {}

    def fit(self, training_embeddings: np.ndarray) -> "EmbeddingOOD":
        scaled = self.scaler.fit_transform(training_embeddings)
        self.covariance.fit(scaled)
        self.knn = NearestNeighbors(n_neighbors=min(self.neighbours + 1, len(scaled))).fit(scaled)
        mahalanobis, knn = self.score(training_embeddings, training_reference=True)
        self.thresholds = {"mahalanobis": float(np.quantile(mahalanobis, self.threshold_quantile)), "knn": float(np.quantile(knn, self.threshold_quantile))}
        return self

    def score(self, embeddings: np.ndarray, training_reference: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if self.knn is None:
            raise RuntimeError("OOD reference is not fitted")
        scaled = self.scaler.transform(embeddings)
        mahalanobis = self.covariance.mahalanobis(scaled)
        distances, _ = self.knn.kneighbors(scaled)
        index = min(self.neighbours, distances.shape[1] - 1) if training_reference else min(self.neighbours - 1, distances.shape[1] - 1)
        return mahalanobis, distances[:, index]

    def is_ood(self, embeddings: np.ndarray) -> np.ndarray:
        mahalanobis, knn = self.score(embeddings)
        return (mahalanobis > self.thresholds["mahalanobis"]) | (knn > self.thresholds["knn"])

