"""Embedding-space nearest evidence examples."""

import numpy as np
from sklearn.neighbors import NearestNeighbors


class NeighbourIndex:
    def __init__(self, neighbours: int = 5) -> None:
        self.neighbours = neighbours
        self.index: NearestNeighbors | None = None
        self.records: list[dict[str, object]] = []

    def fit(self, embeddings: np.ndarray, records: list[dict[str, object]]) -> "NeighbourIndex":
        if len(embeddings) != len(records):
            raise ValueError("Embedding and record counts differ")
        self.index = NearestNeighbors(n_neighbors=min(self.neighbours, len(records))).fit(embeddings)
        self.records = records
        return self

    def query(self, embeddings: np.ndarray) -> list[list[dict[str, object]]]:
        if self.index is None:
            raise RuntimeError("Neighbour index is not fitted")
        distances, indexes = self.index.kneighbors(embeddings)
        return [[self.records[int(index)] | {"distance": float(distance)} for distance, index in zip(row_distances, row_indexes, strict=True)] for row_distances, row_indexes in zip(distances, indexes, strict=True)]

