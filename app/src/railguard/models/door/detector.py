"""Door baseline cross-validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = object

from railguard.data.splits import classification_splits
from railguard.evaluation.classification import classification_metrics
from railguard.models.classical import ClassicalClassifier


if torch is not None:
    from railguard.models.backbones.channel_attention import ChannelAttention
    from railguard.models.backbones.cnn1d import MultiScaleCNN1D
    from railguard.models.backbones.fusion import GatedFusion
    from railguard.models.backbones.pooling import AttentionPool
    from railguard.models.backbones.spectral_encoder import SpectralEncoder

    class DoorNet(nn.Module):
        """Compact temporal/spectral Door classifier."""

        def __init__(self, input_channels: int, hidden_channels: int = 32, operations: int = 2, classes: int = 2) -> None:
            super().__init__()
            self.channel_attention = ChannelAttention(input_channels)
            self.temporal = MultiScaleCNN1D(input_channels, hidden_channels)
            self.pool = AttentionPool(hidden_channels)
            self.spectral = SpectralEncoder(hidden_channels)
            self.operation = nn.Embedding(operations, hidden_channels)
            self.fusion = GatedFusion(hidden_channels)
            self.head = nn.Sequential(nn.LayerNorm(hidden_channels), nn.Dropout(0.2), nn.Linear(hidden_channels, classes))

        def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> torch.Tensor:
            attended, _ = self.channel_attention(x)
            temporal = self.pool(self.temporal(attended, mask), mask)
            spectrum = torch.log1p(torch.fft.rfft(attended, dim=1).abs().mean(dim=2))
            fused = self.fusion(temporal, self.spectral(spectrum))
            return fused + self.operation(metadata.long()) if metadata is not None else fused

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> torch.Tensor:
            return self.head(self.encode(x, mask, metadata))


def cross_validate_door(features: pd.DataFrame, labels: np.ndarray, operations: np.ndarray, folds: int = 5, seed: int = 42) -> list[dict[str, Any]]:
    strata = np.char.add(np.asarray(labels).astype(str), np.char.add("__", np.asarray(operations).astype(str)))
    reports = []
    for fold, (train, validation) in enumerate(classification_splits(strata, folds=folds, seed=seed)):
        model = ClassicalClassifier("extra_trees", seed=seed).fit(features.iloc[train], labels[train])
        prediction = model.predict(features.iloc[validation])
        probability = model.predict_proba(features.iloc[validation])
        report = classification_metrics(labels[validation], prediction, probability, list(model.pipeline.named_steps["model"].classes_))
        report["fold"] = fold
        report["per_operation"] = {
            operation: classification_metrics(labels[validation][operations[validation] == operation], prediction[operations[validation] == operation])
            for operation in sorted(set(operations[validation]))
        }
        reports.append(report)
    return reports
