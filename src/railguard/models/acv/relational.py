"""Native-PyTorch variable-car relational attention."""

import torch
from torch import nn

from railguard.models.backbones.pooling import MaskedMeanPool
from railguard.models.backbones.tcn import TemporalConvNet


class ACVRelationalNet(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.encoder = TemporalConvNet(input_channels, hidden_dim, dilations=(1, 2))
        self.pool = MaskedMeanPool()
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> torch.Tensor:
        batch, cars, time, channels = x.shape
        temporal_mask = mask.reshape(batch * cars, time) if mask is not None else None
        encoded = self.encoder(x.reshape(batch * cars, time, channels), temporal_mask)
        return self.pool(encoded, temporal_mask).reshape(batch, cars, -1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None, car_mask: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.encode(x, mask, metadata)
        relational, _ = self.attention(embeddings, embeddings, embeddings, key_padding_mask=(~car_mask.bool()) if car_mask is not None else None)
        logits = self.head(relational).squeeze(-1)
        return logits.masked_fill(~car_mask.bool(), torch.finfo(logits.dtype).min) if car_mask is not None else logits

