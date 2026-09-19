"""Mask-aware Transformer encoder option."""

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.project = nn.Linear(input_channels, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 4, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(self.project(x), src_key_padding_mask=(~mask.bool()) if mask is not None else None)

