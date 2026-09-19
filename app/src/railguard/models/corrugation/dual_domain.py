"""Temporal/spectral CorrugationNet."""

import torch
from torch import nn

from railguard.models.backbones.channel_attention import ChannelAttention
from railguard.models.backbones.fusion import GatedFusion
from railguard.models.backbones.pooling import AttentionPool
from railguard.models.backbones.spectral_encoder import SpectralEncoder
from railguard.models.backbones.tcn import TemporalConvNet


class CorrugationNet(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 32, classes: int = 3) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(input_channels)
        self.temporal = TemporalConvNet(input_channels, hidden_channels)
        self.pool = AttentionPool(hidden_channels)
        self.spectral = SpectralEncoder(hidden_channels)
        self.fusion = GatedFusion(hidden_channels)
        self.head = nn.Linear(hidden_channels, classes)

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> torch.Tensor:
        attended, _ = self.channel_attention(x)
        temporal = self.pool(self.temporal(attended, mask), mask)
        spectrum = torch.log1p(torch.fft.rfft(attended, dim=1).abs().mean(dim=2))
        return self.fusion(temporal, self.spectral(spectrum))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.encode(x, mask, metadata))

