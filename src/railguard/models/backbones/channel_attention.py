"""Channel masking and squeeze-excitation attribution."""

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.network = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        summary = x.abs().mean(dim=1)
        weights = self.network(summary)
        if channel_mask is not None:
            weights = weights * channel_mask.to(weights.dtype)
        return x * weights.unsqueeze(1), weights

