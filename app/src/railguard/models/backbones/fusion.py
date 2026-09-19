"""Gated branch fusion."""

import torch
from torch import nn


class GatedFusion(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.Sigmoid())

    def forward(self, temporal: torch.Tensor, spectral: torch.Tensor) -> torch.Tensor:
        gate = self.gate(torch.cat([temporal, spectral], dim=-1))
        return gate * temporal + (1 - gate) * spectral

