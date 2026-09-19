"""Learned one-dimensional spectral branch."""

import torch
from torch import nn


class SpectralEncoder(nn.Module):
    def __init__(self, output_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.GELU(), nn.Conv1d(16, output_dim, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim == 2:
            spectrum = spectrum.unsqueeze(1)
        return self.network(spectrum).squeeze(-1)

