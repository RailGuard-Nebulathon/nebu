"""Compact multi-scale 1D convolution encoder."""

import torch
from torch import nn


class MultiScaleCNN1D(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 32, kernels: tuple[int, ...] = (3, 7, 15), dropout: float = 0.1) -> None:
        super().__init__()
        branch = max(hidden_channels // len(kernels), 1)
        self.branches = nn.ModuleList([nn.Conv1d(input_channels, branch, kernel, padding=kernel // 2) for kernel in kernels])
        merged = branch * len(kernels)
        self.project = nn.Conv1d(merged, hidden_channels, 1)
        self.block = nn.Sequential(nn.BatchNorm1d(hidden_channels), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1))
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        transposed = x.transpose(1, 2)
        hidden = self.project(torch.cat([branch(transposed) for branch in self.branches], dim=1))
        hidden = hidden + self.block(hidden)
        output = self.norm(hidden.transpose(1, 2))
        return output * mask.unsqueeze(-1).to(output.dtype) if mask is not None else output

