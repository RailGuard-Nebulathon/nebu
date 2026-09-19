"""Dependency-free message passing for optional car graphs."""

import torch
from torch import nn


class CarGraphLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.update = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.GELU())

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        weights = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1)
        messages = torch.matmul(weights, nodes)
        return nodes + self.update(torch.cat([nodes, messages], dim=-1))

