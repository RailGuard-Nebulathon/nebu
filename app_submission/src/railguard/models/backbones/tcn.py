"""Residual dilated temporal convolution network."""

import torch
from torch import nn


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalConvNet(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 32, dilations: tuple[int, ...] = (1, 2, 4, 8), dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Conv1d(input_channels, hidden_channels, 1)
        self.blocks = nn.Sequential(*[_TCNBlock(hidden_channels, dilation, dropout) for dilation in dilations])
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        output = self.norm(self.blocks(self.stem(x.transpose(1, 2))).transpose(1, 2))
        return output * mask.unsqueeze(-1).to(output.dtype) if mask is not None else output

