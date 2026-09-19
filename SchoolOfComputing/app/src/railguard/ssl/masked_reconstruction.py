"""Masked continuous-signal reconstruction."""

import torch
from torch import nn

from railguard.ssl.augmentations import span_mask


class MaskedReconstruction(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int, channels: int) -> None:
        super().__init__()
        self.encoder, self.decoder = encoder, nn.Linear(hidden_dim, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        corrupted, reconstruction_mask = span_mask(x)
        encoded = self.encoder(corrupted, mask)
        return self.decoder(encoded), x, reconstruction_mask


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, reconstruction_mask: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction[reconstruction_mask] - target[reconstruction_mask]) ** 2)

