"""Task losses."""

import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma, self.weight = gamma, weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        probability = torch.exp(-cross_entropy)
        return ((1 - probability) ** self.gamma * cross_entropy).mean()


def gaussian_nll(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mean, log_variance = output[:, 0], output[:, 1]
    return 0.5 * (log_variance + (target - mean) ** 2 * torch.exp(-log_variance)).mean()

