"""Held-out neural-logit temperature scaling."""

import torch
from torch import nn


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.05, 20)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, calibration_logits: torch.Tensor, calibration_targets: torch.Tensor, steps: int = 50) -> "TemperatureScaler":
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=steps)

        def closure():
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(self(calibration_logits), calibration_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return self

