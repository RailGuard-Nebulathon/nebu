"""Conservative time-series SSL augmentations."""

import torch
from torch.nn import functional as functional


def jitter(x: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma * x.std(dim=1, keepdim=True).clamp_min(1e-6)


def scaling(x: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    scale = 1 + torch.randn(x.shape[0], 1, x.shape[2], device=x.device) * sigma
    return x * scale


def span_mask(x: torch.Tensor, fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    output = x.clone()
    mask = torch.zeros_like(x, dtype=torch.bool)
    width = max(1, int(x.shape[1] * fraction))
    for batch in range(x.shape[0]):
        start = int(torch.randint(0, max(x.shape[1] - width + 1, 1), (1,), device=x.device))
        mask[batch, start : start + width] = True
    output[mask] = 0
    return output, mask


def channel_dropout(x: torch.Tensor, probability: float = 0.1) -> torch.Tensor:
    keep = torch.rand(x.shape[0], 1, x.shape[2], device=x.device) > probability
    return x * keep


def crop_resize(x: torch.Tensor, fraction: float = 0.9) -> torch.Tensor:
    length = max(2, int(x.shape[1] * fraction))
    start = int(torch.randint(0, max(x.shape[1] - length + 1, 1), (1,), device=x.device))
    crop = x[:, start : start + length].transpose(1, 2)
    return functional.interpolate(crop, size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)


def mild_time_warp(x: torch.Tensor, strength: float = 0.05) -> torch.Tensor:
    factor = 1 + float(torch.empty(1).uniform_(-strength, strength))
    intermediate = max(2, int(x.shape[1] * factor))
    warped = functional.interpolate(x.transpose(1, 2), size=intermediate, mode="linear", align_corners=False)
    return functional.interpolate(warped, size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)


def augment(x: torch.Tensor) -> torch.Tensor:
    return channel_dropout(mild_time_warp(crop_resize(scaling(jitter(x)))))

