"""Gradient-based influential signal regions."""

import torch


def temporal_saliency(model, x: torch.Tensor, mask: torch.Tensor | None = None, target: int | None = None, metadata: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    signal = x.detach().clone().requires_grad_(True)
    output = model(signal, mask=mask, metadata=metadata)
    if output.ndim == 2:
        selected = output[:, target] if target is not None else output.max(dim=1).values
    else:
        selected = output
    selected.sum().backward()
    importance = signal.grad.abs()
    return importance.mean(dim=-1), importance.mean(dim=1)

