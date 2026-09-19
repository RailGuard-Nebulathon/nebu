"""Encoder export for explicit SSL runs."""

from pathlib import Path

import torch
from torch import nn


def export_encoder(encoder: nn.Module, path: str | Path, metadata: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder_state": encoder.state_dict(), "metadata": metadata}, target)
    return target

