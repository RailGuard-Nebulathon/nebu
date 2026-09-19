"""NT-Xent contrastive objective."""

import torch
from torch.nn import functional as functional


def nt_xent_loss(first: torch.Tensor, second: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    first, second = functional.normalize(first, dim=-1), functional.normalize(second, dim=-1)
    embeddings = torch.cat([first, second], dim=0)
    similarities = embeddings @ embeddings.T / temperature
    similarities.fill_diagonal_(torch.finfo(similarities.dtype).min)
    batch = first.shape[0]
    target = torch.cat([torch.arange(batch, 2 * batch, device=first.device), torch.arange(batch, device=first.device)])
    return functional.cross_entropy(similarities, target)

