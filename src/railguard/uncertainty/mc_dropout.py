"""Monte-Carlo dropout inference."""

import torch


def mc_dropout_predict(model, *args, passes: int = 20, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    model.train()
    with torch.no_grad():
        outputs = torch.stack([model(*args, **kwargs) for _ in range(passes)])
    return outputs.mean(dim=0), outputs.std(dim=0)

