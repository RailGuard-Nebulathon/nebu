from railguard.models.classical.regressor import ClassicalRegressor

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = object

if torch is not None:
    from railguard.models.backbones.pooling import AttentionPool
    from railguard.models.backbones.tcn import TemporalConvNet

    class DamageNet(nn.Module):
        def __init__(self, input_channels: int, physics_features: int = 0, hidden_channels: int = 32, heteroscedastic: bool = False) -> None:
            super().__init__()
            self.heteroscedastic = heteroscedastic
            self.encoder = TemporalConvNet(input_channels, hidden_channels)
            self.pool = AttentionPool(hidden_channels)
            self.physics = nn.Sequential(nn.Linear(physics_features, hidden_channels), nn.GELU()) if physics_features else None
            combined = hidden_channels * (2 if physics_features else 1)
            self.head = nn.Linear(combined, 2 if heteroscedastic else 1)

        def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> torch.Tensor:
            temporal = self.pool(self.encoder(x, mask), mask)
            return torch.cat([temporal, self.physics(metadata)], dim=-1) if self.physics is not None and metadata is not None else temporal

        def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, metadata: torch.Tensor | None = None) -> torch.Tensor:
            output = self.head(self.encode(x, mask, metadata))
            if self.heteroscedastic:
                mean = torch.nn.functional.softplus(output[:, :1])
                return torch.cat([mean, output[:, 1:].clamp(-10, 10)], dim=1)
            return torch.nn.functional.softplus(output).squeeze(-1)


def create_damage_baseline(seed: int = 42, target_transform: str = "log1p") -> ClassicalRegressor:
    return ClassicalRegressor("extra_trees", seed=seed, target_transform=target_transform)
