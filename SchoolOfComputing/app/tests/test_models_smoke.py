import pytest

torch = pytest.importorskip("torch")

from railguard.models.backbones.channel_attention import ChannelAttention
from railguard.models.backbones.cnn1d import MultiScaleCNN1D
from railguard.models.backbones.pooling import MaskedMeanPool
from railguard.models.backbones.tcn import TemporalConvNet
from railguard.models.backbones.transformer import TemporalTransformer
from railguard.models.acv.relational import ACVRelationalNet
from railguard.models.corrugation.dual_domain import CorrugationNet
from railguard.models.door.detector import DoorNet
from railguard.models.shm.damage_regressor import DamageNet


@pytest.mark.parametrize("length", [31, 64])
def test_backbone_variable_length(length: int) -> None:
    x = torch.randn(3, length, 6)
    mask = torch.ones(3, length, dtype=torch.bool)
    mask[0, -5:] = False
    for model in (MultiScaleCNN1D(6, 24), TemporalConvNet(6, 24), TemporalTransformer(6, 24, heads=4, layers=1)):
        encoded = model(x, mask)
        assert encoded.shape == (3, length, 24)
        assert MaskedMeanPool()(encoded, mask).shape == (3, 24)
    attended, weights = ChannelAttention(6)(x, torch.ones(3, 6, dtype=torch.bool))
    assert attended.shape == x.shape and weights.shape == (3, 6)


def test_task_models_forward_backward() -> None:
    x = torch.randn(2, 48, 6, requires_grad=True)
    mask = torch.ones(2, 48, dtype=torch.bool)
    outputs = [
        DoorNet(6, 16)(x, mask, torch.tensor([0, 1])),
        CorrugationNet(6, 16)(x, mask),
        DamageNet(6, physics_features=3, hidden_channels=16, heteroscedastic=True)(x, mask, torch.randn(2, 3)),
    ]
    acv = ACVRelationalNet(6, 16, heads=4)(torch.randn(2, 4, 48, 6), car_mask=torch.ones(2, 4, dtype=torch.bool))
    outputs.append(acv)
    assert [tuple(output.shape) for output in outputs] == [(2, 2), (2, 3), (2, 2), (2, 4)]
    sum(output.float().mean() for output in outputs).backward()

