import pytest

torch = pytest.importorskip("torch")

from railguard.ssl.augmentations import augment, span_mask
from railguard.ssl.contrastive import nt_xent_loss


def test_ssl_augmentations_and_loss() -> None:
    x = torch.randn(4, 32, 3)
    assert augment(x).shape == x.shape
    corrupted, mask = span_mask(x)
    assert corrupted.shape == x.shape and mask.any()
    assert nt_xent_loss(torch.randn(4, 8), torch.randn(4, 8)).isfinite()

