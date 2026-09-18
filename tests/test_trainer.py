import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader

from railguard.models.door.detector import DoorNet
from railguard.training.datasets import SequenceDataset, pad_collate
from railguard.training.trainer import Trainer, TrainerConfig


def test_one_synthetic_epoch() -> None:
    sequences = [torch.randn(24 + i, 4) for i in range(6)]
    targets = [torch.tensor(i % 2) for i in range(6)]
    metadata = [torch.tensor(i % 2) for i in range(6)]
    loader = DataLoader(SequenceDataset(sequences, targets, metadata), batch_size=3, collate_fn=pad_collate)
    trainer = Trainer(DoorNet(4, 8), nn.CrossEntropyLoss(), TrainerConfig(epochs=1))
    assert trainer.dry_run(loader) > 0
    assert len(trainer.fit(loader)) == 1

