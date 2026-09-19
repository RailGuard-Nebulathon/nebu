"""Variable-length tensor datasets and padding."""

from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, sequences: list[torch.Tensor], targets: list[torch.Tensor], metadata: list[torch.Tensor] | None = None) -> None:
        self.sequences, self.targets, self.metadata = sequences, targets, metadata

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.targets[index], self.metadata[index] if self.metadata is not None else None


def pad_collate(batch):
    sequences, targets, metadata = zip(*batch, strict=True)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    values = pad_sequence(sequences, batch_first=True)
    mask = torch.arange(values.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    meta = torch.stack(metadata) if metadata[0] is not None else None
    return values, torch.stack(targets), mask, meta

