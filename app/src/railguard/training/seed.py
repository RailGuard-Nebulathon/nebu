"""Deterministic random seed setup."""

import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    except ImportError:
        pass

