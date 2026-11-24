"""One call seeds every random number generator the core uses (Python, numpy, torch's action sampling)."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seeds the global RNGs and returns a dedicated generator for minibatch shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return torch.Generator().manual_seed(seed)
