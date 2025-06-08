import random
import numpy as np
import torch


def set_seeds(seed: int) -> None:
    """
    Sets seeds for random, numpy, and torch to ensure reproducibility.

    Args:
        seed (int): The seed to use for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
