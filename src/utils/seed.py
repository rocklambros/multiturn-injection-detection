"""Global seed management for reproducibility."""

import os
import random
import numpy as np
import torch


def set_global_seed(seed=42):
    """Set all random seeds for reproducibility.

    Args:
        seed: Integer seed value. Default 42.

    Side effects:
        Sets PYTHONHASHSEED env var, seeds random, numpy, and torch RNGs,
        configures cuDNN for deterministic behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
