"""
Module containing a function to set the random seed for
reproducibility
"""
import random
import numpy as np
import torch


def set_random_seed(random_seed: int = 42) -> None:
    """
    Set the random seed for reproducibility. The seed is set for the
    random library, the numpy library and the pytorch library.

    Parameters
    ----------
    random_seed : int, optional
        The random seed to use for reproducibility, by default 42.
    """
    random.seed(random_seed)

    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
