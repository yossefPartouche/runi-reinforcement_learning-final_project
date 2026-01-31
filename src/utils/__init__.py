"""
Utility functions and classes for training.
"""
import torch
import numpy as np
import random

from .logging import Logger

def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Make PyTorch deterministic (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """
    Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon GPU
    else:
        return torch.device('cpu')


__all__ = ['Logger', 'set_random_seed', 'get_device']