import numpy as np
import torch

class ReplayBuffer:
    """Replay buffer class for use by agents during training"""

    def __init__(self, capacity: int, obs_shape: np.ndarray, device: torch.device):
        self.capacity = capacity
        self.device = device
        # todo: init arrays (states, actions, rewards, etc)
        self.size = 0
        self.ptr = 0

    # todo: type hints
    def add(self, obs, action, reward, next_obs, done):
        """Store a transition."""
        pass
    
    # todo: type hints
    def sample(self, batch_size: int):
        """Returns a batch of ___""" 
        pass