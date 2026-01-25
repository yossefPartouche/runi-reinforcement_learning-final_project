from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict

# todo: docstrings
# todo: type hints
class BaseAgent(ABC):
    """Base class for agents"""

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        self.config = config
        self.device = device
        # todo: init model + buffer
    
    def choose_action(self, obs, epsilon=0.0) -> int:
        pass
    
    def step(self, obs, action: int, reward: float, next_obs, done):
        """Stores experience + updates the model"""
        pass
        
    def save(self, path: str):
        pass