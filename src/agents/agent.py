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
        self.state_shape = state_shape
        self.num_actions = num_actions
        # todo: init model + buffer
    
    @abstractmethod
    def choose_action(self, obs, epsilon=0.0) -> int:
        """
        Select an action given the current state.
        
        Args:
            state: Current observation
            epsilon: Exploration rate (for epsilon-greedy)
            
        Returns:
            Selected action (int)
        """
        pass
    
    @abstractmethod
    def step(self, obs, action: int, reward: float, next_obs, done):
        """
        Perform one training step.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
            
        Returns:
            Training metrics (dict)
        """
        pass
        
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, filepath):
        """Load agent parameters."""
        pass