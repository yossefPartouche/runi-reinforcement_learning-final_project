from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict

# todo: docstrings
# todo: type hints
class BaseAgent(ABC):
    """Abstract base class for all RL agents"""

    def __init__(self, observation_shape: tuple, num_actions: int, config: dict, device: str = 'cpu'):
        """
        Args:
            observation_shape: Shape of preprocessed observation (C, H, W)
            num_actions: Number of possible actions
            config: Configuration dictionary with hyperparameters
            device: Device to run on ('cpu' or 'mps' or 'cuda')
        """
        self.config = config
        self.device = device
        self.state_shape = observation_shape
        self.num_actions = num_actions
        self.training_step = 0
        # todo: init model + buffer
    
    @abstractmethod
    def choose_action(self, obs, epsilon: float =0.0) -> int:
        """
        Select action given observation
        
        Args:
            observation: Preprocessed observation
            epsilon: Exploration rate (0 = greedy, 1 = random)
            
        Returns:
            Selected action index
        """
        pass

    @abstractmethod
    def store_transition(self, observation, action, reward, next_observation, done):
        """
        Store a transition in memory
        
        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode terminated
        """
        pass
    
    @abstractmethod
    def step(self, obs, action: int, reward: float, next_obs, done):
        """
        Store a transition in memory
        
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