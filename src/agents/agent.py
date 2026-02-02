from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict


class Agent(ABC):
    """
        Abstract base class for all RL agents.
        
        REQUIRED methods (all agents must implement):
        - choose_action()
        - update()
        - save()
        - load()
        
        OPTIONAL methods (override if needed):
        - step()              <- Only for off-policy (DQN)
        - store_transition()  <- Only for replay buffer (DQN)
        """
    
    
    @abstractmethod
    def choose_action(self, obs, epsilon: float =0.0) -> int:
        """
        Select action given observation
        
        Args:
            obs: Preprocessed observation
            **kwargs: Algorithm-specific parameters
                - epsilon (float): For DQN exploration
                - deterministic (bool): For evaluation
            
        Returns:
            Selected action index (int)
        
        Example:
            # DQN
            action = agent.choose_action(obs, epsilon=0.1)
            
            # A2C
            action = agent.choose_action(obs)
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):  # ← Add this (was missing!)
        """
        Update agent parameters based on experience.
        
        Args:
            *args, **kwargs: Algorithm-specific data
                - DQN: batch (dict) from replay buffer
                - A2C: trajectories (list) from episode
        
        Returns:
            dict: Loss values and metrics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save agent state to file."""
        pass

    @abstractmethod
    def load(self, filepath:str):
        """Load agent parameters."""
        pass

    # OPTIONAL

    def store_transition(self, observation, action, reward, next_observation, done):  # ← Remove @abstractmethod
        """
        Optional: Store a transition in memory (for off-policy algorithms).
        
        Only DQN needs this - A2C updates per episode and doesn't store.
        
        Args:
            observation: Current state
            action: Action taken
            reward: Reward received
            next_observation: Next state
            done: Whether episode terminated
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement store_transition(). "
            "This is only needed for replay buffer-based algorithms."
        )
    
    def step(self, obs, action: int, reward: float, next_obs, done):  # ← Remove @abstractmethod
        """
        Optional: Store and train in one call (for off-policy algorithms).
        
        This combines store_transition() + update() in one method.
        Only needed for algorithms that update after every environment step.
        
        Args:
            obs: Current state
            action: Action taken
            reward: Reward received
            next_obs: Next state
            done: Whether episode terminated
            
        Returns:
            dict: Training metrics (loss, q_values, etc.)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement step(). "
            "This is only needed for per-step update algorithms (DQN)."
        )
