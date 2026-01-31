import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    "Experience replay buffer for DQN"

    def __init__(self, capacity: int, observation_shape: tuple, device: str = 'cpu'):
        """
        Args:
            capacity: Maximum number of transitions to store
            observation_shape: Shape of observations (C, H, W) (image)
            device: Device to store tensors on ('cpu' or 'mps' or cuda)
        """
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.device = device
        self.position = 0
        self.size = 0

        self.observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *observation_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def add(self, observation, action, reward, next_observation, done):
        """
        Add a transition to the buffer
        
        Args:
            observation: Current state observation
            action: Action taken
            reward: Reward received
            next_observation: Next state observation
            done: Whether episode terminated
        """
        self.observations[self.position] = observation
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_observation
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (observations, actions, rewards, next_observations, dones)
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer. Currently at {self.size} and need {batch_size}")
        
        indices = np.random.choice(self.size, batch_size, replace=False)

        # Convert to torch tensors and move to decice

        observations = torch.FloatTensor(self.observations[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_observations = torch.FloatTensor(self.next_observations[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return observations, actions, rewards, next_observations, dones
    
    def __len__(self):
        """current buffer size"""
        return self.size
    def is_ready(self, batch_size: int) -> bool:
        """Checks if the buffer has enough samples for training"""
        return self.size >= batch_size

class PrioritizedReplayBuffer(ReplayBuffer):
    
    def __init__(self, capacity: int, observation_shape: tuple,
                 device: str = 'cpu', alpha: float = 0.6, beta : float = 0.4):
        """
        Args:
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling weight (increases to 1 over training)
        """
        super().__init__(capacity, observation_shape, device)

        self.alpha = alpha
        self.beta = beta
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self,observation, action, reward, next_observation, done):
        """Add transition with maximum priority"""
        super().add(observation, action, reward, next_observation, done)

        self.priorities[self.position - 1] = self.max_priority
    
    def sample(self, batch_size:int):
        """Sample batch with priorities"""

        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer. Currently at {self.size} and need {batch_size}")
        
        priorities = self.priorities[:self.size]**self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(self.size, batch_size, replace=False, p=probabilities)

        #Calculate impance sampling weights
        weights = (self.size * probabilities[indices] ** (-self.beta))
        weights /= weights.max()

        observations = torch.FloatTensor(self.observations[indices]).to(self.device)
        actions = torch.LongTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_observations = torch.FloatTensor(self.next_observations[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        
        return observations, actions, rewards, next_observations, dones, torch.FloatTensor(weights).to(self.device)
    
    def update_priorities(self, indices, priorities):
        """update prioriteis for sampmled transitions"""

        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())



