from typing import Tuple
import numpy as np
import torch
from torch import Tensor

class ReplayBuffer:
    """
    Replay buffer class for use by agents during training
    """

    def __init__(self, capacity: int, obs_shape: np.ndarray, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.occupancy = 0  # number of stored transitions
        self.index = 0      # next index to store transition

        # init stoaage arrays
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.action = np.zeros((capacity, 1), dtype=np.int64)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done   = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """Store a transition in the buffer"""
        
        self.obs[self.index] = obs
        self.next_obs[self.index] = next_obs
        self.action[self.index] = action
        self.reward[self.index] = reward
        self.done[self.index] = done

        # update index and occupancy
        self.index = (self.index + 1) % self.capacity
        self.occupancy = min(self.occupancy + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Sample a random batch of transitions from the buffer"""
        idxs = np.random.randint(0, self.occupancy, size=batch_size)

        # sample + normalize (divide by 255) + handle shape convesions
        obs_sample = torch.as_tensor(self.obs[idxs], device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        next_obs_sample = torch.as_tensor(self.next_obs[idxs], device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        actions_sample = torch.as_tensor(self.action[idxs], device=self.device).long()
        reward_sample = torch.as_tensor(self.reward[idxs], device=self.device).float()
        done_sample = torch.as_tensor(self.done[idxs], device=self.device).float()

        return (
            obs_sample, 
            next_obs_sample, 
            actions_sample, 
            reward_sample, 
            done_sample
        )
    
    def __len__(self):
        return self.occupancy

        