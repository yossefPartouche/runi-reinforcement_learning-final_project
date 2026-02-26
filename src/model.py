from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

class MiniGridCNN(nn.Module):
    """
    CNN wrapper for the network(s) used by the agents
    :param observation_shape: (H, W, C) - e.g. (84, 84, 1)
    :return: Q values for each action
    """

    def __init__(self, observation_shape: np.ndarray, num_actions: int):
        super().__init__()

        # parse input dimensions
        height, width, num_channels = observation_shape
        
        # CONV layers:
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # find out dimensions of feature map after CONV
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, height, width)
            conv_output = self.conv(dummy_input)
            self.flat_size = conv_output.view(1, -1).size(1)
        
        # FC layers:
        self.fc = nn.Sequential(
           nn.Flatten(),
           nn.Linear(self.flat_size, 512),
           nn.ReLU(),
           nn.Linear(512, num_actions)
        )
        
    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for A2C
    - shared conv feature extractor
    - actor head: outputs action probabilities pi(a|s)
    - critic head: outputs state value V(s)
    """
    def __init__(self, observation_shape: np.ndarray, num_actions: int):
        """
        :param observation_shape: (H, W, C) - e.g. (84, 84, 1)
        """
        super().__init__()
        height, width, num_channels = observation_shape

        # shared conv feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # find out dimensions of feature map after CONV        
        with torch.no_grad():
            dummy_input = torch.zeros(1, num_channels, height, width)
            conv_output = self.conv(dummy_input)
            flat_size = conv_output.reshape(1, -1).size(1)

        # shared FC layer
        self.fc_shared = nn.Linear(flat_size, 512)
        
        # actor head (policy): outputs action probabilities
        self.actor = nn.Linear(512, num_actions)
        
        # critic head (value): outputs state value V(s)
        self.critic = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass thru both actor + critic
        :param x: observation tensor (B, C, H, W)
        :return action_logits: logits for action distribution (B, num_actions)
        :return state_value: value of current state (B, 1)
        """
        # shared feature extraction
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)  # flatten
        x = F.relu(self.fc_shared(x))
        
        # todo: logits
        # actor: action probabilities (logits)
        action_logits = self.actor(x)
        
        # critic: state value
        state_value = self.critic(x)
        
        return action_logits, state_value
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probability distribution"""
        action_logits, _ = self.forward(x)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted state value"""
        _, state_value = self.forward(x)
        return state_value