import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F

class MiniGridCNN(nn.Module):
    """CNN for processing MiniGrid observations and outputting Q-values"""

    def __init__(self, input_shape: tuple, num_actions: int):
        """
        Args:
            input_shape: Shape of preprocessed observation (C, H, W) e.g., (1, 320, 320)
            num_actions: Number of possible actions (3 or 5)
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_output(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
    # todo: verify this is needed
    def _get_conv_output(self, shape):
        """Helper funtion to compute the flattened layer before FC network"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv(dummy_input)
            return int(np.prod(output.size()))
        
    def forward(self, x):
        """
        Args:
            x: Batch of observations (B, C, H, W)
        Returns:
            Q-values for each action (B, num_actions)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ActorCriticNetwork(nn.Module):
    """
    Combined Actor-Critic network for A2C.
    
    Architecture:
    - Shared convolutional feature extractor
    - Actor head: outputs action probabilities π(a|s)
    - Critic head: outputs state value V(s)
    """
    def __init__(self, observation_shape, num_actions):
        """
        Args:
            observation_shape: (C, H, W) - e.g., (1, 84, 84)
            num_actions: Number of discrete actions
        """
        super(ActorCriticNetwork, self).__init__()
        channels,height, width = observation_shape

        # Shared feature extractor (convolutional layers)
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flattened size
        def conv2d_size_out(size, kernel_size, stride):
            return (size - kernel_size) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(width, 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(height, 8, 4), 4, 2), 3, 1)
        self.flat_size = convw * convh * 64

        # Shared fully connected layer
        self.fc_shared = nn.Linear(self.flat_size, 512)
        
        # Actor head (policy): outputs action probabilities
        self.actor = nn.Linear(512, num_actions)
        
        # Critic head (value): outputs state value V(s)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        """
        Forward pass through both actor and critic.
        
        Args:
            x: Observation tensor (B, C, H, W)
            
        Returns:
            action_logits: Logits for action distribution (B, num_actions)
            state_value: Value of current state (B, 1)
        """
        # Shared feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_shared(x))
        
        # Actor: action probabilities (logits)
        action_logits = self.actor(x)
        
        # Critic: state value
        state_value = self.critic(x)
        
        return action_logits, state_value
    
    def get_action_probs(self, x):
        """Get action probability distribution."""
        action_logits, _ = self.forward(x)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, x):
        """Get state value estimate."""
        _, state_value = self.forward(x)
        return state_value