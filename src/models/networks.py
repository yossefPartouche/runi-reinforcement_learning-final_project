import numpy as np
import torch.nn as nn
import torch

class MiniGridCNN(nn.Module):
    """CNN for processing MiniGrid observations and outputting Q-values"""

    def __init__(self, input_shape: np.ndarray, num_actions: int):
        """
        Args:
            input_shape: Shape of preprocessed observation (C, H, W) e.g., (1, 320, 320)
            num_actions: Number of possible actions (3 or 5)
        """
        super().__init__()
        # todo: define Conv layers according to input_shape
        # todo: define FC layers to output num_actions
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