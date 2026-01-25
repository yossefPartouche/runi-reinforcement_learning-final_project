import numpy as np
import torch.nn as nn

class MiniGridCNN(nn.Module):
    """CNN wrapper for the network(s) used by the agents"""

    def __init__(self, input_shape: np.ndarray, num_actions: int):
        super().__init__()
        # todo: define Conv layers according to input_shape
        # todo: define FC layers to output num_actions
        pass

    # todo: verify this is needed
    def forward(self, x):
        pass