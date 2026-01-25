import random
import numpy as np
import torch
import os
import csv
import time
from typing import Dict 

def set_random_seed(seed: int) -> None:
    """
    Sets seed for random number generator
    - cmopatible with CPU/GPU
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_device() -> torch.device:
    """Returns the device that runs training (GPU/CPU)"""
    processor_type = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(processor_type)


class Logger:
    """Logging service for the system"""

    def __init__(self, config: Dict):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_directory = os.path.join("results", f"{config['algo']}_{timestamp}") # todo: rename algo
        os.makedirs(self.log_directory, exist_ok=True)
        
        # save config
        config_path = os.path.join(self.log_directory, "config.txt")
        with open(config_path, "w") as f:
            f.write(str(config))

        #  init log file
        self.csv_path = os.path.join(self.log_directory, "log.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["episode", "reward", "steps", "epsilon"])

    def log(self, episode: int, reward: float, steps: int, epsilon: float) -> None:
        """Adds single row to log file"""
        self.writer.writerow([episode, reward, steps, epsilon])
        self.csv_file.flush()