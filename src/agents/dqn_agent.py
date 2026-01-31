"""
Docstring for agents.dqn_agent

Initialize:
  - Q-network with random weights θ
  - Target network with θ⁻ = θ
  - Replay buffer D (empty)
  - Epsilon ε = 1.0 (start fully random)

For each episode:
  Reset environment → get initial state s
  
  For each step:
    # 1. Choose action (ε-greedy)
    if random() < ε:
      a = random_action()
    else:
      a = argmax Q(s, a; θ)  # greedy
    
    # 2. Execute action
    Execute a → observe reward r, next state s', done
    
    # 3. Store experience
    Store (s, a, r, s', done) in buffer D
    
    # 4. Sample random batch from buffer
    Sample minibatch {(sⱼ, aⱼ, rⱼ, s'ⱼ, doneⱼ)} from D
    
    # 5. Compute targets
    For each j:
      if doneⱼ:
        yⱼ = rⱼ  # episode ended
      else:
        yⱼ = rⱼ + γ · max Q(s'ⱼ, a'; θ⁻)  # use target net
    
    # 6. Compute loss and update Q-network
    Loss = Σⱼ (Q(sⱼ, aⱼ; θ) - yⱼ)²
    θ ← θ - α · ∇Loss  # gradient descent
    
    # 7. Periodically update target network
    Every C steps: θ⁻ ← θ
    
    # 8. Decay exploration
    ε ← max(ε_min, ε · decay_rate)
    
    s ← s'  # move to next state
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import copy

from src.agents.agent import BaseAgent
from src.models.networks import MiniGridCNN
from src.models.replay_buffer import ReplayBuffer

class DQNAgent(BaseAgent):
    """
    Deep Q-Network agent for MiniGrid environments.
    
    Uses a convolutional neural network to approximate Q-values from pixel observations,
    with experience replay and target network for stable training.
    """
    def __init__(self, observation_shape: tuple, num_actions: int, config: dict, device: str = 'cpu'):
        """
        Initialize DQN agent.
        
        Args:
            observation_shape: Shape of preprocessed observation (C, H, W)
                             e.g., (1, 320, 320) for grayscale
            num_actions: Number of possible actions in the environment
            config: Configuration dictionary with hyperparameters:
                   - gamma: Discount factor (default: 0.99)
                   - learning_rate: Adam optimizer learning rate (default: 1e-4)
                   - batch_size: Minibatch size for training (default: 32)
                   - target_update_freq: Steps between target network updates (default: 1000)
                   - buffer_capacity: Maximum size of replay buffer (default: 100000)
                   - min_buffer_size: Minimum samples before training starts (default: 1000)
            device: Device to run on ('cpu', 'mps', or 'cuda')
        """
        super().__init__(observation_shape, num_actions, config, device)

        # Extract hyperparameters from config with defaults
        self.gamma = config.get('gamma', 0.99)  # Discount factor
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_freq = config.get('target_update_freq', 1000)  # Steps between target net updates
        self.buffer_capacity = config.get('buffer_capacity', 100000)
        self.min_buffer_size = config.get('min_buffer_size', 1000)  # Start training after this many samples

        self.q_network = MiniGridCNN(observation_shape, num_actions).to(device)

        self.target_network = copy.deepcopy(self.q_network)

        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.criterion = nn.MSELoss()

        self.buffer = ReplayBuffer(capacity=self.buffer_capacity,
                                   observation_shape=observation_shape,
                                   device=device)
        
        self.training_step = 0
        self.loss_history =[]
    
    def choose_action(self, obs: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon: choose random action (exploration)
        With probability (1-epsilon): choose action with highest Q-value (exploitation)
        
        Args:
            obs: Preprocessed observation (C, H, W), e.g., (1, 320, 320)
            epsilon: Exploration rate, 0.0 = fully greedy, 1.0 = fully random
            
        Returns:
            Selected action index (int)
        """
        if np.random.random() < epsilon:
            return np.random.randint(self.num_actions)
        
        # Greedy action Inference Choice
        with torch.no_grad():
            
            #  (C, H, W) -> (1, C, H, W)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

            # Forward pass
            q_values = self.q_network(obs_tensor)

            # Choosing greedy choice
            action = q_values.argmax(dim=1).item()

        return action
    
    def store_transition(self, observation:np.ndarray, action: int, reward:float,
                         next_observation: np.ndarray, done:bool):
        """
        Store a transition in the replay buffer.
        
        Args:
            observation: Current state observation (C, H, W)
            action: Action taken
            reward: Reward received
            next_observation: Next state observation (C, H, W)
            done: Whether episode terminated
        """

        self.buffer.add(observation, action, reward, next_observation, done)
    
    def step(self, obs:np.ndarray, action:int, reward: float, 
             next_obs:np.ndarray, done:bool) -> Dict[str, float]:
        """
        Store transition and perform one training step (if buffer is ready).
        
        This implements one iteration of the DQN training loop:
        1. Store experience in replay buffer
        2. Sample random minibatch
        3. Compute Q-value targets using target network
        4. Update Q-network to minimize Bellman error
        5. Periodically sync target network
        
        Args:
            obs: Current observation (C, H, W)
            action: Action taken
            reward: Reward received
            next_obs: Next observation (C, H, W)
            done: Whether episode terminated
            
        Returns:
            Dictionary with training metrics:
                - 'loss': TD error loss
                - 'q_value': Mean predicted Q-value
                - 'target_q_value': Mean target Q-value
        """
        self.store_transition(obs, action, reward, next_obs, done)

        if not self.buffer.is_ready(max(self.batch_size, self.min_buffer_size)):
            return {'loss':0.0, 'q_value':0.0,'target_q_value':0.0}
        
        batch = self.buffer.sample(self.batch_size)
        observations, actions, rewards, next_observations, dones = batch

        current_q_values = self.q_network(observations)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():

            next_q_values = self.target_network(next_observations)

            max_next_q_values = next_q_values.max(dim=1)[0]

            target_q_values = rewards + (1-dones) *self.gamma*max_next_q_values

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()  # Update weights
        
        # 8. Update target network periodically (every C steps)
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print(f"[DQN] Target network updated at step {self.training_step}")
        
        # 9. Track metrics for logging
        self.loss_history.append(loss.item())
        
        return {
            'loss': loss.item(),
            'q_value': current_q_values.mean().item(),
            'target_q_value': target_q_values.mean().item()
        }
    def save(self, path: str):
        """
        Save agent state to file.
        
        Saves:
        - Q-network weights
        - Target network weights
        - Optimizer state
        - Training step counter
        - Configuration
        
        Args:
            path: File path to save checkpoint (e.g., 'checkpoints/dqn_agent.pt')
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'config': self.config
        }, path)
        print(f"[DQN] Agent saved to {path}")

    def load(self, filepath: str):
        """
        Load agent state from file.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        print(f"[DQN] Agent loaded from {filepath} (training step: {self.training_step})")