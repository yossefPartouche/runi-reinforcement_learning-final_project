import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.agents.agent import Agent
from src.models.networks import ActorCriticNetwork
import os

class A2CAgent(Agent):
    """
    A2C (Advantage Actor-Critic) Agent.
    
    Key components:
    - Actor: Learns policy π(a|s) (what action to take)
    - Critic: Learns value function V(s) (how good is the state)
    - Advantage: A(s,a) = R + γV(s') - V(s) (how much better is action a)
    """
    
    def __init__(self, observation_shape, num_actions, config, device):
        """
        Initialize A2C agent.
        
        Args:
            observation_shape: (C, H, W) tuple
            num_actions: Number of discrete actions
            config: Configuration dictionary
            device: torch device (cuda/mps/cpu)
        """
        super().__init__()
        
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.config = config
        self.device = device
        
        # Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.value_loss_coef = config.get('value_loss_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        
        # Actor-Critic network
        self.network = ActorCriticNetwork(observation_shape, num_actions).to(device)
        
        # Optimizer (single optimizer for both actor and critic)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )
        
        # Training step counter
        self.steps = 0
        
        print(f"A2C Agent initialized")
        print(f"  - Network parameters: {sum(p.numel() for p in self.network.parameters()):,}")
        print(f"  - Learning rate: {self.learning_rate}")
        print(f"  - Gamma: {self.gamma}")
        print(f"  - Value loss coefficient: {self.value_loss_coef}")
        print(f"  - Entropy coefficient: {self.entropy_coef}")
    
    def choose_action(self, observation, epsilon=None):
        """
        Select action using the current policy (stochastic).
        
        Args:
            observation: Preprocessed observation (C, H, W)
            epsilon: Not used in A2C (exploration is built into policy)
            
        Returns:
            action: Selected action (int)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_logits, _ = self.network(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)
            
            # Sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            
            return action.item()
    
    def update(self, trajectories):
        """
        Update actor and critic using collected trajectories.
        
        Args:
            trajectories: List of (state, action, reward, next_state, done) tuples
            
        Returns:
            dict: Loss components (actor_loss, critic_loss, entropy, total_loss)
        """
        if len(trajectories) == 0:
            return {}
        
        # Unpack trajectories
        states, actions, rewards, next_states, dones = zip(*trajectories)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Forward pass
        action_logits, state_values = self.network(states)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Get values for next states
        with torch.no_grad():
            _, next_state_values = self.network(next_states)
            next_state_values = next_state_values.squeeze()
        
        state_values = state_values.squeeze()
        
        # Compute TD target and advantage
        # TD target: R + γ * V(s') * (1 - done)
        td_target = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Advantage: A(s,a) = TD_target - V(s)
        advantage = td_target - state_values
        
        # Actor loss (policy gradient)
        # L_actor = -log π(a|s) * A(s,a)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantage.detach()).mean()
        
        # Critic loss (MSE between V(s) and TD target)
        # L_critic = (V(s) - TD_target)²
        critic_loss = F.mse_loss(state_values, td_target.detach())
        
        # Entropy bonus (encourages exploration)
        # H(π) = -Σ π(a|s) * log π(a|s)
        entropy = action_dist.entropy().mean()
        
        # Total loss
        total_loss = (
            actor_loss + 
            self.value_loss_coef * critic_loss - 
            self.entropy_coef * entropy
        )
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.steps += 1
        
        # Return loss components
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item(),
            'mean_advantage': advantage.mean().item()
        }
    
    def save(self, filepath):
        """Save agent state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'config': self.config
        }, filepath)
        
        print(f"✅ A2C checkpoint saved to {filepath}")
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        
        print(f"[A2C] Agent loaded from {filepath} (training step: {self.steps})")