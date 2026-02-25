import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.agents.agent import Agent
from src.models.networks import ActorCriticNetwork
import os

class PPOAgent(Agent):

    def __init__(self, observation_shape, num_actions, config, device):
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.config = config
        self.device = device

        # General Hyperparameters
        self.gamma = config.get('gamma', 0.99)
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.clip_eps = config.get('clip_eps', 0.1)
        self.batch_size = config.get('batch_size', 32)
        self.gae = config.get('gae_lambda', 0.95)

        # PPO Hyperparameters
        self.network = ActorCriticNetwork(observation_shape, num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), 
                                          lr=self.learning_rate)

    
    def choose_action(self, observation, epsilon=None):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action_logits, _ = self.network(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)

            # sample action from probability distribution
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()

            return action.item()

    def update(self, trajectories):
        
        states, actions, rewards, next_states, dones, log_probs = zip(*trajectories)
        
        # Enable Network usage
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_prob = torch.FloatTensor(log_probs).to(self.device)
        
        # Forward pass
        action_logits, state_values = self.network(states)
        action_probs = F.softmax(action_logits, dim=-1)

        with torch.no_grad():
            _, next_state_values = self.network(next_states)
            next_state_values = next_state_values.squeeze()
        
        state_values = state_values.squeeze()

        td_target = rewards + self.gamma*next_state_values*(1-dones)

        advantage = td_target - state_values

        action_dist = torch.distributions.Categorical(action_probs)

        new_log_probs = action_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - old_log_prob)

        #PPO surrogate loss
        clip_eps = self.config.get('clip_eps', 0.2)
        surr1 = ratio*advantage
        surr2 = torch.clamp(ratio, 1-clip_eps, 1+ clip_eps)*advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(state_values, td_target.detach())


        loss = policy_loss + 0.5*value_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': loss.item(),
            'mean_advantage': advantage.mean().item()
        }
    
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exists_ok=True)

        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'config': self.config
        }, filepath)

        print(f"✅ PPO checkpoint saved to {filepath}")
    
    def load(self, filepath):
        """Load agent state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        
        print(f"[PPO] Agent loaded from {filepath} (training step: {self.steps})")





    