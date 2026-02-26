from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.buffer import ReplayBuffer
from src.experiments import PROJECT_BASE_CONFIG
from src.model import ActorCriticNetwork, MiniGridCNN


class BaseAgent(ABC):
    """Base class for agents"""
    gamma: float
    epsilon: float
    learning_rate: float
    device: torch.device
    config: Dict

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        self.device = device
    
    @abstractmethod
    def choose_action(self, obs, epsilon=0.0) -> int:
        raise NotImplementedError
    
    def step(self, obs, action: int, reward: float, next_obs, done):
        """
        Optional method: stores experience + updates the model in single call (off-policy)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement step(). "
            "This is only needed for per-step update algorithms (e.g. DQN)."
        )

    @abstractmethod
    def update(self, *args, **kwargs) -> Dict:
        """
        Update agent parameters
        DQN: called via step()
        A2C: called with trajectory list
        """
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path: str):
        raise NotImplementedError
    
    @abstractmethod
    def load(self, filepath: str):
        """Load agent params from checkpoint"""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return self.config.get("algo", "BaseAgent")


class RandomAgent(BaseAgent):
    """Dummy agent for testing infrastructure"""
    def __init__(self, config, obs_shape, num_actions, device):
        super().__init__(config, obs_shape, num_actions, device)
        self.num_actions = num_actions

    def choose_action(self, obs, epsilon=0.0) -> int:
        return np.random.randint(0, self.num_actions)

    def step(self, obs, action, reward, next_obs, done):
        pass # do nothing

    def update(self, *args, **kwargs):
        pass # do nothing

    def save(self, path):
        pass # do nothing

    def load(self, filepath: str):
        pass # do nothing


class DQNAgent(BaseAgent):
    """
    DQN Agent with Target Network and Replay Buffer.
    """
    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
        self.num_actions = num_actions

        self.config = PROJECT_BASE_CONFIG.copy()
        if config:
            self.config.update(config)
        
        # hyperparams
        self.gamma: float = self.config.get("gamma")
        self.epsilon: float = self.config.get("epsilon_start")
        self.epsilon_min: float = self.config.get("epsilon_min")
        self.epsilon_decay: float = self.config.get("epsilon_decay")
        self.learning_rate: float = self.config.get("learning_rate")
        self.batch_size: int = self.config.get("batch_size")
        self.min_buffer_size: int = self.config.get("min_buffer_size")
        self.training_freq: int = self.config.get("training_freq")
        self.target_update_freq: int = self.config.get("target_update_freq")

        # init networks:
        # policy net (main trained network)
        self.policy_net = MiniGridCNN(observation_shape=obs_shape, num_actions=num_actions).to(device)

        # target net (stable reference)
        self.target_net = MiniGridCNN(observation_shape=obs_shape, num_actions=num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # target net never in training mode

        # optimizer + loss # todo: consider other optimizer/loss options?
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=self.learning_rate) 
        self.loss_fn = nn.MSELoss()
        # self.loss_fn = nn.SmoothL1Loss()

        # memory (replay buffer)
        buffer_capacity = self.config.get("buffer_capacity") # default to 100,000
        self.memory = ReplayBuffer(capacity=buffer_capacity, obs_shape=obs_shape, device=device)
        
        # init step counter
        self.steps_done = 0

    def choose_action(self, obs: np.ndarray, epsilon: float| None = None) -> int:
        """
        Epsilon-greedy action selection
        :param obs: current observation (np array)
        :param epsilon: exploration probability
        :return: action index
        """
        if epsilon is None:
            epsilon = self.epsilon

        # exploration
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # exploitation
        with torch.no_grad():
            # process state:
            state = torch.as_tensor(data=obs, device=self.device)   # np -> tensor
            state= state.float().div(255.0)                         # normalize (divide by 255)
            state = state.permute(2, 0, 1).unsqueeze(0)             # (height, width, channel) -> (1, channel, height, width)
            
            q_values = self.policy_net(state)   # get Q vals from POLICY net
            best_action = q_values.argmax()     # pick best action
            action_idx = best_action.item()
            return action_idx

    def step(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """
        Execute single step in the env:
        1. store transition
        2. train (if buffer has enough data)
        3. decay epsilon
        """
        self.steps_done += 1
        
        # store step
        self.memory.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        
        # train
        if (len(self.memory) >= self.min_buffer_size) and (self.steps_done % self.training_freq == 0):
            self.update()
            
        # update target net logic:
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self):
        """
        Core DQN update logic
        """
        # sample batch from buffer
        state_batch, next_state_batch, action_batch, reward_batch, done_batch = self.memory.sample(self.batch_size)
        
        # calculate current Q values Q(s, a):
        all_q_vals = self.policy_net(state_batch)                       # trigger POLICY net forward, get Q vals
        current_q_vals = all_q_vals.gather(dim=1, index=action_batch)   # filter Q vals for specific action(s) taken 
        
        # calculate Q_target ( max Q(s', a') , from target net):
        with torch.no_grad():
            #  
            all_next_q_vals = self.target_net(next_state_batch)   # trigger TARGET net forward, get next Q vals
            next_q_vals = all_next_q_vals.max(1)[0].unsqueeze(1)  # filter next Q vals for specific action(s) taken 
            
            # Bellman eq: R + gamma * max(Q(s')) * (1 - done)
            target_q_vals = reward_batch + (self.gamma * next_q_vals * (1 - done_batch))
            
        # calculate loss
        loss = self.loss_fn(current_q_vals, target_q_vals)
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()

        # todo: necessary?
        # optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            parameters=self.policy_net.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()
    
    def save(self, path: str) -> None:
        """
        Save agent snapshot at checkpoint
        - policy net
        - target net
        - optimizer state
        - step counter
        - epsilon
        - config
        """
        agent_dict = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'config': self.config,
        }
        torch.save(agent_dict, path)
        print(f"[DQN] Checkpoint saved to {path}")

    def load(self, filepath: str) -> None:
        """
        Load agent state from a checkpoint file.
        Restores networks, optimizer, step counter, and epsilon so training can resume exactly.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon    = checkpoint['epsilon']
        print(f"[DQN] Checkpoint loaded from {filepath} (step: {self.steps_done}, ε: {self.epsilon:.4f})")


class A2CAgent(BaseAgent):
    """
    A2C (Actor-Critic) Agent
    Components:
    - Actor: learns policy π(a|s) — what action to take
    - Critic: learns value function V(s) — how good is the state
    - Advantage: A(s,a) = R + γ*V(s') - V(s) — how much better is action a vs avg

    Obs format: receives uint8 (H, W, C), normalizes + transposes internally
    No replay buffer — updates once per episode (or other freq)
    """

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        """
        :param obs_shape:   (H, W, C) — e.g. (84, 84, 1)
        """
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)

        self.num_actions = num_actions
        self.config = config

        # hyperparams
        self.gamma          = config.get('gamma', 0.99)
        self.learning_rate  = config.get('learning_rate', 1e-4)
        self.value_loss_coefficient = config.get('value_loss_coefficient', 0.5)
        self.entropy_coefficient   = config.get('entropy_coefficient', 0.01)
        self.max_grad_norm  = config.get('max_grad_norm', 0.5)

        # exploitation only (A2C stochastic not epsilon-greedy)
        self.epsilon = 0.0  # attribute kept to adhere to interface

        # step counter
        self.steps_done = 0

        # Actor-Critic network — receives (H, W, C)
        self.network = ActorCriticNetwork(observation_shape=obs_shape, num_actions=num_actions).to(device)

        # single optimizer for both actor and critic heads
        self.optimizer = torch.optim.Adam(params=self.network.parameters(), lr=self.learning_rate)

    def choose_action(self, obs: np.ndarray, epsilon: float | None = None) -> int:
        """
        Sample action from current policy π(a|s) (stochastic during training)
        If epsilon=0 (eval mode), act greedily (argmax action probs)
        
        :param obs: uint8 ndarray (H, W, C)
        :param epsilon: ignored during training (A2C explores via policy entropy) # todo
        - pass 0.0 explicitly to get deterministic/greedy inference behaviour
        :return: action index
        """
        with torch.no_grad():
            # ingest observation
            obs_tensor = torch.as_tensor(obs, device=self.device).float()   # uint8 -> float32
            obs_tensor = obs_tensor.div(255.0)                              # normalize to [0,1]
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)           # (H, W, C) -> (1, C, H, W)

            # get action probs from network
            action_logits, _ = self.network(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)

            # greedy for eval (epsilon=0.0), stochastic for training
            if epsilon == 0.0:
                return int(action_probs.argmax(dim=-1).item())

            action_dist = torch.distributions.Categorical(action_probs)
            return int(action_dist.sample().item())

    def update(self, trajectories: List[Tuple[np.ndarray, int, float, np.ndarray, bool]]) -> Dict:
        """
        Update actor + critic from full episode (or some freq)

        :param trajectories: list of trajectory data objects (obs, action, reward, next_obs, done)
                          obs, next_obs are uint8 (H, W, C) ndarrays
        :return: update values object (actor_loss, critic_loss, entropy, total_loss, mean_advantage)
        """
        if len(trajectories) == 0:
            return {}

        trajectories = [t[:5] for t in trajectories]    # handle 6th field (PPO interface) # todo
        states, actions, rewards, next_states, dones = zip(*trajectories)

        # states: uint8 (N, H, W, C) -> float32 (N, C, H, W), normalized
        states      = torch.as_tensor(np.array(states),      device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        next_states = torch.as_tensor(np.array(next_states), device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        
        actions     = torch.tensor(actions, dtype=torch.long,    device=self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones       = torch.tensor(dones,   dtype=torch.float32, device=self.device)

        # fwd pass thru shared network
        action_logits, state_vals = self.network(states)
        action_probs = F.softmax(action_logits, dim=-1)
        state_vals = state_vals.squeeze()

        # bootstrap next state values from critic
        with torch.no_grad():
            _, next_state_vals = self.network(next_states)
            next_state_vals = next_state_vals.squeeze()

        # TD target: R + γ * V(s') * (1 - done)
        td_target = rewards + self.gamma * next_state_vals * (1 - dones)

        # advantage: A(s,a) = TD target - V(s)
        advantage = td_target - state_vals

        # Actor loss: -log π(a|s) * A(s,a)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        actor_loss = -(log_probs * advantage.detach()).mean()

        # Critic loss: MSE(V(s), TD target)
        critic_loss = F.mse_loss(state_vals, td_target.detach())

        # entropy bonus: encourage exploration (punish overconfident policies)
        entropy = action_dist.entropy().mean()

        # combined loss
        total_loss = (
            actor_loss
            + self.value_loss_coefficient * critic_loss
            - self.entropy_coefficient * entropy
        )

        # optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(parameters=self.network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        self.steps_done += 1

        return {
            'actor_loss':     actor_loss.item(),
            'critic_loss':    critic_loss.item(),
            'entropy':        entropy.item(),
            'total_loss':     total_loss.item(),
            'mean_advantage': advantage.mean().item(),
        }

    def save(self, path: str) -> None:
        """
        Save agent snapshot at checkpoint
        - network state
        - optimizer state
        - step counter
        - config 
        """
        agent_dict = {
            'network_state_dict':   self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done':           self.steps_done,
            'config':               self.config,
        }
        torch.save(agent_dict, path)
        print(f"[A2C] Checkpoint saved to {path}")

    def load(self, filepath: str) -> None:
        """
        Load network + optimizer state from checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        print(f"[A2C] Checkpoint loaded from {filepath} (step: {self.steps_done})")


class PPOAgent(BaseAgent):
    """
    PPO (Proximal Policy Optimization) Agent

    - collects one full episode, then trains on it for a few passes
    - limits magnitude of individual policy updates, keeps training stable
    - stores old log_probs to compare current policy to prev
    - on-policy, discards episode after updating

    Obs format: uint8 (H, W, C) → normalized + transposed internally
    choose_action() returns tuple (action, log_prob)
    """

    def __init__(self, config: Dict, obs_shape: np.ndarray, num_actions: int, device: torch.device):
        """
        :param obs_shape:   (H, W, C) — e.g. (84, 84, 1)
        """
        super().__init__(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)

        self.num_actions = num_actions
        self.config = config
        self.steps_done = 0

        # hyperparams
        self.gamma           = config.get('gamma', 0.99)
        self.learning_rate   = config.get('learning_rate', 3e-4)
        self.clip_eps        = config.get('clip_eps', 0.2)
        self.gae_lambda      = config.get('gae_lambda', 0.95)
        self.update_epochs   = config.get('update_epochs', 4)   # reuse trajectory N times
        self.value_loss_coefficient = config.get('value_loss_coefficient', 0.5)
        self.entropy_coefficient = config.get('entropy_coefficient', 0.01)
        self.max_grad_norm   = config.get('max_grad_norm', 0.5)

        # exploitation only (A2C stochastic not epsilon-greedy)
        self.epsilon    = 0.0   # attribute kept to adhere to interface
        
        # shared Actor-Critic network
        self.network   = ActorCriticNetwork(observation_shape=obs_shape, num_actions=num_actions).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def choose_action(self, obs: np.ndarray, epsilon: float | None = None) -> Tuple[int, float]:
        """
        Sample action from current policy π(a|s) + return its log_prob

        - log_prob stored in the trajectory, used later as old_log_prob in PPO clipped-ratio calculation
        - when epsilon=0.0 (eval mode): greedy argmax, log_prob=0.0 (unused)

        :param obs: uint8 ndarray (H, W, C)
        :return: (action index, log-prob of the action)
        """

        with torch.no_grad():
            # ingest observation
            obs_tensor = torch.as_tensor(obs, device=self.device).float()   # uint8 -> float32
            obs_tensor = obs_tensor.div(255.0)                              # normalize to [0,1]
            obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)           # (H,W,C) -> (1,C,H,W)

            # get action probs from network
            action_logits, _ = self.network(obs_tensor)
            action_probs = F.softmax(action_logits, dim=-1)

            if epsilon == 0.0:
                # greedy eval - log_prob meaningless here, return 0.0
                action = int(action_probs.argmax(dim=-1).item())
                return action, 0.0 # todo: output n/a for probability?

            # sample action, get its log_prob for PPO update
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return int(action.item()), log_prob.item()

    def update(self, trajectories: List[Tuple[np.ndarray, int, float, np.ndarray, bool, float]]) -> Dict:
        """
        PPO update: run <update_epochs> grad steps on the collected episode
        Trajectory format: (obs, action, reward, next_obs, done, log_prob) - note extra log_prob field
        
        :param trajectories: list of trajectory data objects (obs, action, reward, next_obs, done, log_prob)
                          obs, next_obs are uint8 (H, W, C) ndarrays
        :return: dict with last epoch's loss components: policy_loss, value_loss, entropy, total_loss, mean_advantage
        """
        if len(trajectories) == 0:
            return {}

        states, actions, rewards, next_states, dones, old_log_probs = zip(*trajectories)

        # uint8 (N,H,W,C) -> float32 (N,C,H,W), normalized
        states      = torch.as_tensor(np.array(states),      device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        next_states = torch.as_tensor(np.array(next_states), device=self.device).float().div(255.0).permute(0, 3, 1, 2)
        
        actions       = torch.tensor(actions,       dtype=torch.long,    device=self.device)
        rewards       = torch.tensor(rewards,       dtype=torch.float32, device=self.device)
        dones         = torch.tensor(dones,         dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)

        # calculate TD targets once (fixed across all update epochs)
        with torch.no_grad():
            _, next_vals = self.network(next_states)
            next_vals = next_vals.squeeze()
        td_target = rewards + self.gamma * next_vals * (1 - dones)

        last_epoch_losses = {}
        for _ in range(self.update_epochs):
            # fwd pass thru shared network
            action_logits, state_vals = self.network(states)
            state_vals   = state_vals.squeeze()
            action_probs = F.softmax(action_logits, dim=-1)
            dist = torch.distributions.Categorical(action_probs)

            # advantage: action vs critic's baseline 
            advantage = (td_target - state_vals).detach()

            # surrogate objective (ratio clipped to [1-eps, 1+eps])
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, (1 - self.clip_eps), (1 + self.clip_eps)) * advantage # clipping
            
            # losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_vals, td_target.detach())

            # combined loss with entropy bonus (for exploration)
            entropy = dist.entropy().mean()
            total_loss = (
                policy_loss
                + self.value_loss_coefficient * value_loss
                - self.entropy_coefficient * entropy
            )

            # apply grads
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            last_epoch_losses = {
                'policy_loss':    policy_loss.item(),
                'value_loss':     value_loss.item(),
                'entropy':        entropy.item(),
                'total_loss':     total_loss.item(),
                'mean_advantage': advantage.mean().item(),
            }

        self.steps_done += 1
        return last_epoch_losses

    def save(self, path: str) -> None:
        """
        Save agent snapshot at checkpoint
        - network state
        - optimizer state
        - step counter
        - config
        """
        agent_dict = {
            'network_state_dict':   self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done':           self.steps_done,
            'config':               self.config,
        }
        torch.save(agent_dict, path)
        print(f"[PPO] Checkpoint saved to {path}")

    def load(self, filepath: str) -> None:
        """Load network + optimizer state from checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        print(f"[PPO] Checkpoint loaded from {filepath} (step: {self.steps_done})")