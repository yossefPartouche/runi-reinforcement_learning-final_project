import os
import time
import torch
from typing import Dict

from src.agent import DQNAgent, A2CAgent, PPOAgent
from src.constants import EPISODE_WINDOW_SIZE
from src.template import SimpleGridEnv, KeyDoorBallEnv, pre_process
from src.utils import ExperimentLogger, MetricsHandler, VideoRecorder, timer


# --- helpers ---
def _get_milestones(env: SimpleGridEnv | KeyDoorBallEnv) -> Dict:
    """
    Reads per-episode milestone flags from KeyDoorBallEnv
    :return: milestone dict for envs that track milestones (e.g. KeyDoorBall), otherwise empty dict (e.g. SimpleGrid)
    """
    if not hasattr(env, "prev_key"):
        return {}
    
    milestones = {attr: int(getattr(env, attr)) for attr in ["has_crossed_door",]
        if hasattr(env, attr)
    }
    milestones |= {
        "got_key":    int(getattr(env, "prev_key",  False)),
        "opened_door": int(getattr(env, "prev_door", False)),
        "got_ball":   int(getattr(env, "prev_ball", False)),
    }
    return milestones


class Experiment:
    def __init__(self, config: Dict, device: torch.device, exp_name: str = ""):
        self.config = config
        self.exp_name = exp_name
        self.device = device
        self.training_episodes = config.get("training_episodes", 1000)
        self.inference_episodes = config.get("inference_episodes", 100)

        # init env
        step_limit = config.get("max_steps", 200)
        env_class = self._get_env_class()
        self.env = env_class(preprocess=pre_process, max_steps=step_limit)

        # inject reward config to env
        self.env.reward_shaping = config.get("reward_shaping")
        
        # init agent
        agent_class = self._determine_agent_class()
        self.agent = agent_class(
            config=config, 
            obs_shape=config["obs_shape"], 
            num_actions=self.env.action_space.n, 
            device=device
        )

        # experiment results folder
        exp_timestamp = time.strftime('%Y%m%d-%H%M%S')
        exp_name = self.exp_name if self.exp_name else config['algo']
        folder_name = f"{exp_name}_{exp_timestamp}"
        self.results_dir = os.path.join("results", folder_name)

        # logger + video recorder
        self.logger = ExperimentLogger(save_dir=self.results_dir)
        self.video_recorder = VideoRecorder(save_dir=self.results_dir, env=self.env)
        
    @timer
    def train(self) -> Dict:
        """
        Runs training loop
        Also handles:
        - video recording
        - logging results
        Returns training metrics object
        """
        print(f"Starting training: {self.agent.name} agent on environment {self.config['env_name']}")

        update_per_step = self.config.get("use_per_step_update", True)
        metrics_handler = MetricsHandler(num_episodes=self.training_episodes, window_size=EPISODE_WINDOW_SIZE)   

        for episode in range(1, self.training_episodes + 1):
            # episode resets:
            obs, _ = self.env.reset()
            done = False
            episode_rewards = 0
            episode_steps = 0
            trajectories = []  # for episode-level updates (A2C)

            # todo: record at trianing end as well (last episode)?
            record_episode_video = (episode == self.training_episodes // 2)
            if record_episode_video:
                self.video_recorder.start(stage="mid-training")
            
            while not done:
                # capture video frame
                if record_episode_video: self.video_recorder.capture()

                # select action
                action = self.agent.choose_action(obs)
                log_prob = 0.0  # dummy value for A2C, DQN
                if isinstance(action, tuple):   # handle PPO choose_action output
                    action, log_prob = action

                # env step
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # todo: base this condition on agent type?
                # agent step
                if update_per_step:
                    # DQN: store transition, update, epsilon decay every step
                    self.agent.step(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
                else:
                    # A2C/PPO: store episode trajectory, update at episode end 
                    # note: log_prob used by PPO, ignored by A2C
                    trajectories.append((obs, action, reward, next_obs, float(done), log_prob))

                # updates:
                episode_rewards += reward
                episode_steps += 1
                obs = next_obs
            
            if not update_per_step:
                self.agent.update(trajectories)

            # save recorded video
            if record_episode_video:
                self.video_recorder.stop()
                print(f"Mid-training video saved to {self.video_recorder.filename}")
            
            # collect milestone state (KeyDoorBall only, no-op for SimpleGrid)
            milestones = _get_milestones(self.env)

            # log + print episode metrics
            is_success = terminated
            metrics_handler.update(reward=episode_rewards, steps=episode_steps, success=is_success)
            metrics_handler.print_training_status(episode=episode, epsilon=self.agent.epsilon)
            self.logger.log(filename="training_log", 
                            episode=episode, reward=episode_rewards, steps=episode_steps, epsilon=self.agent.epsilon, success=is_success)
        
            if milestones:
                self.logger.log(filename="milestone_log", episode=episode, **milestones)
        
        # save final model
        self.agent.save(path=os.path.join(self.results_dir, "final_model.pt"))

        # training metrics for whole experiment
        return metrics_handler.get_training_metrics(epsilon=self.agent.epsilon)

    @timer
    def evaluate(self) -> Dict[str, int | float]:
        """
        Runs  inference stage with greedy action (epsilon=0)
        Also handles:
        - video recording
        - logging results
        Returns inference metrics object
        """
        print(f"\nStarting Inference ({self.inference_episodes} episodes)...")

        metrics_handler = MetricsHandler(num_episodes=self.inference_episodes)   

        for episode in range(1, self.inference_episodes + 1):
            # episode resets
            obs, _ = self.env.reset()
            done = False
            episode_rewards = 0
            episode_steps = 0

            if episode == 1: self.video_recorder.start(stage="inference")
            
            while not done:
                # record video of first inference episode (post training)
                if episode == 1: self.video_recorder.capture()
                
                # take greedy action (no exploration)
                action = self.agent.choose_action(obs=obs, epsilon=0.0)
                if isinstance(action, tuple):   # handle PPO choose_action output
                    action = action[0]

                # env step
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_rewards += reward
                episode_steps += 1
            
            # record video of first inference episode
            if episode == 1:
                self.video_recorder.stop()
                print(f"Post-training video saved during inference to {self.video_recorder.filename}")

            # log episode metrics
            is_success = terminated
            metrics_handler.update(reward=episode_rewards, steps=episode_steps, success=is_success)
            self.logger.log(filename="inference_log", 
                            episode=episode, reward=episode_rewards, steps=episode_steps, success=is_success)

        return metrics_handler.get_inference_metrics()

    def _get_env_class(self):
        env_name = self.config.get("env_name")
        if env_name == "SimpleGrid":
            return SimpleGridEnv
        elif env_name == "KeyDoorBall":
            return KeyDoorBallEnv
        else:
            raise ValueError(f"Unknown Environment: {env_name}")

    def _determine_agent_class(self):
        algo_name = self.config.get("algo")
        if algo_name == "DQN":
            return DQNAgent
        elif algo_name == "A2C":
            return A2CAgent
        elif algo_name == "PPO":
            return PPOAgent
        else:
            raise ValueError(f"Unknown Agent Algo: {algo_name}")