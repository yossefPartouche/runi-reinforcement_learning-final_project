"""
Standalone script to analyze training results and generate visualizations.
Run this after training is complete.
"""

import yaml
import torch
from src.utils.visualization import plot_training_progress
from src.evaluation.video_recoder import record_agent_video
from src.agents.dqn_agent import DQNAgent
from src.environments.simple_grid_env import SimpleGridEnv
from src.environments.key_door_ball_env import KeyDoorBallEnv
from src.preprocessing.image_preprocessing import preprocess_observation
from src.utils import get_device

def main():
    # ========================================
    # CONFIGURATION
    # ========================================
    config_path = 'configs/simple_grid_config.yaml'
    log_file = 'logs/SimpleGridEnv_DQN.csv'
    checkpoint_path = 'checkpoints/SimpleGridEnv_DQN_final.pt'
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("="*60)
    print("Analyzing Training Results")
    print("="*60)
    print(f"Environment: {config['env_name']}")
    print(f"Algorithm: {config['algo']}")
    print(f"Log file: {log_file}")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*60)
    
    # ========================================
    # 1. GENERATE TRAINING PLOTS
    # ========================================
    print("\n📊 Generating training progress plots...")
    plot_training_progress(log_file, save_dir='results/plots')
    
    # ========================================
    # 2. LOAD TRAINED AGENT
    # ========================================
    print("\n🤖 Loading trained agent...")
    device = get_device()
    
    # Create environment
    if config['env_name'] == 'SimpleGridEnv':
        env = SimpleGridEnv(
            size=config['env_size'],
            max_steps=config['max_steps'],
            render_mode='rgb_array',
            preprocess=preprocess_observation
        )
    elif config['env_name'] == 'KeyDoorBallEnv':
        env = KeyDoorBallEnv(
            size=config['env_size'],
            max_steps=config['max_steps'],
            render_mode='rgb_array',
            preprocess=preprocess_observation
        )
    else:
        raise ValueError(f"Unknown environment: {config['env_name']}")
    
    # Create agent
    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    agent = DQNAgent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        config=config,
        device=device
    )
    
    # Load trained weights
    agent.load(checkpoint_path)
    print(f"Agent loaded from {checkpoint_path}")
    
    # ========================================
    # 3. RECORD VIDEO
    # ========================================
    print("\n🎬 Recording agent video...")
    video_path = record_agent_video(
        env=env,
        agent=agent,
        num_episodes=5,  # Record 5 episodes
        max_steps=config['max_steps'],
        save_path='results/videos',
        filename=f"{config['env_name']}_{config['algo']}_trained.mp4",
        fps=10
    )
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f" Training plots: results/plots/{config['env_name']}_{config['algo']}_progress.png")
    print(f" Agent video: {video_path}")
    print("="*60)

if __name__ == "__main__":
    main()