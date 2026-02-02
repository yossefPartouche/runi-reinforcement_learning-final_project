import yaml
import torch
from src.utils import set_random_seed, Logger, get_device
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import train
from src.environments.simple_grid_env import SimpleGridEnv
from src.environments.key_door_ball_env import KeyDoorBallEnv
from src.preprocessing.image_preprocessing import preprocess_observation
from src.utils.visualization import plot_training_progress  # ← ADD THIS
from src.evaluation.video_recoder import record_agent_video 

def main():

    config_path = 'configs/simple_grid_config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        return
    
    print(f"Loaded config from {config_path}")
    print(f"Training {config['algo']} on {config['env_name']}")

    # setup infra:
    set_random_seed(seed=config["seed"])

    device = get_device()
    print(f"Using device: {device}")

    # 4. Initialize environment
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
    
    print(f"Environment created: {config['env_name']}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    observation_shape = env.observation_space.shape  # e.g., (1, 320, 320)
    num_actions = env.action_space.n 

    agent = DQNAgent(
        observation_shape=observation_shape,
        num_actions=num_actions,
        config=config,
        device=device
    )
    
    print(f"DQN Agent initialized")
    print(f"  - Q-network parameters: {sum(p.numel() for p in agent.q_network.parameters()):,}")
    print(f"  - Buffer capacity: {agent.buffer_capacity:,}")
    print(f"  - Batch size: {agent.batch_size}")
    print(f"  - Learning rate: {agent.learning_rate}")
    print(f"  - Gamma: {agent.gamma}")

    logger = Logger(config=config)
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    # run training
    train(env=env, agent=agent, logger=logger, config=config)

    final_checkpoint = f"checkpoints/{config['env_name']}_{config['algo']}_final.pt"
    agent.save(final_checkpoint)

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)

    # ======================
    # POST-TRAINING ANALYSIS
    # ======================
    print("\n" + "="*50)
    print("Generating training plots...")
    print("="*50)

    log_file = logger.log_file
    plot_training_progress(log_file, save_dir='results/plots')

    # 2. Record video of trained agent
    print("\n" + "="*50)
    print("Recording agent video...")
    print("="*50)
    video_path = record_agent_video(
        env=env,
        agent=agent,
        num_episodes=3,
        max_steps=config['max_steps'],
        save_path='results/videos',
        filename=f"{config['env_name']}_{config['algo']}_trained.mp4",
        fps=10
    )

    print("\n" + "="*50)
    print("Analysis Complete")
    print("="*50)
    print(f"Training plots: results/plots/")
    print(f"Agent video: {video_path}")
    print(f"Final checkpoint: {final_checkpoint}")
    print(f"Training log: {log_file}")
    print("="*50)


if __name__ == "__main__":
    main()