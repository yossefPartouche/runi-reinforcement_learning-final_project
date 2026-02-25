import yaml
import torch
from pathlib import Path
from src.utils import set_random_seed, Logger, get_device
from src.agents.dqn_agent import DQNAgent
from src.agents.a2c_agent import A2CAgent
from src.agents.ppo_agent import PPOAgent
from src.training.dqn_trainer import train
from src.training.a2c_trainer import train_a2c
from src.training.ppo_trainer import train_ppo
from src.environments.simple_grid_env import SimpleGridEnv
from src.environments.key_door_ball_env import KeyDoorBallEnv
from src.preprocessing.image_preprocessing import preprocess_observation
from src.evaluation.video_recoder import record_agent_video
from src.utils.visualization import plot_milestone_progress 

def create_environment(config):
    """
    Create environment based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        env: Initialized environment
    """
    env_name = config['env_name']
    
    if env_name == 'SimpleGridEnv':
        env = SimpleGridEnv(
            size=config['env_size'],
            max_steps=config['max_steps'],
            render_mode='rgb_array',
            preprocess=preprocess_observation
        )
    elif env_name == 'KeyDoorBallEnv':
        env = KeyDoorBallEnv(
            size=config['env_size'],
            max_steps=config['max_steps'],
            render_mode='rgb_array',
            preprocess=preprocess_observation
        )
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    return env

def create_agent(observation_shape, num_actions, config, device):
    """
    Create agent based on algorithm specified in config.
    
    Args:
        observation_shape: Shape of observations (C, H, W)
        num_actions: Number of discrete actions
        config: Configuration dictionary
        device: torch device
        
    Returns:
        agent: Initialized agent
    """
    algo = config['algo']
    
    if algo == 'DQN':
        agent = DQNAgent(
            observation_shape=observation_shape,
            num_actions=num_actions,
            config=config,
            device=device
        )
    elif algo == 'A2C':
        agent = A2CAgent(
            observation_shape=observation_shape,
            num_actions=num_actions,
            config=config,
            device=device
        )
    elif algo == 'PPO':
        agent = PPOAgent(
            observation_shape=observation_shape,
            num_actions=num_actions,
            config=config,
            device=device
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    return agent

def get_trainer(algo):
    """
    Get appropriate training function for the algorithm.
    
    Args:
        algo: Algorithm name (str)
        
    Returns:
        trainer_fn: Training function
    """
    trainer_map = {
        'DQN': train,
        'A2C': train_a2c,
        'PPO': train_ppo,
    }
    
    if algo not in trainer_map:
        raise ValueError(f"No trainer found for algorithm: {algo}")
    
    return trainer_map[algo]

def print_training_header(config, device, env, agent):
    """
    Print formatted training information.
    
    Args:
        config: Configuration dictionary
        device: torch device
        env: Environment
        agent: Agent
    """
    print("\n" + "=" * 80)
    print(f"{'REINFORCEMENT LEARNING TRAINING':^80}")
    print("=" * 80)
    print(f"Algorithm:        {config['algo']}")
    print(f"Environment:      {config['env_name']}")
    print(f"Device:           {device}")
    print(f"Random Seed:      {config['seed']}")
    print("-" * 80)
    print(f"Environment Info:")
    print(f"  - Grid Size:    {config['env_size']}×{config['env_size']}")
    print(f"  - Max Steps:    {config['max_steps']}")
    print(f"  - Obs Space:    {env.observation_space}")
    print(f"  - Action Space: {env.action_space} ({env.action_space.n} actions)")
    print("-" * 80)
    print(f"Training Config:")
    print(f"  - Episodes:     {config['num_episodes']}")
    print(f"  - Gamma:        {config['gamma']}")
    print(f"  - Learning Rate: {config['learning_rate']}")
    
    # Algorithm-specific parameters
    if config['algo'] == 'DQN':
        print(f"  - Batch Size:   {config.get('batch_size', 'N/A')}")
        print(f"  - Buffer Size:  {config.get('buffer_capacity', 'N/A')}")
        print(f"  - Epsilon:      {config.get('epsilon_start', 1.0)} → {config.get('epsilon_end', 0.01)}")
        print(f"  - Target Update: Every {config.get('target_update_freq', 'N/A')} steps")
    elif config['algo'] == 'A2C':
        print(f"  - Value Loss Coef: {config.get('value_loss_coef', 'N/A')}")
        print(f"  - Entropy Coef:    {config.get('entropy_coef', 'N/A')}")
        print(f"  - Max Grad Norm:   {config.get('max_grad_norm', 'N/A')}")
        n_steps = config.get('n_steps', None)
        print(f"  - Update Freq:     {'Per episode' if n_steps is None else f'Every {n_steps} steps'}")
    
    print("-" * 80)
    print(f"Agent Info:")
    if hasattr(agent, 'q_network'):  # DQN
        q_params = sum(p.numel() for p in agent.q_network.parameters())
        target_params = sum(p.numel() for p in agent.target_network.parameters())
        print(f"  - Q-Network Params:      {q_params:,}")
        print(f"  - Target Network Params: {target_params:,}")
    elif hasattr(agent, 'network'): #A2C
        network_params = sum(p.numel() for p in agent.network.parameters())
        print(f" - Network Params: {network_params}")
    
    print("=" * 80 + "\n")

def print_training_complete(config, log_file, final_checkpoint):
    """
    Print training completion summary.
    
    Args:
        config: Configuration dictionary
        log_file: Path to training log
        final_checkpoint: Path to final checkpoint
    """
    print("\n" + "=" * 80)
    print(f"{'TRAINING COMPLETE':^80}")
    print("=" * 80)
    print(f"Algorithm:         {config['algo']}")
    print(f"Environment:       {config['env_name']}")
    print(f"Total Episodes:    {config['num_episodes']}")
    print("-" * 80)
    print("Generated Files:")
    print(f"  Training Log:    {log_file}")
    print(f"  Checkpoint:      {final_checkpoint}")
    print(f"  Plots:           results/plots/")
    print(f"  Video:           results/videos/")
    print("=" * 80 + "\n")

def main():
    # -------------------------------------------------------------------------
    # 1. LOAD CONFIGURATION
    # -------------------------------------------------------------------------
    config_path = 'configs/key_door_ball_config.yaml'
    #config_path = 'configs/simple_grid_config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f" Error: Config file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f" Error parsing YAML config: {e}")
        return
    
    # -------------------------------------------------------------------------
    # 2. SETUP
    # -------------------------------------------------------------------------
    set_random_seed(seed=config["seed"])
    device = get_device()
    
    # -------------------------------------------------------------------------
    # 3. CREATE ENVIRONMENT
    # -------------------------------------------------------------------------
    try:
        env = create_environment(config)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    # -------------------------------------------------------------------------
    # 4. CREATE AGENT
    # -------------------------------------------------------------------------
    try:
        agent = create_agent(observation_shape, num_actions, config, device)
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # -------------------------------------------------------------------------
    # 5. PRINT TRAINING INFO
    # -------------------------------------------------------------------------
    print_training_header(config, device, env, agent)
    
    # -------------------------------------------------------------------------
    # 6. INITIALIZE LOGGER
    # -------------------------------------------------------------------------
    logger = Logger(config=config)
    
    # -------------------------------------------------------------------------
    # 7. RUN TRAINING
    # -------------------------------------------------------------------------
    try:
        trainer_fn = get_trainer(config['algo'])
        trainer_fn(env=env, agent=agent, logger=logger, config=config)
    except ValueError as e:
        print(f"❌ {e}")
        return
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        return
    
    # -------------------------------------------------------------------------
    # 8. SAVE FINAL CHECKPOINT
    # -------------------------------------------------------------------------
    final_checkpoint = f"checkpoints/{config['env_name']}_{config['algo']}_final.pt"
    agent.save(final_checkpoint)
    
    # -------------------------------------------------------------------------
    # 9. POST-TRAINING ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{'POST-TRAINING ANALYSIS':^80}")
    print("=" * 80)
    
    print("\n📊 Generating milestone visualization...")
    log_file = logger.log_file  # Define log_file here
    milestone_file = log_file.replace('.csv', '_milestones.csv')
    if Path(milestone_file).exists():
        plot_milestone_progress(milestone_file, save_dir='results/plots')
        print("✅ Milestone plot saved to results/plots/")
    else:
        print(f"⚠️  Milestone file not found: {milestone_file}")
    
    # Record agent video
    print("\n🎬 Recording agent video...")
    try:
        video_path = record_agent_video(
            env=env,
            agent=agent,
            num_episodes=5,
            max_steps=config['max_steps'],
            save_path='results/videos',
            filename=f"{config['env_name']}_{config['algo']}_trained.mp4",
            fps=10
        )
        print(f"Video saved to {video_path}")
    except Exception as e:
        print(f"Warning: Could not record video: {e}")
    
    # -------------------------------------------------------------------------
    # 10. PRINT SUMMARY
    # -------------------------------------------------------------------------
    print_training_complete(config, logger.log_file, final_checkpoint)

if __name__ == "__main__":
    main()