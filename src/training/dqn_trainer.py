"""
Generic training loop for different types of agents that'll be implemented
"""
import time
from typing import Dict, Any, Optional 
from src.agents.agent import Agent
from src.utils import Logger

def train(env, agent: Agent, logger: Logger, config: Dict[str, Any]) -> None:
    """
    Train RL Agents on given environment

    Currently only training DQN but should be (untested) compatible with (PPO, AC, A2C, A3C ...)

    Args:
        env: Gymnasium environment
        agent: Agent implementing Agent interface with:
            - choose_action(obs, epsilon=None) -> action
            - step(obs, action, reward, next_obs, done) -> None
            - save(path) -> None
        logger: Logger for tracking metrics
        config: Configuration dictionary with keys:
            Required:
                - num_episodes: Total training episodes
                - env_name: Environment name
                - algo: Algorithm name
            Optional (for epsilon-greedy agents):
                - epsilon_start: Initial exploration rate (default: 1.0)
                - epsilon_end: Minimum exploration rate (default: 0.01)
                - epsilon_decay: Decay factor per episode (default: 0.995)
            Optional (logging/saving):
                - log_interval: Log every N episodes (default: 10)
                - save_interval: Save checkpoint every N episodes (default: 500)
                - use_epsilon: Whether agent uses epsilon-greedy (default: True for DQN)
    Notes:
        - For non-epsilon agents (e.g., policy gradient), set use_epsilon=False
        - Agent's choose_action() should handle epsilon=None gracefully
        - Epsilon tracking will be skipped if use_epsilon=False
    """
    print("="*70)
    print(f"Starting training: {config['algo']} agent on {config['env_name']}")
    print("="*70)
    
    num_episodes = config['num_episodes']
    log_interval = config.get('log_interval', 10)
    save_interval = config.get('save_interval', 500)

    use_epsilon = config.get('use_epsilon', True)
    epsilon_start = config.get('epsilon_start', 1.0) if use_epsilon else None
    epsilon_end = config.get('epsilon_end', 0.01) if use_epsilon else None
    epsilon_decay = config.get('epsilon_decay', 0.995) if use_epsilon else None

    epsilon = epsilon_start if use_epsilon else None

    start_time = time.time()
    best_reward = float('-inf')

    # Print configuration
    print(f"\nTraining Configuration:")
    print(f"  Algorithm: {config['algo']}")
    print(f"  Environment: {config['env_name']}")
    print(f"  Episodes: {num_episodes}")

    if use_epsilon:
        print(f"  Exploration (epsilon): {epsilon_start} -> {epsilon_end} (decay={epsilon_decay})")
    else:
        print(f"  Exploration: Handled by agent's policy")
    
    print(f"  Log interval: every {log_interval} episodes")
    print(f"  Save interval: every {save_interval} episodes")

    # Print agent-specific info if available
    if hasattr(agent, 'buffer_capacity'):
        print(f"  Replay buffer capacity: {agent.buffer_capacity:,}")
    if hasattr(agent, 'batch_size'):
        print(f"  Batch size: {agent.batch_size}")
    if hasattr(agent, 'learning_rate'):
        print(f"  Learning rate: {agent.learning_rate}")
    if hasattr(agent, 'gamma'):
        print(f"  Gamma (discount): {agent.gamma}")
    if hasattr(agent, 'target_update_freq'):
        print(f"  Target update frequency: {agent.target_update_freq}")
    
    print("="*70 + "\n")

    # ====================================================== 
    # MAIN TRAINING LOOP
    # ======================================================


    for episode in range(1, num_episodes + 1):
        # episode resets:
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_start_time = time.time()

        while not done:
            # If agent uses epsilon-greedy, pass epsilon; otherwise agent decides
            if use_epsilon:
                action = agent.choose_action(obs, epsilon=epsilon)
            else:
                action = agent.choose_action(obs)
        
            # env step
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # agent step
            agent.step(
                obs=obs, 
                action=action, 
                reward=reward, 
                next_obs=next_obs, 
                done=done
            )
            
            # updates:
            total_reward += reward
            steps += 1
            obs = next_obs
        
        if use_epsilon:
            epsilon = max(epsilon_end, epsilon*epsilon_decay)
        
        episode_duration = time.time() - episode_start_time 

        log_data = {
            'episode': episode,
            'reward': total_reward,
            'steps': steps,
            'duration': episode_duration
        }
        if use_epsilon:
            log_data['epsilon'] = epsilon

        logger.log(**log_data)
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Progress 
        if episode % log_interval == 0:
            elapsed_time = time.time() - start_time
            avg_reward = logger.get_average_reward(last_n=log_interval)

            progress_parts = [
                f"Episode {episode}/{num_episodes}",
                f"Reward: {total_reward:.2f}",
                f"Avg Reward (last {log_interval}): {avg_reward:.2f}",
                f"Steps: {steps}",
            ]

            if use_epsilon:
                progress_parts.append(f"Epsilon: {epsilon:.4f}")

            if hasattr(agent, 'replay_buffer'):
                buffer_size = len(agent.replay_buffer)
                buffer_capacity = agent.buffer_capacity
                progress_parts.append(f"Buffer: {buffer_size:,}/{buffer_capacity:,}")
            
            progress_parts.append(f"Time: {elapsed_time:.1f}s")
            
            print(" | ".join(progress_parts))
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = f"checkpoints/{config['env_name']}_{config['algo']}_ep{episode}.pt"
            agent.save(checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # =============================
    # TRAINING COMPLETE
    # =============================
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print("Training Complete")
    print("="*70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f}")

    if use_epsilon:
        print(f"Final epsilon {epsilon: .4f}")
    
    if hasattr(logger, 'total_steps'):
        print(f"Total steps: {logger.total_steps:,}")
    print("="*70 + "\n")

    # Save final Model
    final_checkpoint = f"checkpoints/{config['env_name']}_{config['algo']}_final.pt"
    agent.save(final_checkpoint)
    print(f"💾 Final model saved: {final_checkpoint}\n")
