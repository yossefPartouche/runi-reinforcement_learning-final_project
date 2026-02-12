import time
import numpy as np

def train_a2c(env, agent, logger, config):
    """
    Training loop for A2C (Advantage Actor-Critic).
    
    On-policy learning: updates after each episode using collected trajectories.
    """
    num_episodes = config['num_episodes']
    log_interval = config.get('log_interval', 10)
    save_interval = config.get('save_interval', 500)
    n_steps = config.get('n_steps', None)
    
    print("=" * 70)
    print(f"Starting A2C training: {config['env_name']}")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Update: {'After each episode' if n_steps is None else f'Every {n_steps} steps'}")
    print(f"Gamma: {agent.gamma}")
    print(f"Learning rate: {agent.learning_rate}")
    print("=" * 70)
    
    episode_rewards = []
    episode_lengths = []
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        trajectories = []
        
        while not done:
            # Select action (stochastic policy)
            action = agent.choose_action(obs)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            trajectories.append((obs, action, reward, next_obs, float(done)))
            
            episode_reward += reward
            episode_steps += 1
            obs = next_obs
            
            # Optional: n-step update
            if n_steps and len(trajectories) >= n_steps:
                losses = agent.update(trajectories)
                trajectories = []
        
        # Episode finished - update with remaining trajectories
        if len(trajectories) > 0:
            losses = agent.update(trajectories)
        
        milestones = {
            'got_key': env.got_key_this_episode,
            'opened_door': env.opened_door_this_episode,
            'crossed_door': env.has_crossed_door,
            'got_ball': env.got_ball_this_episode,
            'reached_goal': env.reached_goal_this_episode
        }
        
        # Logging
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        
        logger.log(
            episode=episode,
            reward=episode_reward,
            steps=episode_steps,
            epsilon=0.0,  # A2C doesn't use epsilon
            milestones=milestones,
            **losses
        )
        
        # Print progress
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:])
            avg_steps = np.mean(episode_lengths[-log_interval:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode}/{num_episodes} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Avg Reward (last {log_interval}): {avg_reward:.2f} | "
                  f"Steps: {episode_steps} | "
                  f"Avg Steps: {avg_steps:.1f} | "
                  f"Time: {elapsed:.1f}s")
            
            if 'actor_loss' in losses:
                print(f"  Actor: {losses['actor_loss']:.4f} | "
                      f"Critic: {losses['critic_loss']:.4f} | "
                      f"Entropy: {losses['entropy']:.4f}")
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = f"checkpoints/{config['env_name']}_A2C_episode_{episode}.pt"
            agent.save(checkpoint_path)
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best reward: {max(episode_rewards):.2f}")
    print(f"Final 50 avg reward: {np.mean(episode_rewards[-50:]):.2f}")
    print("=" * 70)