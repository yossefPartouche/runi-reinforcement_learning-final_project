import time
import numpy as np

def train_ppo(env, agent, logger, config):
    """
    Training loop for PPO
    on-Policy learning: updates after each episode 
    using collected trajectories
    """
    print("=" * 70)
    print(f"Starting PPO training: {config['env_name']}")
    print("=" * 70)
    num_episodes = config['num_episodes']
    log_interval = config.get('interval', 25)
    save_interval = config.get('save_interval', 500)
    n_steps = config.get('n_steps', None)

    print(f"Episodes: {num_episodes}")
    print(f"Update: {'After each episode' if n_steps is None else f'Every {n_steps} steps'}")
    print(f"Gamma: {agent.gamma}")
    print(f"Learning rate: {agent.learning_rate}")
    print("=" * 70)

    episode_rewards = []
    episode_lengths = []
    start_time = time.time()

    for episode in range(1, num_episodes +1):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        trajectories = []

        while not done:
            action = agent.choose_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            # The last one is a placeholder for the log_prob
            trajectories.append((obs, action, reward, next_obs, float(done), 0.0)) 

            episode_reward += reward
            episode_steps += 1
            obs = next_obs

            if len(trajectories) > 0:
                losses = agent.update(trajectories)
            
            milestones = {
                'got_key': getattr(env, 'got_key_this_episode', False),
                'opened_door': getattr(env, 'opened_door_this_episode', False),
                'crossed_door': getattr(env, 'has_crossed_door', False),
                'got_ball': getattr(env, 'got_ball_this_episode', False),
                'reached_goal': getattr(env, 'reached_goal_this_episode', False)
            }

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)

            logger.log(
                episode=episode,
                reward=episode_reward,
                steps=episode_steps,
                epsilon=0.0,  # PPO doesn't use epsilon
                milestones=milestones,
                **(losses if losses else {})
            )
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
                
                if losses and 'policy_loss' in losses:
                    print(f"  Policy: {losses['policy_loss']:.4f} | "
                        f"Value: {losses['value_loss']:.4f}")
            
            if episode % save_interval == 0:
                checkpoint_path = f"checkpoints/{config['env_name']}_PPO_episode_{episode}.pt"
                agent.save(checkpoint_path)
        
        total_time = time.time() - start_time


        print("\n" + "=" * 70)
        print("Training Complete")
        print("=" * 70)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best reward: {max(episode_rewards):.2f}")
        print(f"Final 50 avg reward: {np.mean(episode_rewards[-50:]):.2f}")
        print("=" * 70)
            
            

            



        