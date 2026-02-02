import imageio
import numpy as np
import os

def record_agent_video(
    env,
    agent,
    num_episodes: int = 3,
    max_steps: int = 200,
    save_path: str = 'results/videos',
    filename: str = None,
    fps: int = 10
):
    """
    Record video of trained agent acting in environment.
    
    Args:
        env: Gymnasium environment
        agent: Trained agent with choose_action method
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        save_path: Directory to save video
        filename: Custom filename (auto-generated if None)
        fps: Frames per second for video
    
    Returns:
        str: Path to saved video file
    """
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        env_name = env.__class__.__name__
        filename = f'{env_name}_trained_agent.mp4'
    
    video_path = os.path.join(save_path, filename)
    
    print(f"\n🎬 Recording agent video...")
    print(f"Episodes: {num_episodes}, Max steps per episode: {max_steps}")
    
    # Start video writer
    with imageio.get_writer(video_path, fps=fps) as video:
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            # Add episode title frame
            frame = env.render()
            video.append_data(frame)
            
            while not done and steps < max_steps:
                # Agent selects action (greedy, no exploration)
                action = agent.choose_action(obs, epsilon=0.0)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                
                # Record frame
                frame = env.render()
                video.append_data(frame)
            
            print(f"  Episode {episode + 1}/{num_episodes}: "
                  f"Steps = {steps}, Reward = {total_reward:.2f}, "
                  f"Done = {done}")
    
    print(f"Video saved to: {video_path}")
    return video_path