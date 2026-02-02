import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_progress(log_file: str, save_dir: str = 'results/plots'):
    """
    Plot training metrics from CSV log file.
    
    Args:
        log_file: Path to CSV log file (e.g., 'logs/SimpleGridEnv_DQN.csv')
        save_dir: Directory to save plots
    """
    # Create save directory if it doesn't exist
    try:
        os.makedirs(save_dir, exist_ok=True)
    except FileExistsError:
        pass
    except Exception as e:
        print(f"Warning: Could not create directory {save_dir}: {e}")
        print(f"Attempting to save in current directory instead...")
        save_dir = '.'
    
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")
    
    # Load training log
    df = pd.read_csv(log_file)
    
    # Extract environment and algorithm name from log file
    env_algo_name = os.path.basename(log_file).replace('.csv', '')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Progress: {env_algo_name}', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    axes[0, 0].plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward')
    # Rolling average (window=10)
    if len(df) >= 10:
        rolling_reward = df['reward'].rolling(window=10).mean()
        axes[0, 0].plot(df['episode'], rolling_reward, linewidth=2, label='Avg (10 episodes)')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Reward Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Steps per Episode
    axes[0, 1].plot(df['episode'], df['steps'], alpha=0.3, label='Steps')
    if len(df) >= 10:
        rolling_steps = df['steps'].rolling(window=10).mean()
        axes[0, 1].plot(df['episode'], rolling_steps, linewidth=2, label='Avg (10 episodes)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Steps per Episode (Lower = More Efficient)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Epsilon Decay
    axes[1, 0].plot(df['episode'], df['epsilon'], linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].set_title('Exploration Rate (Epsilon) Decay')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Cumulative Reward
    cumulative_reward = df['reward'].cumsum()
    axes[1, 1].plot(df['episode'], cumulative_reward, linewidth=2, color='green')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].set_title('Cumulative Reward Over Training')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{env_algo_name}_progress.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f" Training progress plot saved to: {save_path}")
    
    # Also show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print(f"Training Summary: {env_algo_name}")
    print("="*60)
    print(f"Total Episodes:        {len(df)}")
    print(f"Best Reward:           {df['reward'].max():.2f} (Episode {df.loc[df['reward'].idxmax(), 'episode']:.0f})")
    print(f"Worst Reward:          {df['reward'].min():.2f}")
    print(f"Average Reward:        {df['reward'].mean():.2f}")
    print(f"Final 10 Avg Reward:   {df['reward'].tail(10).mean():.2f}")
    print(f"Minimum Steps:         {df['steps'].min():.0f} (Episode {df.loc[df['steps'].idxmin(), 'episode']:.0f})")
    print(f"Average Steps:         {df['steps'].mean():.1f}")
    print(f"Final 10 Avg Steps:    {df['steps'].tail(10).mean():.1f}")
    print(f"Final Epsilon:         {df['epsilon'].iloc[-1]:.4f}")
    print("="*60)