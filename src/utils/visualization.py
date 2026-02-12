import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

def plot_milestone_progress(milestone_csv: str, save_dir: str = 'results/plots'):
    """
    Plot cumulative milestone achievements over training.
    Simple line graph showing 5 milestone counts vs training steps.
    
    Args:
        milestone_csv: Path to milestone CSV file
        save_dir: Directory to save plot
    """
    if not os.path.exists(milestone_csv):
        print(f"⚠️  Milestone file not found: {milestone_csv}")
        return
    
    # Read data
    df = pd.read_csv(milestone_csv)
    
    # Create simple figure
    plt.figure(figsize=(12, 6))
    
    # Plot 5 lines - simple, clean
    plt.plot(df['total_steps'], df['cumulative_key_pickups'], 
             label='Key Pickups', linewidth=2)
    
    plt.plot(df['total_steps'], df['cumulative_door_opens'], 
             label='Door Opens', linewidth=2)
    
    plt.plot(df['total_steps'], df['cumulative_door_crosses'], 
             label='Door Crosses', linewidth=2)
    
    plt.plot(df['total_steps'], df['cumulative_ball_pickups'], 
             label='Ball Pickups', linewidth=2)
    
    plt.plot(df['total_steps'], df['cumulative_goal_reaches'], 
             label='Goal Reaches', linewidth=2.5)
    
    # Labels
    plt.xlabel('Training Steps')
    plt.ylabel('Count')
    plt.title('Cumulative Milestone Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / 'milestone_progress.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Milestone plot saved to {save_path}")
    plt.close()