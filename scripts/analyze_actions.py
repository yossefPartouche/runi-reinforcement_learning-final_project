import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import yaml

from src.environments.key_door_ball_env import KeyDoorBallEnv
from src.preprocessing.image_preprocessing import preprocess_observation
from src.agents.a2c_agent import A2CAgent
from src.utils import get_device, set_random_seed


def analyze_action_patterns(action_history, window_size=10):
    """
    Analyze action patterns for wasteful behavior.
    
    Returns:
        dict: Analysis results including turn-back patterns
    """
    # Count turn-back patterns (Left->Right or Right->Left)
    turn_backs = 0
    spinning = 0
    
    for i in range(len(action_history) - 1):
        curr = action_history[i]
        next_act = action_history[i + 1]
        
        if (curr == 0 and next_act == 1) or (curr == 1 and next_act == 0):
            turn_backs += 1
    
    # Detect spinning (3+ consecutive turns in same direction)
    for i in range(len(action_history) - 2):
        if action_history[i] in [0, 1]:
            if (action_history[i] == action_history[i+1] == action_history[i+2]):
                spinning += 1
    
    # Calculate movement efficiency
    total_turns = sum(1 for a in action_history if a in [0, 1])
    total_moves = sum(1 for a in action_history if a == 2)
    
    return {
        'turn_backs': turn_backs,
        'spinning_patterns': spinning,
        'total_turns': total_turns,
        'total_moves': total_moves,
        'turn_to_move_ratio': total_turns / max(total_moves, 1)
    }


def run_action_analysis(config_path, num_episodes=100):
    """Run agent for N episodes and analyze action distribution."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    set_random_seed(config.get('seed', 42))
    
    env = KeyDoorBallEnv(
        size=config.get('env_size', 10),
        max_steps=config.get('max_steps', 1000),
        preprocess=preprocess_observation
    )
    
    device = get_device()
    agent = A2CAgent(
        observation_shape=env.observation_space.shape,
        num_actions=env.action_space.n,
        config=config,
        device=device
    )
    
    checkpoint_path = Path('checkpoints') / config.get('env_name', 'KeyDoorBallEnv') / 'best_agent.pt'
    if checkpoint_path.exists():
        agent.load(str(checkpoint_path))
        print(f"✅ Loaded agent from {checkpoint_path}")
    else:
        print("⚠️  No checkpoint found, using random agent")
    
    print(f"\n🧪 Running {num_episodes} episodes to analyze actions...\n")
    
    all_action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    episode_patterns = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        for action, count in env.action_counts.items():
            all_action_counts[action] += count
        
        patterns = analyze_action_patterns(env.action_history)
        episode_patterns.append(patterns)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1}/{num_episodes} completed")
    
    return all_action_counts, episode_patterns, env


def plot_action_distribution(action_counts, patterns, save_path='results/plots'):
    """Create visualization of action usage."""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    action_names = ["Turn Left", "Turn Right", "Move Forward", "Pick Up", "Toggle"]
    
    # Calculate percentages
    total_actions = sum(action_counts.values())
    if total_actions == 0:
        print("⚠️ No actions recorded!")
        return
    
    percentages = [(count / total_actions * 100) for count in action_counts.values()]
    
    # ✅ FIX: Use constrained layout instead of tight_layout
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Action Distribution Pie Chart
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    wedges, texts, autotexts = ax1.pie(
        percentages, 
        labels=action_names, 
        autopct='%1.1f%%',
        colors=colors, 
        startangle=90,
        textprops={'fontsize': 10}
    )
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax1.set_title('Action Distribution', fontsize=14, fontweight='bold', pad=20)
    
    # 2. Action Counts Bar Chart
    ax2 = fig.add_subplot(gs[0, 1])
    bars = ax2.bar(range(len(action_names)), list(action_counts.values()), color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_xticks(range(len(action_names)))
    ax2.set_xticklabels(action_names, rotation=30, ha='right')
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Total Action Counts', fontsize=14, fontweight='bold', pad=20)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Wasteful Behavior Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    avg_turn_backs = np.mean([p['turn_backs'] for p in patterns])
    avg_spinning = np.mean([p['spinning_patterns'] for p in patterns])
    avg_ratio = np.mean([p['turn_to_move_ratio'] for p in patterns])
    
    metrics = ['Turn-Backs', 'Spinning', 'Turn/Move\nRatio']
    values = [avg_turn_backs, avg_spinning, avg_ratio]
    
    bars = ax3.bar(range(len(metrics)), values, color=['#ff6b6b', '#ee5a6f', '#c44569'], edgecolor='black', linewidth=1.2)
    ax3.set_xticks(range(len(metrics)))
    ax3.set_xticklabels(metrics)
    ax3.set_ylabel('Average Value', fontsize=12, fontweight='bold')
    ax3.set_title('Inefficiency Patterns (per Episode)', fontsize=14, fontweight='bold', pad=20)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    # 4. Statistics Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    stats_text = f"""📊 ACTION ANALYSIS SUMMARY

Total Actions: {total_actions:,}

🔄 Turning Actions:
  • Left:  {action_counts[0]:,} ({percentages[0]:.1f}%)
  • Right: {action_counts[1]:,} ({percentages[1]:.1f}%)
  • Total: {action_counts[0] + action_counts[1]:,} ({percentages[0] + percentages[1]:.1f}%)

➡️  Movement Actions:
  • Forward: {action_counts[2]:,} ({percentages[2]:.1f}%)

🎯 Interactive Actions:
  • Pick Up: {action_counts[3]:,} ({percentages[3]:.1f}%)
  • Toggle:  {action_counts[4]:,} ({percentages[4]:.1f}%)

⚠️  Inefficiency Metrics:
  • Turn-backs/ep: {avg_turn_backs:.2f}
  • Spinning/ep: {avg_spinning:.2f}
  • Turn/move ratio: {avg_ratio:.2f}

💡 Status:
  {' HIGH TURNING' if percentages[0] + percentages[1] > 50 else 'Turning OK'}
  {'LOW MOVEMENT' if percentages[2] < 30 else 'Movement OK'}
"""
    
    ax4.text(0.05, 0.95, stats_text, 
             fontsize=10, 
             family='monospace',
             verticalalignment='top',
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Save with high DPI
    plt.savefig(f'{save_path}/action_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to {save_path}/action_analysis.png")
    
    # Also save as PDF for better quality
    plt.savefig(f'{save_path}/action_analysis.pdf', bbox_inches='tight')
    print(f"✅ Saved PDF to {save_path}/action_analysis.pdf")
    
    plt.close()  # Close to free memory
    
    # Print detailed report
    print("\n" + "="*60)
    print("📊 DETAILED ACTION ANALYSIS REPORT")
    print("="*60)
    
    turn_percentage = percentages[0] + percentages[1]
    print(f"\n🎯 Turning Percentage: {turn_percentage:.1f}%")
    if turn_percentage > 50:
        print(f"   ✅ CONFIRMED: Agent relies heavily on turning")
    else:
        print(f"   NOT CONFIRMED: Turning is moderate")
    
    print(f"\n Wasteful Behavior:")
    print(f"   • Turn-backs: {avg_turn_backs:.2f}/episode")
    print(f"   • Spinning: {avg_spinning:.2f}/episode")
    print(f"   • Turn/Move ratio: {avg_ratio:.2f}")
    
    # Comparison with targets
    print(f"\n Progress Toward Targets:")
    print(f"   Turn-backs:  {avg_turn_backs:.2f} → Target: < 5.0  {'✅' if avg_turn_backs < 5 else '❌'}")
    print(f"   Turn/Move:   {avg_ratio:.2f} → Target: < 1.5  {'✅' if avg_ratio < 1.5 else '❌'}")
    
    if avg_turn_backs > 10:
        print(f"\n  CRITICAL: Turn-backs still very high!")
        print(f"   → Increase turn-back penalty further")
    elif avg_turn_backs > 5:
        print(f"\n  MODERATE: Turn-backs above target")
        print(f"   → Current penalties are working, need more training")
    else:
        print(f"\n EXCELLENT: Turn-backs within acceptable range!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze agent action distribution')
    parser.add_argument('--config', type=str, 
                       default='configs/key_door_ball_config.yaml',
                       help='Path to config file')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of episodes to analyze')
    
    args = parser.parse_args()
    
    action_counts, patterns, env = run_action_analysis(args.config, args.episodes)
    plot_action_distribution(action_counts, patterns)
    
    print("\n Analysis complete!")