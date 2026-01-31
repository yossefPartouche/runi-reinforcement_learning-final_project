"""
Logger for tracking training metrics and saving to CSV.
"""
import os
from typing import List, Dict, Any, Optional

class Logger:
    """
    Logger for tracking and saving training metrics.
    
    Supports flexible metric logging (epsilon, loss, etc.) and 
    calculates running statistics.
    """

    def __init__(self, log_dir: str='logs', experiment_name: str = "experiment", config: Optional[Dict[str, Any]]= None):
        """
        Initialize the Logger

        Args:
            log_dir: Dictionary to save logs
            experiment_name: use in filenames
        """
        if config is not None:
            log_dir = config.get('log_dir', 'logs')
            experiment_name = f"{config['env_name']}_{config['algo']}"

        self.log_dir = log_dir
        self.experiment_name = experiment_name

        os.makedirs(log_dir, exist_ok=True)

        self.episode_rewards: List[float] = []
        self.episode_steps: List[int] = []
        self.episode_durations: List[float] = []
        self.epsilons: List[float] = []

        self._extra_metrics : Dict[str, List[Any]] = {}

        # Create CSV file with header
        self.log_file = os.path.join(log_dir, f"{experiment_name}.csv")
        self._init_csv()
    
    def _init_csv(self) -> None:
        """Initialize CSV file with header."""
        with open(self.log_file, 'w') as f:
            f.write("episode,reward,steps,duration,epsilon\n")
    
    def log(self, episode: int, reward: float, steps: int, 
            duration: float = 0.0, epsilon: Optional[float] = None, 
            **kwargs) -> None:
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Total episode reward
            steps: Number of steps in episode
            duration: Episode duration in seconds
            epsilon: Current epsilon value (optional)
            **kwargs: Additional metrics (loss, learning_rate, etc.)
        """
        # Store core metrics
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_durations.append(duration)

        if epsilon is not None:
            self.epsilons.append(epsilon)
        
        for key, value in kwargs.items():
            if key not in self._extra_metrics:
                self._extra_metrics[key] = []
            self._extra_metrics[key].append(value)

        # Write to CSV
        self._write_to_csv(episode, reward, steps, duration, epsilon, **kwargs)
    
    def _write_to_csv(self, episode: int, reward: float, steps: int,
                      duration: float, epsilon: Optional[float], **kwargs) -> None:
        """Write metrics to CSV file."""
        with open(self.log_file, 'a') as f:
            metrics = [
                str(episode),
                f"{reward:.2f}",
                str(steps),
                f"{duration:.2f}",
                f"{epsilon:.4f}" if epsilon is not None else "N/A"
            ]
            
            # Add extra metrics in sorted order
            for key in sorted(kwargs.keys()):
                value = kwargs[key]
                if isinstance(value, float):
                    metrics.append(f"{value:.4f}")
                else:
                    metrics.append(str(value))
            
            f.write(','.join(metrics) + '\n')
    def get_average_reward(self, last_n: int = 100) -> float:
        """
        Calculate average reward over last N episodes.
        
        Args:
            last_n: Number of recent episodes to average
        
        Returns:
            Average reward, or 0.0 if no episodes logged yet
        """
        if not self.episode_rewards:
            return 0.0
        
        recent_rewards = self.episode_rewards[-last_n:]
        return sum(recent_rewards) / len(recent_rewards)
    
    def get_average_steps(self, last_n: int = 100) -> float:
        """
        Calculate average steps over last N episodes.
        
        Args:
            last_n: Number of recent episodes to average
        
        Returns:
            Average steps, or 0.0 if no episodes logged yet
        """
        if not self.episode_steps:
            return 0.0
        
        recent_steps = self.episode_steps[-last_n:]
        return sum(recent_steps) / len(recent_steps)
    
    @property
    def total_steps(self) -> int:
        """Total steps across all episodes."""
        return sum(self.episode_steps)
    
    @property
    def total_episodes(self) -> int:
        """Total number of episodes logged."""
        return len(self.episode_rewards)
    
    def get_best_reward(self) -> float:
        """Get the best reward achieved so far."""
        return max(self.episode_rewards) if self.episode_rewards else float('-inf')
    
    def get_metric(self, metric_name: str, last_n: Optional[int] = None) -> List[Any]:
        """
        Get stored metric values.
        
        Args:
            metric_name: Name of metric (e.g., 'loss', 'learning_rate')
            last_n: Return only last N values (None = all values)
        
        Returns:
            List of metric values
        """
        if metric_name == 'reward':
            data = self.episode_rewards
        elif metric_name == 'steps':
            data = self.episode_steps
        elif metric_name == 'duration':
            data = self.episode_durations
        elif metric_name == 'epsilon':
            data = self.epsilons
        elif metric_name in self._extra_metrics:
            data = self._extra_metrics[metric_name]
        else:
            return []
        
        if last_n is None:
            return data
        return data[-last_n:]
    
    def summary(self) -> str:
        """
        Generate summary statistics.
        
        Returns:
            Formatted summary string
        """
        if not self.episode_rewards:
            return "No episodes logged yet."
        
        summary = [
            "\n" + "="*60,
            "TRAINING SUMMARY",
            "="*60,
            f"Total episodes: {self.total_episodes}",
            f"Total steps: {self.total_steps:,}",
            f"Best reward: {self.get_best_reward():.2f}",
            f"Average reward (last 100): {self.get_average_reward(100):.2f}",
            f"Average steps (last 100): {self.get_average_steps(100):.1f}",
        ]
        
        if self.epsilons:
            summary.append(f"Final epsilon: {self.epsilons[-1]:.4f}")
        
        summary.append("="*60)
        
        return "\n".join(summary)



