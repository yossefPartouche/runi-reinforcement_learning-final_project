import csv
import json
import os
import random
import time
from collections import deque
from functools import wraps
from typing import Dict, Optional, Tuple

import imageio
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from src.agent import BaseAgent
from src.template import BaseMiniGridEnv


def set_random_seed(seed: int) -> None:
    """
    Sets seed for random number generator
    - cmopatible with CPU/GPU
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_device() -> torch.device:
    """Returns the device that runs training (GPU/CPU)"""
    processor_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {processor_type}")
    return torch.device(processor_type)


# ** TIMER UTIL(S) **
def timer(func):
    """
    Measures running time of decorated function
    :returns: (function output, time delta)
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Tuple[object, float]:
        # sync CPU & GPU if using cuda
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        result = func(*args, **kwargs)      # output of decorated function
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        time_delta = (end_time - start_time)
        # print(f"Function {func.__name__} took {time_delta}")
        return result, time_delta
    return wrapper


class MetricsHandler:
    """Helper class to track/print/log metrics """

    def __init__(self, num_episodes: int, window_size: int = None):
        self.num_episodes = num_episodes
        self.window = window_size
        if window_size:
            self.rewards = deque(maxlen=window_size)
            self.step_counts = deque(maxlen=window_size)
            self.success_counts = deque(maxlen=window_size)
        else:
            self.rewards = []
            self.step_counts = []
            self.success_counts = []
        
    def update(self, reward: float, steps: int, success: bool):
        self.rewards.append(reward)
        self.step_counts.append(steps)
        self.success_counts.append(int(success))

    @property
    def avg_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def avg_steps(self) -> float:
        return sum(self.step_counts) / len(self.step_counts) if self.step_counts else 0.0

    @property
    def success_rate(self) -> float:
        return sum(self.success_counts) / len(self.success_counts) if self.success_counts else 0.0

    def get_inference_metrics(self) -> Dict:
        return {
            "inference_episodes": self.num_episodes,
            "inference_success_rate": self.success_rate,
            "inference_avg_reward": self.avg_reward,
            "inference_avg_steps": self.avg_steps
        }

    def get_training_metrics(self, epsilon: float) -> Dict:
        return {
            "train_episodes": self.num_episodes,
            "train_final_epsilon": epsilon,
            "train_window_success_rate": self.success_rate,
            "train_window_avg_reward": self.avg_reward,
            "train_window_avg_steps": self.avg_steps
        }

    def print_training_status(self, episode: int, epsilon: float | None = None) -> None:
        print(f"\rEpisode {episode}/{self.num_episodes}", end="", flush=True)
        if self.window and episode % self.window == 0:
            epsilon_str = f"{epsilon:.3f}" if epsilon is not None else "n/a" # handle epsilon=None case
            print(f"\rEpisodes {episode-self.window}-{episode}/{self.num_episodes} | "
                  f"Avg R: {self.avg_reward:.2f} | Avg Steps: {self.avg_steps:.1f} | "
                  f"Success Rate: {self.success_rate:.2f} | ε: {epsilon_str}")


# ##########
# UTILS FOR RECORDING RESULTS (PLOTTING, REPORTS)
# ##########
def plot_training_curves(log_dir: str, window: int = 50) -> None:
    """
    Generates single-run convergence graphs with smoothing
    - reqards
    - steps
    - success
    """

    csv_path = os.path.join(log_dir, "training_log.csv")
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    # create 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # rewards
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='tab:blue')
    if len(df) > window:
        ax1.plot(df['episode'], df['reward'].rolling(window).mean(), color='tab:blue', linewidth=2)
    ax1.set_title('Reward')
    ax1.set_xlabel('Episode')

    # steps
    ax2.plot(df['episode'], df['steps'], alpha=0.3, color='tab:orange')
    if len(df) > window:
        ax2.plot(df['episode'], df['steps'].rolling(window).mean(), color='tab:orange', linewidth=2)
    ax2.set_title('Steps to Goal')
    ax2.set_xlabel('Episode')

    # success rate (rolling avg of binary success column)
    if 'success' in df.columns:
        success_data = pd.to_numeric(df['success'], errors='coerce').fillna(0) # cleans col naming/type issues
        if len(df) > window:
            success_data = success_data.rolling(window).mean()

        ax3.plot(df['episode'], success_data, color='tab:green', linewidth=2)
        ax3.set_title(f'Success Rate (Rolling {window})')
        ax3.set_ylim(0, 1.1)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Success %')
    else:
        print(f"success column missing, available columns: {df.columns}")
    
    plt.tight_layout()
    save_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(save_path)
    plt.close()


def plot_milestone_progress(log_dir: str, window: int = 50) -> None:
    """
    Plots cumulative milestone completion counts over training episodes
    **KeyDoorBall experiments only (file won't exist for SimpleGrid)

    Milestones tracked: 
    - key pickup
    - door open
    - room crossing
    - ball pickup
    - goal reached
    """
    csv_path = os.path.join(log_dir, "milestone_log.csv")
    if not os.path.exists(csv_path):
        return  # skip for SimpleGrid exps

    df = pd.read_csv(csv_path)

    milestones = {
        "got_key":      ("Key Pickup",     "tab:blue"),
        "opened_door":  ("Door Opened",    "tab:orange"),
        "has_crossed_door": ("Room Crossed", "tab:green"),
        "got_ball":     ("Ball Pickup",    "tab:purple"),
        "success":      ("Goal Reached",   "tab:red"),
    }

    # merge with training_log to get success col
    training_csv = os.path.join(log_dir, "training_log.csv")
    if os.path.exists(training_csv):
        train_df = pd.read_csv(training_csv)[["episode", "success"]]
        df = df.merge(train_df, on="episode", how="left")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # left plot: cumulative counts
    for col, (label, color) in milestones.items():
        if col not in df.columns:
            continue
        cumsum = pd.to_numeric(df[col], errors="coerce").fillna(0).cumsum()
        ax1.plot(df["episode"], cumsum, label=label, color=color, linewidth=2)

    ax1.set_title("Cumulative Milestone Completions")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Count")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # right plot: rolling success rate per milestone (conditional progress)
    for col, (label, color) in milestones.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if len(df) > window:
            series = series.rolling(window).mean()
        ax2.plot(df["episode"], series, label=label, color=color, linewidth=2)

    ax2.set_title(f"Milestone Rate (Rolling {window} eps)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rate")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(log_dir, "milestone_progress.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Milestone plot saved to: {save_path}")


def plot_comparison(run_directories: Dict, window: int = 50, save_dir: str = "results") -> None:
    """
    Plots multiple runs to compare algorithms    
    :param run_directories: dict like {'DQN': 'results/DQN_.../', 'DoubleDQN': 'results/Double_.../'}
    """

    plt.figure(figsize=(12, 6))
    
    for label, log_dir in run_directories.items():
        csv_path = os.path.join(log_dir, "training_log.csv")
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        if len(df) > window:
            # plot only smoothed curve for clarity
            rolling_avg = df['reward'].rolling(window=window).mean()
            plt.plot(df['episode'], rolling_avg, linewidth=2, label=label)
        else:
            plt.plot(df['episode'], df['reward'], linewidth=2, label=label)

    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title(f'Algorithm Comparison (Smoothed over {window} eps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(save_dir, "comparison_plot.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to: {save_path}")


def analyze_inference(log_dir: str) -> None:
    """
    Reads inference_log.csv, calculates stats, generates report & plots
    """
    csv_path = os.path.join(log_dir, "inference_log.csv")
    if not os.path.exists(csv_path):
        print(f"No inference log found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        
        # calculate stats:
        avg_reward = df['reward'].mean()
        avg_steps = df['steps'].mean()
        success_rate = df['success'].mean() * 100
        
        print(f"Inference Analysis:")
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Avg Steps:    {avg_steps:.1f}")

        # save text report
        report_path = os.path.join(log_dir, "inference_report.txt")
        with open(report_path, "w") as f:
            f.write(f"Inference Analysis ({len(df)} episodes):\n")
            f.write(f"Success Rate: {success_rate:.2f}%\n")
            f.write(f"Avg Reward:   {avg_reward:.4f}\n")
            f.write(f"Avg Steps:    {avg_steps:.2f}\n")
            f.write(f"Std Dev Steps:{df['steps'].std():.2f}\n")

        # histogram plot
        plt.figure(figsize=(8, 6))
        
        # todo consider success only?
        success_steps = df[df['success'] == 1]['steps']
        fail_steps = df[df['success'] == 0]['steps']
        plt.hist(
            [success_steps, fail_steps], 
            bins=20,
            stacked=True,
            color=['green', 'red'],
            alpha=0.7, 
            label=['Success', 'Failure']
        )

        # plot avg steps line
        plt.axvline(avg_steps, color='red', linestyle='dashed', linewidth=2, label=f'Avg Steps: {avg_steps:.1f}')

        plt.title(f"Inference: Steps Distribution (Success: {success_rate:.0f}%)")
        plt.xlabel("Steps to Goal")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(log_dir, "inference_histogram.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Inference plots saved to: {save_path}")

    except Exception as e:
        print(f"Error analyzing inference: {e}")
         

def save_experiment_report(log_dir: str, config: Dict, metrics: Dict, timings: Dict) -> None:
    """
    Collects high-level experiment information in JSON report
    - config
    - results
    - time measurements
    """
    report = {
        "meta": {
            "algo": config.get("algo"),
            "env": config.get("env_name"),
            "timestamp": time.strftime("%Y%m%d-%H%M%S")
        },
        "configuration": config,
        "performance": timings,
        "results": metrics
    }
    
    report_path = os.path.join(log_dir, "experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Experiment report saved to: {report_path}")


class ExperimentLogger:
    def __init__(self, save_dir: str):
        self.results_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log(self, filename: str, **metrics):
        """
        Appends a dict of metrics to a CSV file
        Creates the file and headers automatically on the first call.
        """
        path = os.path.join(self.results_dir, f"{filename}.csv")
        
        # convert dict to single-row df
        df = pd.DataFrame([metrics])
        
        # append to CSV
        write_header = not os.path.exists(path)
        df.to_csv(
            path_or_buf=path, 
            mode='a',               # apend mode
            header=write_header,    # if file is new
            index=False
        )


class VideoRecorder:
    """Handles episode video recording, saves to MP4"""
    def __init__(self, save_dir: str, env: BaseMiniGridEnv, fps: int = 10):
        self.save_dir = save_dir
        self.fps = fps
        self.frames = []
        self.recording = False
        self.filename = None
        self.env = env

    def start(self, stage: str) -> None:
        self.recording = True
        self.frames = []
        self.filename = f"{stage}.mp4"

    def capture(self) -> None:
        if self.recording:
            try:
                frame = self.env.render() # render mode must be 'rgb_array'
                self.frames.append(frame)
            except Exception as e:
                print(f"WARNING: Failed to capture video frame: {e}")

    def stop(self) -> None:
        if self.recording and self.frames:
            path = os.path.join(self.save_dir, self.filename)
            try:
                imageio.mimsave(uri=path, ims=self.frames, fps=self.fps)
                print(f"Video saved: {path}")
            except Exception as e:
                print(f"Video save failed: {e}")
        self.recording = False
        self.frames = []