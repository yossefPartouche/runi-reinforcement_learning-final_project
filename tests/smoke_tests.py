"""
Smoke Test(s) - verify full pipeline runs e2e w/o crashing
Run with: pytest tests/test_smoke.py -v
"""
import copy
import numpy as np
import os
import shutil
import pytest
from typing import Dict, Tuple

from src.agent import DQNAgent, A2CAgent
from src.experiments import (
    DQN_SIMPLEGRID_BASELINE,
    A2C_SIMPLEGRID_BASELINE,
    DQN_KEYDOORBALL_BASELINE,
    A2C_KEYDOORBALL_BASELINE,
    PPO_SIMPLEGRID_BASELINE,
    PPO_KEYDOORBALL_BASELINE,
)
from src.experiment_runner import Experiment
from src.utils import get_device

# --- helpers ---

SMOKE_RESULTS_DIR = "results/_smoke_tests"

def create_smoke_config(base_config: Dict, training_episodes: int = 5, inference_episodes: int = 3) -> Dict:
    """Returns minimal copy of a config suitable for smoke tests"""
    config = copy.deepcopy(base_config)
    config["training_episodes"] = training_episodes
    config["inference_episodes"] = inference_episodes
    config["max_steps"] = 50          # short episodes
    config["min_buffer_size"] = 10    # DQN: don't wait long before first update
    config["batch_size"] = 8          # DQN: smaller batch for small buffer
    config["seed"] = 42
    return config


@pytest.fixture(scope="session", autouse=True)
def cleanup_smoke_results():
    """Remove smoke test results folder after the session"""
    yield
    if os.path.exists(SMOKE_RESULTS_DIR):
        shutil.rmtree(SMOKE_RESULTS_DIR)


def run_smoke_test(base_config: Dict, exp_name: str) -> Tuple[Dict, Dict]:
    """
    Shared helper: 
    - builds experiment
    - runs train + inference
    - checks outputs
    """
    config = create_smoke_config(base_config)
    device = get_device()
    exp = Experiment(config=config, exp_name=f"smoke_{exp_name}", device=device)

    # override results dir to keep smoke results isolated
    exp.results_dir = os.path.join(SMOKE_RESULTS_DIR, exp_name)
    exp.logger.results_dir = exp.results_dir
    exp.video_recorder.save_dir = exp.results_dir
    os.makedirs(exp.results_dir, exist_ok=True)

    # train
    train_metrics, train_time = exp.train()
    assert isinstance(train_metrics, Dict), "train() must return a Dict"
    assert train_time > 0, "train_time must be positive"

    # evaluate
    inference_metrics, inference_time = exp.evaluate()
    assert isinstance(inference_metrics, Dict), "evaluate() must return a Dict"
    assert inference_time > 0, "inference_time must be positive"

    # check CSV logs were created
    assert os.path.exists(os.path.join(exp.results_dir, "training_log.csv")), "training_log.csv missing"
    assert os.path.exists(os.path.join(exp.results_dir, "inference_log.csv")), "inference_log.csv missing"

    return train_metrics, inference_metrics


# --- DQN on SimpleGrid ---

def test_smoke_1_dqn_simplegrid():
    """1 — DQN on SimpleGrid: full pipeline runs w/o crash"""
    train_metrics, inference_metrics = run_smoke_test(DQN_SIMPLEGRID_BASELINE["config"], "dqn_simplegrid")
    assert "train_episodes" in train_metrics
    assert "inference_success_rate" in inference_metrics


# --- A2C on SimpleGrid ---

def test_smoke_2_a2c_simplegrid():
    """2 — A2C on SimpleGrid: full pipeline runs w/o crash"""
    train_metrics, inference_metrics = run_smoke_test(A2C_SIMPLEGRID_BASELINE["config"], "a2c_simplegrid")
    assert "train_episodes" in train_metrics
    assert "inference_success_rate" in inference_metrics


# --- DQN on KeyDoorBall ---

def test_smoke_3_dqn_keydoorball():
    """3 — DQN on KeyDoorBall: full pipeline + reward shaping runs w/o crash"""
    train_metrics, inference_metrics = run_smoke_test(DQN_KEYDOORBALL_BASELINE["config"], "dqn_keydoorball")
    assert "train_episodes" in train_metrics
    assert "inference_success_rate" in inference_metrics


# --- A2C on KeyDoorBall ---

def test_smoke_4_a2c_keydoorball():
    """4 — A2C on KeyDoorBall: full pipeline + reward shaping runs w/o crash"""
    train_metrics, inference_metrics = run_smoke_test(A2C_KEYDOORBALL_BASELINE["config"], "a2c_keydoorball")
    assert "train_episodes" in train_metrics
    assert "inference_success_rate" in inference_metrics


# --- Checkpoint save/load round-trip ---

def test_smoke_5_dqn_checkpoint_roundtrip():
    """5 — DQN checkpoint saves and loads correctly"""
    config = create_smoke_config(DQN_SIMPLEGRID_BASELINE["config"])
    device = get_device()
    obs_shape = config["obs_shape"]
    num_actions = 3  # SimpleGrid

    agent = DQNAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    agent.steps_done = 99
    agent.epsilon = 0.42

    path = os.path.join(SMOKE_RESULTS_DIR, "test_checkpoint.pt")
    os.makedirs(SMOKE_RESULTS_DIR, exist_ok=True)
    agent.save(path)
    assert os.path.exists(path), "Checkpoint file not created"

    # load into fresh agent + verify state restored
    agent2 = DQNAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    agent2.load(path)
    assert agent2.steps_done == 99, f"steps_done mismatch: {agent2.steps_done}"
    assert abs(agent2.epsilon - 0.42) < 1e-6, f"epsilon mismatch: {agent2.epsilon}"


def test_smoke_6_a2c_checkpoint_roundtrip():
    """6 — A2C checkpoint saves and loads correctly"""
    config = create_smoke_config(A2C_SIMPLEGRID_BASELINE["config"])
    device = get_device()
    obs_shape = config["obs_shape"]
    num_actions = 3  # SimpleGrid

    agent = A2CAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    agent.steps_done = 77

    path = os.path.join(SMOKE_RESULTS_DIR, "test_a2c_checkpoint.pt")
    os.makedirs(SMOKE_RESULTS_DIR, exist_ok=True)
    agent.save(path)
    assert os.path.exists(path), "A2C checkpoint file not created"

    agent2 = A2CAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    agent2.load(path)
    assert agent2.steps_done == 77, f"steps_done mismatch: {agent2.steps_done}"


# --- A2C deterministic eval mode ---

def test_smoke_7_a2c_deterministic_eval():
    """7 — A2C choose_action with epsilon=0.0 is deterministic (same obs -> same action)"""
    config = create_smoke_config(A2C_SIMPLEGRID_BASELINE["config"])
    device = get_device()
    obs_shape = config["obs_shape"]
    num_actions = 3

    agent = A2CAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)

    # fixed dummy obs
    obs = np.zeros(obs_shape, dtype=np.uint8)
    obs[10, 10, 0] = 128  # non-trivial pixel

    actions = [agent.choose_action(obs, epsilon=0.0) for _ in range(10)]
    assert len(set(actions)) == 1, f"A2C greedy not deterministic: got {set(actions)}"

# --- PPO ---

def test_smoke_8_ppo_simplegrid():
    """8 — PPO on SimpleGrid: full pipeline runs w/o crash"""
    train_metrics, inference_metrics = run_smoke_test(PPO_SIMPLEGRID_BASELINE["config"], "ppo_simplegrid")
    assert "train_episodes" in train_metrics
    assert "inference_success_rate" in inference_metrics


def test_smoke_9_ppo_keydoorball():
    """9 — PPO on KeyDoorBall: full pipeline + reward shaping runs w/o crash"""
    train_metrics, inference_metrics = run_smoke_test(PPO_KEYDOORBALL_BASELINE["config"], "ppo_keydoorball")
    assert "train_episodes" in train_metrics
    assert "inference_success_rate" in inference_metrics


def test_smoke_10_ppo_checkpoint_roundtrip():
    """10 — PPO checkpoint saves and loads correctly"""
    from src.agent import PPOAgent

    config = create_smoke_config(PPO_SIMPLEGRID_BASELINE["config"])
    device = get_device()
    obs_shape = config["obs_shape"]
    num_actions = 3  # SimpleGrid

    agent = PPOAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    agent.steps_done = 55

    path = os.path.join(SMOKE_RESULTS_DIR, "test_ppo_checkpoint.pt")
    os.makedirs(SMOKE_RESULTS_DIR, exist_ok=True)
    agent.save(path)
    assert os.path.exists(path), "PPO checkpoint file not created"

    agent2 = PPOAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    agent2.load(path)
    assert agent2.steps_done == 55, f"steps_done mismatch: {agent2.steps_done}"


def test_smoke_11_ppo_returns_log_prob():
    """11 — PPO choose_action returns (action, log_prob) tuple during training"""
    import numpy as np
    from src.agent import PPOAgent

    config = create_smoke_config(PPO_SIMPLEGRID_BASELINE["config"])
    device = get_device()
    obs_shape = config["obs_shape"]
    num_actions = 3

    agent = PPOAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    obs = np.zeros(obs_shape, dtype=np.uint8)

    result = agent.choose_action(obs)  # training mode — no epsilon arg
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    action, log_prob = result
    assert isinstance(action, int), f"Expected int action, got {type(action)}"
    assert isinstance(log_prob, float), f"Expected float log_prob, got {type(log_prob)}"
    assert 0 <= action < num_actions


def test_smoke_12_ppo_deterministic_eval():
    """12 — PPO choose_action with epsilon=0.0 is deterministic and returns int"""
    import numpy as np
    from src.agent import PPOAgent

    config = create_smoke_config(PPO_SIMPLEGRID_BASELINE["config"])
    device = get_device()
    obs_shape = config["obs_shape"]
    num_actions = 3

    agent = PPOAgent(config=config, obs_shape=obs_shape, num_actions=num_actions, device=device)
    obs = np.zeros(obs_shape, dtype=np.uint8)
    obs[10, 10, 0] = 128

    results = [agent.choose_action(obs, epsilon=0.0) for _ in range(10)]
    # each result is (action, 0.0) tuple
    actions = [r[0] for r in results]
    assert len(set(actions)) == 1, f"PPO greedy not deterministic: got {set(actions)}"