import copy
import time

# =====================================================================
# 1. BASE CONFIGURATIONS (The Defaults)
# =====================================================================
PROJECT_BASE_CONFIG = {
    # Run settings
    "env_name": "SimpleGrid",
    "algo": "DQN",
    "obs_shape": (84, 84, 1),
    "seed": int(time.time()),
    "max_steps": 200,
    "training_episodes": 1000,
    "inference_episodes": 20,
    
    # DQN Hyperparameters
    "gamma": 0.99,
    "learning_rate": 2.5e-4,
    "epsilon_start": 1.0,
    "epsilon_min": 0.05,
    "epsilon_decay": 0.995,
    "batch_size": 32,
    "buffer_capacity": 100000,
    "min_buffer_size": 1000,
    "training_freq": 4,
    "target_update_freq": 1000,
    "grad_clip": 1.0,

    # A2C/PPO Hyperparameters
    "value_loss_coefficient": 0.5,
    "entropy_coefficient": 0.1,
    "max_grad_norm": 0.5,
    "use_per_step_update": True, # True = DQN, False = A2C/PPO

    # PPO-specific
    "clip_eps": 0.2,
    "gae_lambda": 0.95,
    "update_epochs": 4,
    
    # Default Reward Shaping
    "reward_shaping": {"step": 0.0, "goal": 1.0}
}

# --- ALGORITHM-SPECIFIC BASES ---
A2C_BASE_CONFIG = copy.deepcopy(PROJECT_BASE_CONFIG)
A2C_BASE_CONFIG.update({
    "algo": "A2C",
    "use_per_step_update": False,
    "learning_rate": 3e-4,
    "epsilon_start": None, "epsilon_min": None, "epsilon_decay": None,
    "batch_size": None, "buffer_capacity": None, "min_buffer_size": None,
    "target_update_freq": None,
})

PPO_BASE_CONFIG = copy.deepcopy(A2C_BASE_CONFIG)
PPO_BASE_CONFIG.update({
    "algo": "PPO",
    "entropy_coefficient": 0.01,
})


# =====================================================================
# 2. THE EXPERIMENT BUILDER (Clean & Intuitive)
# =====================================================================
def build_experiment(name: str, base_config: dict, **overrides) -> dict:
    """
    Intuitively creates an experiment dictionary.
    Inherits all properties from 'base_config' and overwrites specific keys with '**overrides'.
    """
    config = copy.deepcopy(base_config)
    config.update(overrides)
    return {
        "name": name,
        "config": config
    }

BALANCED_SHAPING = {
    "key": 1.0, "door": 2.0, "room_crossing": 2.5,
    "ball": 2.5, "goal": 5.0,
    "turn_penalty": 0.005,
    "step": 0.001,
    "invalid_action": 0.02
}

# =====================================================================
# 3. CURRENT EXPERIMENTS (Set 2)
# =====================================================================
# Shared globals for this set
SET2_MAX_STEPS = 400
SET2_EPISODES = 2000
SET2_INFERENCE_EPISODES = 100

# 1. PPO on KeyDoorBall
SET2_PPO_KDB = build_experiment(
    name="SET2_PPO_KDB",
    base_config=PPO_BASE_CONFIG,
    env_name="KeyDoorBall",
    training_episodes=SET2_EPISODES,
    inference_episodes=SET2_INFERENCE_EPISODES,
    max_steps=SET2_MAX_STEPS,
    reward_shaping=BALANCED_SHAPING
)

# 2. A2C on KeyDoorBall
SET2_A2C_KDB = build_experiment(
    name="SET2_A2C_KDB",
    base_config=A2C_BASE_CONFIG,
    env_name="KeyDoorBall",
    training_episodes=SET2_EPISODES,
    inference_episodes=SET2_INFERENCE_EPISODES,
    max_steps=SET2_MAX_STEPS,
    entropy_coefficient=0.01,
    reward_shaping=BALANCED_SHAPING
)

# 3. DQN on KeyDoorBall
SET2_DQN_KDB = build_experiment(
    name="SET2_DQN_KDB",
    base_config=PROJECT_BASE_CONFIG,
    env_name="KeyDoorBall",
    training_episodes=SET2_EPISODES,
    inference_episodes=SET2_INFERENCE_EPISODES,
    max_steps=SET2_MAX_STEPS,
    training_freq=10,
    reward_shaping=BALANCED_SHAPING
)

# =====================================================================
# 4. EXPORT SETS (For main.py to import)
# =====================================================================
# Just iterate over this list in your main.py to run the whole suite
exp_set_2 = [
    SET2_PPO_KDB, 
    SET2_A2C_KDB, 
    #SET2_DQN_KDB
]