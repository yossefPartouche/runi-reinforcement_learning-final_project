"""
Test suite for MiniGrid environments with preprocessing integration.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from src.environments.simple_grid_env import SimpleGridEnv
from src.environments.key_door_ball_env import KeyDoorBallEnv
from src.preprocessing.image_preprocessing import preprocess_observation

def test_simple_grid_env_creation():
    """Test SimpleGridEnv can be created."""
    print("\n" + "="*60)
    print("Testing SimpleGridEnv creation...")
    print("="*60)
    
    env = SimpleGridEnv(
        size=10,
        max_steps=1000,
        preprocess=preprocess_observation
    )
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space} (n={env.action_space.n})")
    
    assert env.observation_space.shape == (84, 84, 1), \
        f"Expected shape (84, 84, 1), got {env.observation_space.shape}"
    assert env.observation_space.dtype == np.float32, \
        f"Expected dtype float32, got {env.observation_space.dtype}"
    assert env.action_space.n == 3, f"Expected 3 actions, got {env.action_space.n}"
    
    print("✅ SimpleGridEnv configuration is correct")


def test_simple_grid_env_reset():
    """Test SimpleGridEnv reset produces valid observations."""
    print("\n" + "="*60)
    print("Testing SimpleGridEnv reset...")
    print("="*60)
    
    env = SimpleGridEnv(size=10, max_steps=1000, preprocess=preprocess_observation)
    obs, info = env.reset()
    
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Verify observation matches observation_space
    assert obs.shape == env.observation_space.shape, \
        f"Obs shape {obs.shape} doesn't match space {env.observation_space.shape}"
    assert obs.dtype == env.observation_space.dtype, \
        f"Obs dtype {obs.dtype} doesn't match space {env.observation_space.dtype}"
    assert obs.min() >= 0.0 and obs.max() <= 1.0, \
        f"Obs range [{obs.min()}, {obs.max()}] outside [0, 1]"
    
    print("✅ Observation matches observation_space")


def test_simple_grid_env_step():
    """Test SimpleGridEnv step function."""
    print("\n" + "="*60)
    print("Testing SimpleGridEnv step...")
    print("="*60)
    
    env = SimpleGridEnv(size=10, max_steps=1000, preprocess=preprocess_observation)
    obs, info = env.reset()
    
    print("   Running 10 random steps...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Verify observation format
        assert obs.shape == env.observation_space.shape, \
            f"Step {i}: obs shape mismatch"
        assert obs.dtype == env.observation_space.dtype, \
            f"Step {i}: obs dtype mismatch"
        
        print(f"   Step {i+1}: action={action}, reward={reward:.2f}, "
              f"terminated={terminated}, truncated={truncated}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {i+1}")
            obs, info = env.reset()
            break
    
    print("✅ SimpleGridEnv step function works correctly")


def test_key_door_ball_env_creation():
    """Test KeyDoorBallEnv can be created."""
    print("\n" + "="*60)
    print("Testing KeyDoorBallEnv creation...")
    print("="*60)
    
    env = KeyDoorBallEnv(
        size=10,
        max_steps=1000,
        preprocess=preprocess_observation
    )
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space} (n={env.action_space.n})")
    
    assert env.observation_space.shape == (84, 84, 1), \
        f"Expected shape (84, 84, 1), got {env.observation_space.shape}"
    assert env.observation_space.dtype == np.float32, \
        f"Expected dtype float32, got {env.observation_space.dtype}"
    assert env.action_space.n == 5, f"Expected 5 actions, got {env.action_space.n}"
    
    print("✅ KeyDoorBallEnv configuration is correct")


def test_key_door_ball_env_reset():
    """Test KeyDoorBallEnv reset produces valid observations."""
    print("\n" + "="*60)
    print("Testing KeyDoorBallEnv reset...")
    print("="*60)
    
    env = KeyDoorBallEnv(size=10, max_steps=1000, preprocess=preprocess_observation)
    obs, info = env.reset()
    
    print(f"✅ Environment reset successful")
    print(f"   Observation shape: {obs.shape}")
    print(f"   Observation dtype: {obs.dtype}")
    print(f"   Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Verify observation matches observation_space
    assert obs.shape == env.observation_space.shape, \
        f"Obs shape {obs.shape} doesn't match space {env.observation_space.shape}"
    assert obs.dtype == env.observation_space.dtype, \
        f"Obs dtype {obs.dtype} doesn't match space {env.observation_space.dtype}"
    
    print("✅ Observation matches observation_space")


def test_key_door_ball_env_actions():
    """Test KeyDoorBallEnv with all action types."""
    print("\n" + "="*60)
    print("Testing KeyDoorBallEnv actions...")
    print("="*60)
    
    env = KeyDoorBallEnv(size=10, max_steps=1000, preprocess=preprocess_observation)
    obs, info = env.reset()
    
    action_names = ['Turn Left', 'Turn Right', 'Move Forward', 'Pickup', 'Toggle']
    
    print("   Testing all action types...")
    for action in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert obs.shape == env.observation_space.shape, \
            f"Action {action}: obs shape mismatch"
        
        print(f"   Action {action} ({action_names[action]}): "
              f"reward={reward:.2f}, terminated={terminated}")
        
        if terminated or truncated:
            obs, info = env.reset()
            break
    
    print("✅ All KeyDoorBallEnv actions work correctly")


def run_all_tests():
    """Run all environment tests."""
    print("\n" + "🧪 "*30)
    print("ENVIRONMENT TESTS")
    print("🧪 "*30)
    
    try:
        # SimpleGridEnv tests
        test_simple_grid_env_creation()
        test_simple_grid_env_reset()
        test_simple_grid_env_step()
        
        # KeyDoorBallEnv tests
        test_key_door_ball_env_creation()
        test_key_door_ball_env_reset()
        test_key_door_ball_env_actions()
        
        print("\n" + "="*60)
        print("🎉 ALL ENVIRONMENT TESTS PASSED! 🎉")
        print("="*60)
        print("\nYour environments are correctly configured and ready to use!")
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ TEST FAILED")
        print("="*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    run_all_tests()