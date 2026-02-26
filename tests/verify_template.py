import sys
import os
import numpy as np

# enable importing from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import the OFFICIAL (clean) template
# might print stuff/open windows
from tests.official_template import SimpleGridEnv, KeyDoorBallEnv

# import LOCAL implementations
from src.template import pre_process 
# from src.agent import DQNAgent # todo uncomment when agent implemented

def run_verification():
    print("\n Starting Verification...")

    try:
        test_preprocessing_compaitibility()
        test_agent_integration()
    except Exception as e:
        print(f"\n FAILED: {e}")

def test_agent_integration():
    print("   Checking Agent Integration...", end=" ")
    # todo: implement
    # agent = DQNAgent(...) 
    # action = agent.act(obs)
    print("  Skipped (agent not built yet)")

    print("\n SUCCESS: solution works with the official template")

def test_preprocessing_compaitibility():
    print("   Checking Preprocessing...", end=" ")
        
    # use OFFICIAL env + inject LOCAL pre_process function
    env = SimpleGridEnv(preprocess=pre_process, max_steps=10)
    obs, _ = env.reset()
        
    # verify shape
    expected_shape = (320, 320, 1)
    if obs.shape != expected_shape:
        raise ValueError(f"Shape mismatch: got {obs.shape}, expected {expected_shape}")
            
    # verify dtype
    if obs.dtype != np.uint8:
        raise ValueError(f"Dtype mismatch: got {obs.dtype}, expected uint8")
            
    print(" Passed")

if __name__ == "__main__":
    run_verification()