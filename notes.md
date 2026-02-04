JAN 21

Workflow:
* Develop as a python project - separate scripts, easier git, less conflicts, more maintainable. Copy/export to notebook for necessary testing (when possible, run/test directly in the .py script to reduce copy/pasting) or even just for the final delivery.
* Adhere to the template structure - whatever logic lives in the template stays there, and we import template.py as needed. It'll make final integration a copy/paste with minor adjustments for jupyter.

---

FEB 2

### 1. Multi-Agent Architecture Implemented
- **Status**: ✅ Working
- **Agents**: DQN, A2C
- **Key Changes**:
  - Created flexible base `Agent` class in `src/agents/agent.py`
  - Made `step()` and `store_transition()` optional (not abstract)
  - DQN uses replay buffer (off-policy)
  - A2C uses episode trajectories (on-policy)

### 2. File Structure
```
src/agents/
├── agent.py          # Base Agent class (4 required methods)
├── dqn_agent.py      # DQNAgent(Agent) - implements optional methods
└── a2c_agent.py      # A2CAgent(Agent) - only required methods

src/models/
├── networks.py       # DQN and ActorCriticNetwork
└── replay_buffer.py

src/training/
├── dqn_trainer.py    # train() for DQN
└── a2c_trainer.py    # train_a2c() for A2C

configs/
├── simple_grid_config.yaml          # DQN on SimpleGrid
├── key_door_ball_config.yaml        # DQN on KeyDoorBall
└── key_door_ball_a2c_config.yaml    # A2C on KeyDoorBall
```

### 3. Key Design Decisions

#### Base Agent Class (`agent.py`)
- **Required methods** (abstract):
  - `choose_action(obs, **kwargs)` - Flexible for epsilon/temperature/etc.
  - `update(*args, **kwargs)` - Works for batches or trajectories
  - `save(filepath)` - Save checkpoint
  - `load(filepath)` - Load checkpoint

- **Optional methods** (not abstract):
  - `step()` - Only DQN implements (per-step updates)
  - `store_transition()` - Only DQN implements (replay buffer)

#### Why This Works
- DQN: Off-policy, uses replay buffer, updates per step
- A2C: On-policy, uses trajectories, updates per episode
- Easy to add PPO, SAC, etc. later

### 4. Command-Line Interface
```bash
# Train A2C on KeyDoorBall (default)
python main.py

# Train DQN on SimpleGrid
python main.py --config configs/simple_grid_config.yaml

# Custom seed
python main.py --config configs/simple_grid_config.yaml --seed 123
```

### 5. Next Steps (TODO)
- [ ] Train A2C on KeyDoorBallEnv (1000 episodes)
- [ ] Compare DQN vs A2C performance
- [ ] Add reward shaping for KeyDoorBall
- [ ] Implement PPO (optional)
- [ ] Hyperparameter tuning


## 🔑 Key Code Snippets

### Adding a New Algorithm
```python
# 1. Create agent in src/agents/new_agent.py
from src.agents.agent import Agent

class NewAgent(Agent):
    def choose_action(self, obs, **kwargs):
        pass
    def update(self, data):
        pass
    def save(self, filepath):
        pass
    def load(self, filepath):
        pass

# 2. Add to main.py create_agent()
elif algo == 'NewAlgo':
    agent = NewAgent(...)

# 3. Add trainer to main.py get_trainer()
trainer_map = {
    'DQN': train,
    'A2C': train_a2c,
    'NewAlgo': train_new,  # ← Add this
}
```

## Current Training Status
- **DQN on SimpleGrid**: Working
- **DQN on KeyDoorBall**: Not tested yet
- **A2C on KeyDoorBall**: Ready to train

## Common Issues Resolved
1. `BaseAgent` → ✅ `Agent` (renamed class)
2. Abstract `step()` → ✅ Optional (only DQN needs it)
3. Abstract `store_transition()` → ✅ Optional (only DQN needs it)
4. Hardcoded `epsilon` → ✅ `**kwargs` (flexible)
5. DQN-specific prints → ✅ Algorithm-agnostic `print_training_header()`

---

FEB 4

Working on KeyDoorBall environment with A2C agent. Agent successfully completes first 2 subtasks (key pickup + door opening) but fails to cross the door.

### Investigation
**Hypothesis**: Agent wastes actions through inefficient movement  
**Tool**: Created `analyze_actions.py` to track action patterns  
**Finding**: Confirmed - agent exhibits high turn-back behavior (turning left then immediately right, or vice versa)

### Initial Fix (v1)
- Added turn-back penalty: -0.05
- Added forward movement reward: +0.002
- Added door-crossing urgency penalties
- **Result**: Minimal improvement (~0.2-0.3 reduction in wasteful actions)

### Aggressive Fix (v2) - Current
**Problem Metrics**:
- Turn-backs: 15.93/episode
- Turn/move ratio: 1.99 (agent turns 2x more than it moves)

**Changes**:
- Stronger turn-back penalties: -0.10 base, progressive up to -0.25
- Higher forward reward: +0.005
- Increased door-crossing penalties: up to -0.40
- Extended training: 250 steps/episode, 2000 episodes
- Higher exploration: entropy_coef 0.1
- Fixed visualization scripts (layouts + path resolution)

### Key Insight
Agent's problem isn't total turning (39.8% of actions), but *wasteful* turning (constant direction reversal = indecision). Progressive penalties aims to teach commitment to decisions.
