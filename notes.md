JAN 21 (Shay)

Workflow:
* Develop as a python project - separate scripts, easier git, less conflicts, more maintainable. Copy/export to notebook for necessary testing (when possible, run/test directly in the .py script to reduce copy/pasting) or even just for the final delivery.
* Adhere to the template structure - whatever logic lives in the template stays there, and we import template.py as needed. It'll make final integration a copy/paste with minor adjustments for jupyter.

---

FEB 2 (Yossi)

### 1. Multi-Agent Architecture Implemented
- **Status**: ✅ Working
- **Agents**: DQN, A2C
- **Key Changes**:
  - Created flexible base `Agent` class in `src/agents/agent.py`
  - Made `step()` and `store_transition()` optional (not abstract)
  - DQN uses replay buffer (off-policy)
  - A2C uses episode trajectories (on-policy)

### 2. File Structure (Yossi's branch — later reorganized into flat `src/`)
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

### 4. Common Issues Resolved
1. `BaseAgent` → ✅ `Agent` (renamed class)
2. Abstract `step()` → ✅ Optional (only DQN needs it)
3. Abstract `store_transition()` → ✅ Optional (only DQN needs it)
4. Hardcoded `epsilon` → ✅ `**kwargs` (flexible)
5. DQN-specific prints → ✅ Algorithm-agnostic `print_training_header()`

---

FEB 4 (Yossi)

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

---

FEB 12 (Yossi)

- Simplified the reward system, making it more sparse, at the critical points
- Fixed farming problem of subtasks reward shaping, by enforcing reward only once per completion of subtask per episode
- Cleaned variables duplication
- Implemented milestone tracking systems, for subtasks across training session
- Added milestone progress visualisation

**Current result** after 1000 episodes of training A2C (max 300 steps per ep)

783/1000 - key pickups
246/1000 - door Opens
89/1000 - Room Crossing
25/1000 - Ball pickup
2/1000 - Final Destination reached

**Improvement** (Increased the number of episodes to 3000 and steps per episodes to 500)

2666/3000 - key pickups (+10.6% Improvement)
1608/3000 - door Opens (+29.0% Improvement)
1044/3000 - Room Crossing (+25.9% Improvement)
448/3000 - Ball pickup (+12.4% Improvement)
238/3000 - Final Destination reached (+4.1% Improvement)

### Key Insights
1. **Time matters**: 300 steps wasn't enough for full navigation sequence
2. **Data matters**: Agent needed 3x more episodes to learn task dependencies
3. **No reward shaping needed yet**: Sparse rewards + more data worked well
4. **Bottleneck shifted**: Now stuck at ball pickup → goal (29% conditional success)

---

FEB 23 (Shay)

### Codebase Integration: Yossi's branch → infra_merge

Decided to keep Shaytanne's flat `src/` infra as the base and port Yossi's algorithmic work into it. Rationale:
- Shaytanne's infra has batch experiment runner, isolated result folders, proper train/eval split, and richer analysis utils
- Yossi's infra has better agent base class design, full checkpoint save/load, and good algorithmic implementations (DQN, A2C)
- Manual port preferred over git merge due to divergent directory structures

Ported from Yossi:
- `BaseAgent` improvements: `update()` abstract, `load()` abstract, `step()` optional with descriptive error
- `DQNAgent.save()` upgraded to full checkpoint (policy net, target net, optimizer, steps, epsilon, config)
- `min_buffer_size` config param added to DQN training condition
- `A2CAgent` class ported into `src/agent.py` (adapted to Shaytanne's H,W,C uint8 obs convention)
- `ActorCriticNetwork` ported into `src/model.py` (accepts H,W,C, handles C,H,W transpose internally)
- A2C training path added to `experiment_runner.py` via `use_per_step_update` flag
- A2C wired into `_determine_agent_class()`

Observation space decision: kept `(84, 84, 1) uint8` (H,W,C) convention throughout. Both DQNAgent and A2CAgent normalize + transpose internally in `choose_action()` and `update()`. No env change needed.

---

FEB 25 (Shay)

### Phase 5: Reward Shaping + Experiment Configs

#### Staff clarification received (FEB 25)
Course staff explicitly expanded what's allowed in reward shaping:
- Auxiliary variables permitted in `__init__`, `reset`, and `step` (not just the marked reward block)
- Must represent discrete task-related events — NOT continuous/distance-based signals
- Step penalty explicitly allowed
- Distance to goal/key/door/ball explicitly prohibited

#### KeyDoorBallEnv reward shaping implemented (`src/template.py`)
Added to `__init__` and `reset()` (per staff clarification):
- `prev_action` — tracks last action for turn-back penalty
- `prev_pos` — tracks last position for room-crossing detection
- `has_crossed_door` — one-time flag, prevents reward farming on room crossing

Reward signals added to `step()`:
| Signal | Value (default) | Trigger |
|---|---|---|
| `key` | +0.5 | Key pickup (transition: no key → have key) |
| `door` | +0.5 | Door opened (transition: had key + door closed → door open) |
| `room_crossing` | +1.0 | First crossing from left to right room (once per episode) |
| `ball` | +0.5 | Ball pickup (transition: no ball → have ball) |
| `goal` | +2.0 | Goal reached with ball |
| `turn_penalty` | -0.1 | Immediate direction reversal (left→right or right→left) |
| `step` | -0.001 | Every step (encourages shorter solutions) |

Room-crossing implemented as binary discrete check (`agent_pos[0] > partition_col`) — not distance-based, qualifies as "progressing between task stages" per staff clarification.

#### Experiment configs updated (`src/experiments.py`)
- Added `DQN_KEYDOORBALL_BASELINE` (experiment 7) with full reward shaping config
- Updated `A2C_KEYDOORBALL_BASELINE` (renumbered to experiment 8) with full reward shaping config
- Both KeyDoorBall configs include all 7 reward keys
- SimpleGrid configs unchanged (only use `step` + `goal`)

#### Phase 5 checklist status
- ✅ Phases 1, 2, 3 (agent improvements, A2C, training loop) — done during infra merge
- ✅ Phase 5 (reward shaping + env enhancements + obs space decision)
- ⏭️ Phase 4 (PrioritizedReplayBuffer) — deferred, lowest priority
- ⬜ Phase 6 (milestone logging/visualization) — pending
- ⬜ Phase 8 (smoke tests) — next up