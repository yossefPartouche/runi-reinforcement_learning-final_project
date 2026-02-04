from __future__ import annotations

import random
import numpy as np

from gymnasium import spaces
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Ball, Wall
from minigrid.minigrid_env import MiniGridEnv as BaseMiniGridEnv

# =============================================================================
# ENVIRONMENT 2: KEY-DOOR WITH BALL PICKUP
# =============================================================================
class KeyDoorBallEnv(BaseMiniGridEnv):
    """
    Grid environment with two rooms separated by a locked door.

    Task sequence:
        1. Pick up key -> 2. Unlock door -> 3. Pick up ball -> 4. Reach goal

    Actions:
        0: Turn Left
        1: Turn Right
        2: Move Forward
        3: Pick Up
        4: Toggle (open/close door)
    """
    def __init__(
        self,
        size=10,
        max_steps=1000,
        render_mode="rgb_array",
        partition_col=3,
        require_ball_pickup=True,
        preprocess=None,
        **kwargs,
    ):
        self.agent_start_pos = (1, 1)
        self.agent_start_dir = 0
        self.partition_col = partition_col
        self.walls_init = []
        self.inventory = []
        self.require_ball_pickup = require_ball_pickup
        self.preprocess = preprocess if preprocess is not None else lambda x: x

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            render_mode=render_mode,
            highlight=False,
            **kwargs,
        )

        # 5 actions: left, right, forward, pickup, toggle
        self.action_space = spaces.Discrete(5)

        # ╔═════════════════════════════════════════════════════════════════════╗
        # ║  ✅ STUDENT TODO: Update observation_space to match preprocessing   ║
        # ╚═════════════════════════════════════════════════════════════════════╝
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1, 84, 84),
            dtype=np.float32
        )

        # State tracking for reward shaping (you can use these in your reward logic)
        self.prev_key = False
        self.prev_door = False
        self.prev_ball = False

        # This will be used once the first two subtasks are completed
        self.is_door_opened_step = None
        self.has_crossed_door = False
        
        # for analysis 
        self.action_counts = {0: 0, 1:0, 2: 0, 3:0, 4:0}
        self.action_history = []
        self.prev_action = None
        # ╔═════════════════════════════════════════════════════════════════════╗
        # ║                     END OF EDITABLE SECTION                         ║
        # ╚═════════════════════════════════════════════════════════════════════╝

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║  ⛔ DO NOT MODIFY: Core environment methods below                       ║
    # ╚═════════════════════════════════════════════════════════════════════════╝

    @staticmethod
    def _gen_mission():
        return "Pick up the key to open the door, pick up the ball, then reach the goal"

    def _get_obs(self, obs=None):
        """Returns the current observation after applying preprocessing."""
        obs = self.get_frame(highlight=False, tile_size=32)
        return self.preprocess(obs)

    def reset(self, *, seed=None, options=None):
        # Reset state tracking
        self.prev_key = False
        self.prev_door = False
        self.prev_ball = False
        self.inventory = []

        self.door_opened_step = None
        self.has_crossed_door = False

        self.action_history = []
        for key in self.action_counts:
            self.action_counts[key] = 0
        self.prev_action = None
        self.turnback_count = 0

        # Call parent reset, which internally calls _gen_grid()
        obs, info = super().reset(seed=seed, options=options)

        return self._get_obs(obs), info

    def _gen_grid(self, width, height):
        """Generate grid: walls, partition, door, key, ball, goal, agent."""
        # Grid with outer walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Partition wall
        self.walls_init = [(self.partition_col, i) for i in range(height)]
        for col, row in self.walls_init:
            if 0 <= col < width and 0 <= row < height:
                self.grid.set(col, row, Wall())

        # Key in left room
        self.key_pos = (
            random.choice(range(1, self.partition_col)),
            random.choice(range(2, height - 1))
        )
        self.grid.set(self.key_pos[0], self.key_pos[1], Key(COLOR_NAMES[0]))

        # Door in partition
        door_y = random.choice(range(1, height - 1))
        self.door_pos = (self.partition_col, door_y)
        self.env_door = Door(COLOR_NAMES[0], is_locked=True)
        self.grid.set(self.door_pos[0], self.door_pos[1], self.env_door)

        # Goal
        self.goal_pos = (8, 8)
        self.put_obj(Goal(), self.goal_pos[0], self.goal_pos[1])

        # Ball in right room (if required)
        if self.require_ball_pickup:
            right_x = range(self.partition_col + 2, width - 2)
            right_y = range(1, height - 1)
            while True:
                ball_x = random.choice(list(right_x))
                ball_y = random.choice(list(right_y))
                self.ball_pos = (ball_x, ball_y)
                if self.ball_pos != self.goal_pos:
                    break

            self.grid.set(ball_x, ball_y, Ball(COLOR_NAMES[1]))

        # Agent
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

    def try_pickup_ball(self):
        """Pick up ball in front and add to inventory."""
        obj = self.grid.get(self.front_pos[0], self.front_pos[1])
        if isinstance(obj, Ball):
            self.grid.set(self.front_pos[0], self.front_pos[1], None)
            self.inventory.append(obj)

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║  ✅ STUDENT TODO: Modify reward shaping below                           ║
    # ╚═════════════════════════════════════════════════════════════════════════╝
    def step(self, action):
        """
        Step function with sparse reward.

        Available helper methods for reward shaping:
            - self.is_carrying_key()  : Returns True if agent has the key
            - self.is_carrying_ball() : Returns True if agent has the ball
            - self.is_door_open()     : Returns True if door is open
            - self.prev_key           : Key status before this step
            - self.prev_door          : Door status before this step
            - self.prev_ball          : Ball status before this step

        Current reward scheme (sparse):
            - +1.0 for reaching goal with ball
            - 0.0 otherwise

        💡 You can add reward shaping here
        """
        original_action = action
        # Map action 4 to toggle (internal MiniGrid uses 5 for toggle)
        if action == 4:
            action = 5

        # Track previous state for reward shaping
        self.prev_key = self.is_carrying_key()
        self.prev_door = self.is_door_open()
        self.prev_ball = self.is_carrying_ball()
        prev_in_right_room = self.agent_pos[0] > self.partition_col

        if not hasattr(self, 'prev_action'):
            self.prev_action = None

        # Handle ball pickup
        if action == 3:
            self.try_pickup_ball()

        # Standard step
        obs, reward, terminated, truncated, info = super().step(action)

        self.action_counts[original_action] +=1 
        self.action_history.append(original_action)

        # Goal only counts if ball is picked up (when required)
        terminated = terminated and (not self.require_ball_pickup or self.is_carrying_ball())

        reward = 0.0

        # ANALYSIS BASED: Reduce wastful turning
        if original_action in [0, 1]:
            
            if self.prev_action is not None:
                if (original_action == 0 and self.prev_action == 1) or \
               (original_action == 1 and self.prev_action == 0):
                    
                    self.turnback_count += 1

                    if self.turnback_count == 1:
                        penalty = -0.10  
                    elif self.turnback_count == 2:
                        penalty = -0.15
                    elif self.turnback_count >= 3:
                        penalty = -0.25 
                    else:
                        penalty = -0.10

                    reward += penalty
                    if self.step_count % 20 == 0:  # Occasional logging
                        print(f"   Turn-back detected at step {self.step_count} (-0.05)")
                else:
                    # Reset counter if not turn-backing
                    self.turnback_count = max(0, self.turnback_count - 1)

        if original_action == 2:
            reward += 0.005
            self.turnback_count = max(0, self.turnback_count - 1)

        if not self.prev_key and self.is_carrying_key():
            reward += 0.3
            print(f"Step {self.step_count}: Key picked up! (+0.3)")
            self.turnback_count = 0 

        if self.prev_key and not self.prev_door and self.is_door_open():
            reward += 0.3
            self.door_opened_step = self.step_count
            print(f" Step {self.step_count}: Door opened! (+0.3)")
            print(f"   Door crossing penalty now ACTIVE!")
            self.turnback_count = 0 
        
        current_in_right_room = self.agent_pos[0] > self.partition_col

        # Agent Cross over "second room"
        if not prev_in_right_room and current_in_right_room:
            self.has_crossed_door = True
            reward += 0.5
            print(f" Step {self.step_count}: Crossed to right room! (+0.5)")
            # Current test 
            print(f"   Door crossing penalty DEACTIVATED!")
            self.turnback_count = 0

        if (self.door_opened_step is not None and 
            not self.has_crossed_door and not current_in_right_room):
            steps_since_door = self.step_count - self.door_opened_step

            if steps_since_door <= 5:
                penalty = -0.08
            elif steps_since_door <= 15:
                penalty = -0.20
            else:
                penalty =-0.40
            
            reward += penalty

            # user update on the stuck in the mud state 
            if steps_since_door % 3 == 0:
                print(f"   Step {self.step_count}: Still in left room "
                      f"({steps_since_door} steps since door opened) "
                      f"Penalty: {penalty:.3f}")
                
        if not self.prev_ball and self.is_carrying_ball():
            reward += 0.3
            print(f" Step {self.step_count}: Ball picked up! (+0.3)")

        # ----- REWARD SHAPING: EDIT BELOW THIS LINE -----
        if terminated:
            reward += 1.0
            print(f"Step {self.step_count}: Goal reached! (+1.0)")

            #  episode summary
            if self.door_opened_step is not None:
                if self.has_crossed_door:
                    steps_to_cross = self.step_count - self.door_opened_step
                    print(f"\n Episode Summary:")
                    print(f"   Door opened:  Step {self.door_opened_step}")
                    print(f"   Door crossed: Step {self.door_opened_step + steps_to_cross}")
                    print(f"   Time to cross: {steps_to_cross} steps")
                    print(f"   Total steps:  {self.step_count}\n")
                else:
                    print(f"\n  Episode ended without crossing door!")
        
        reward -= 0.002

        self.prev_action = original_action
        # ----- REWARD SHAPING: EDIT ABOVE THIS LINE -----

        return self._get_obs(obs), reward, terminated, truncated, info
    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║                     END OF EDITABLE SECTION                             ║
    # ╚═════════════════════════════════════════════════════════════════════════╝

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║  ⛔ DO NOT MODIFY: State getter methods (use these in reward shaping)   ║
    # ╚═════════════════════════════════════════════════════════════════════════╝

    def is_carrying_key(self):
        """Check if agent has key (in hand or inventory)."""
        key_in_hand = self.carrying and isinstance(self.carrying, Key)
        key_in_inventory = any(isinstance(item, Key) for item in self.inventory)
        return key_in_hand or key_in_inventory

    def is_carrying_ball(self):
        """Check if agent has ball (in hand or inventory)."""
        ball_in_hand = self.carrying and isinstance(self.carrying, Ball)
        ball_in_inventory = any(isinstance(item, Ball) for item in self.inventory)
        return ball_in_hand or ball_in_inventory

    def is_door_open(self):
        """Returns True if the door is open."""
        if hasattr(self, 'env_door'):
            return self.env_door.is_open
        return False