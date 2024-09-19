from traceback import print_tb

import gym
import math
import numpy as np

from typing import Any, List, Optional
from mcts4py.MDP import MDP
from mcts4py.Types import TState


# def simulate_step(state, action):
#     min_position = -1.2
#     max_position = 0.6
#     max_speed = 0.07
#     goal_position = 0.5
#     goal_velocity = 0
#
#     force = 0.001
#     gravity = 0.0025
#
#     position, velocity = state
#     velocity += (action - 1) * force + math.cos(3 * position) * (-gravity)
#     velocity = np.clip(velocity, -max_speed, max_speed)
#     position += velocity
#     position = np.clip(position, min_position, max_position)
#     if position == min_position and velocity < 0:
#         velocity = 0
#
#     terminated = bool(
#         position >= goal_position and velocity >= goal_velocity
#     )
#     reward = -1.0
#
#     state = (position, velocity)
#     return state, reward, terminated

class MountainCarWrapper(MDP, gym.Wrapper):

    def __init__(self):
        super(MountainCarWrapper, self).__init__(gym.make("MountainCar-v0", render_mode='human'))
        self.initial = self.reset()

    def initial_state(self) -> Any:
        return self.initial

    def transition(self, state: Any, action: int) -> Any:
        new_state, reward, terminated = self.simulate_step(state, action)
        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        new_state, reward, terminated = self.simulate_step(previous_state, action)
        return reward

    def is_terminal(self, state: TState) -> bool:
        position = 0.0
        if isinstance(state, tuple):
            state = state[0]  # Extract the state array from the tuple if it includes info

            # If the state is a single float (like numpy.float64), convert it to an array
        if isinstance(state, np.ndarray):
            position, _ = state  # Unpack the position and velocity

        return position >= 0.5

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:
        return list(range(self.env.action_space.n))

    def simulate_step(self, state, action):
        position = 0.0
        velocity = 0.0
        if isinstance(state, tuple):
            state = state[0]  # Extract the state array from the tuple if it includes info

            # If the state is a single float (like numpy.float64), convert it to an array
        if isinstance(state, np.ndarray):
            position, velocity = state  # Unpack the position and velocity

        # position, velocity = state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        state = (position, velocity)
        return state, reward, terminated
