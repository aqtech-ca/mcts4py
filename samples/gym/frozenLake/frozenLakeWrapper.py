import gym
import copy

from typing import Any, List, Optional, Tuple
from gym.core import ActType, ObsType

from mcts4py.MDP import MDP
from mcts4py.Types import TState


class FrozenLakeMDP(MDP, gym.Wrapper):

    def __init__(self, is_slippery):
        super(FrozenLakeMDP, self).__init__(gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode='human'))
        self.initial = self.reset()

    def initial_state(self) -> Any:
        return self.initial

    def transition(self, state: Any, action: int) -> Any:
        new_state = 0
        if isinstance(state, tuple):
            state = state[0]

        if action == 0:
            new_state = state - 1
        elif action == 1:
            new_state = state + 4
        elif action == 2:
            new_state = state + 1
        elif action == 3:
            new_state = state - 4
        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        new_state = self.transition(previous_state, action)
        if new_state in [5, 7, 11, 12]:
            reward = -10  # Penalty
        elif new_state in [15]:
            reward = 100  # Goal
        else:
            reward = 0
        return reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        my_reward = self.reward(self.env.unwrapped.s, action)
        state, reward, done, trunc, info = self.env.step(action)
        return state, my_reward, done, trunc, info

    def is_terminal(self, state: TState) -> bool:
        return state == 15

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:

        if isinstance(state, tuple):
            state = state[0]
        valid_actions = []
        if state % 4 != 0:
            # left
            valid_actions.append(0)
        if state not in [12, 13, 14, 15]:
            # down
            valid_actions.append(1)
        if state % 4 != 3:
            # right
            valid_actions.append(2)
        if state > 3:
            # up
            valid_actions.append(3)

        return valid_actions
