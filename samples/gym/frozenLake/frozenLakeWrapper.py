import gym
import copy

from typing import Any, List, Optional, Tuple
from gym.core import ActType, ObsType

from mcts4py.MDP import MDP
from mcts4py.Types import TState


class FrozenLakeMDP(MDP, gym.Wrapper):

    def __init__(self, is_slippery):
        super(FrozenLakeMDP, self).__init__(gym.make("FrozenLake-v1",is_slippery=is_slippery, render_mode='rgb_array'))
        self.initial = self.reset()

    def initial_state(self) -> Any:
        return self.initial

    def transition(self, state: Any, action: int) -> Any:
        s = self.env.unwrapped.s
        new_state, _, _, _, _ = self.env.step(action)
        self.env.unwrapped.s = s
        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        s = self.env.unwrapped.s
        state, reward, _, _, _ = self.env.step(action)
        if state in [5, 7, 11, 12]:
            reward = -100  # Penalty
        elif state == 15:
            reward = 10  # Goal
        else:
            reward = -0.01
        self.env.unwrapped.s = s
        # print(f"action: {action}, reward: {reward}, state: {self.env.unwrapped}")
        return reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        my_reward = self.reward(self.env.unwrapped.s, action)
        state, reward, done, trunc, obs = self.env.step(action)
        return state, my_reward, done, trunc, obs

    def is_terminal(self, state: TState) -> bool:
        return state == 15

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:

        if isinstance(state, tuple):
            state = state[0]
        valid_actions = []
        if state % 4 != 0: #left
            valid_actions.append(0)
        if state not in [12, 13, 14, 15]: #down
            valid_actions.append(1)
        if state % 4 != 3: #right
            valid_actions.append(2)
        if state > 3: #up
            valid_actions.append(3)

        return valid_actions
