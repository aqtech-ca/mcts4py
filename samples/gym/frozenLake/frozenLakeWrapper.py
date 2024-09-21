import gym

from typing import Any, List, Optional, Tuple
from gym.core import ActType, ObsType
from gym.envs.toy_text.utils import categorical_sample

from mcts4py.MDP import MDP
from mcts4py.Types import TState


class FrozenLakeMDP(MDP, gym.Wrapper):

    def __init__(self, is_slippery, map_name='4x4'):
        super(FrozenLakeMDP, self).__init__(gym.make("FrozenLake-v1", is_slippery=is_slippery, render_mode='human', map_name=map_name))
        self.initial = self.reset()
        self.map_name = map_name

    def initial_state(self) -> Any:
        return self.initial

    def transition(self, state: Any, action: int) -> Any:
        new_state, reward, trunc = self.simulate_action(state,action)
        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        reward = 0
        new_state = self.transition(previous_state, action)
        if self.map_name == '4x4':
            if new_state in [5, 7, 11, 12]:
                reward = -10  # Penalty
            elif new_state == 15:
                reward = 10  # Goal
        elif self.map_name == '8x8':
            if new_state in [19, 29, 35, 41, 42, 46, 49, 52, 54, 59]:
                reward = -10  # Penalty
            elif new_state == 63:
                reward = 10  # Goal

        return reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        my_reward = self.reward(self.env.unwrapped.s, action)
        state, reward, done, trunc, info = self.env.step(action)
        return state, my_reward, done, trunc, info

    def is_terminal(self, state: TState) -> bool:
        if self.map_name == '4x4':
            return state in [5, 7, 11, 12, 15]
        elif self.map_name == '8x8':
            return state in [19, 29, 35, 41, 42, 46, 49, 52, 54, 59, 63]
        return False

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:

        if isinstance(state, tuple):
            state = state[0]

        valid_actions = []
        if self.map_name == '4x4':
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

        elif self.map_name == '8x8':
            if state % 8 != 0:
                # left
                valid_actions.append(0)
            if state < 56:
                # down
                valid_actions.append(1)
            if state % 8 != 7:
                # right
                valid_actions.append(2)
            if state > 7:
                # up
                valid_actions.append(3)

        return valid_actions

    def simulate_action(self, state, action):
        if isinstance(state, tuple):
            state = state[0]
        transitions = self.P[state][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        p, new_state, r, t = transitions[i]

        return int(new_state), r, t
