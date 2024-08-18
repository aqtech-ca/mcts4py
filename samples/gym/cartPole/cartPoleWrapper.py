import time
import gym
import copy
import numpy as np

from typing import Any, List, Optional
from mcts4py.MDP import MDP
from mcts4py.Types import TState


class CartPoleMDP(MDP, gym.Wrapper):
    def __init__(self, time_limit):
        super(CartPoleMDP, self).__init__(gym.make("CartPole-v1"))
        self.initial = self.reset()
        self.time_limit = time_limit
        self.start = time.time()

    def initial_state(self) -> Any:
        return self.initial

    def transition(self, state: Any, action: int) -> Any:
        mdp_copy = copy.deepcopy(self.env)
        new_state, _, _, _, _ = self.env.step(action)
        self.env.close()
        self.env = mdp_copy
        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        mdp_copy = copy.deepcopy(self.env)
        _, reward, _, _, _ = self.env.step(action)
        self.env.close()
        self.env = mdp_copy
        return reward

    def is_terminal(self, state: TState) -> bool:
        return time.time() - self.start > self.time_limit

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:
        return list(range(self.env.action_space.n))