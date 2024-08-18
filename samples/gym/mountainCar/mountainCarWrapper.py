import gym
import copy

from typing import Any, List, Optional
from mcts4py.MDP import MDP
from mcts4py.Types import TState


class MountainCarWrapper(MDP, gym.Wrapper):

    def __init__(self):
        super(MountainCarWrapper, self).__init__(gym.make("MountainCar-v0"))
        self.initial = self.reset()

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
        if isinstance(state,tuple):
            state = state[0]
        return state[0] >= 0.5

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:
        return list(range(self.env.action_space.n))
