import gym
import copy

from typing import Any, List, Optional, Tuple
from gym.core import ActType, ObsType

from mcts4py.MDP import MDP
from mcts4py.Types import TState


class CartPoleMDP(MDP, gym.Wrapper):
    def __init__(self, goal_step):
        super(CartPoleMDP, self).__init__(gym.make("CartPole-v1"))
        self.initial = self.reset()[0]
        # self._max_episode_steps = goal_step

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
        state, reward, done, _, _ = self.env.step(action)
        if done:
            cart_position, cart_velocity, pole_angle, pole_velocity = state
            if abs(cart_position) > 2.4 or abs(pole_angle) > 0.2095:
                reward = -10
        else:
            reward = 10
        self.env.close()
        self.env = mdp_copy
        return reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        my_reward = self.reward(self.env.state, action)
        state, reward, done, trunc, info = self.env.step(action)
        return state, my_reward, done, trunc, info

    def is_terminal(self, state: TState) -> bool:
        return False

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:
        return list(range(self.env.action_space.n))