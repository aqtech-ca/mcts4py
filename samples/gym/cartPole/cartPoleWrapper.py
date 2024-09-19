import gym
import copy

from typing import Any, List, Optional, Tuple
from gym.core import ActType, ObsType

from mcts4py.MDP import MDP
from mcts4py.Types import TState


class CartPoleMDP(MDP, gym.Wrapper):
    def __init__(self):
        super(CartPoleMDP, self).__init__(gym.make("CartPole-v1", render_mode='human'))
        self.initial = self.reset()[0]

    def initial_state(self) -> Any:
        return self.initial

    def transition(self, state: Any, action: int) -> Any:

        tmp = gym.make("CartPole-v1", render_mode='rgb_array')
        tmp.reset()
        tmp.state = state
        new_state, reward, done, _, _ = tmp.step(action)

        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        tmp = gym.make("CartPole-v1", render_mode='rgb_array')
        tmp.reset()
        tmp.state = previous_state

        state, reward, done, _, _ = tmp.step(action)
        if done:
            cart_position, cart_velocity, pole_angle, pole_velocity = state
            if abs(cart_position) > 2.4 or abs(pole_angle) > 0.2095:
                reward = -10
        else:
            reward = 10
        return reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        my_reward = self.reward(self.env.state, action)
        state, reward, done, trunc, info = self.env.step(action)
        return state, my_reward, done, trunc, info

    def is_terminal(self, state: TState) -> bool:
        return False

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:
        return [0, 1]