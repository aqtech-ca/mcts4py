import gym
import math
import numpy as np

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
        new_state, reward, done = self.simulate_step(state, action)
        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        if previous_state is None:
            return 0
        new_state, reward, done = self.simulate_step(previous_state, action)
        return reward

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        my_reward = self.reward(self.env.state, action)
        state, reward, done, trunc, info = self.env.step(action)
        return state, my_reward, done, trunc, info

    def is_terminal(self, state: TState) -> bool:
        cart_position, cart_velocity, pole_angle, pole_velocity = state
        return abs(cart_position) > 2.4 or abs(pole_angle) > 0.2095

    def actions(self, state: Any, state_visit=None, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action=False) -> List[int]:
        return [0, 1]

    def simulate_step(self, state, action):

        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else - self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        new_state = (x, x_dot, theta, theta_dot)
        done = (theta < -np.pi / 4 or theta > np.pi / 4 or x < -2.4 or x > 2.4)

        reward = 0 if done else 1

        return new_state, reward, done
