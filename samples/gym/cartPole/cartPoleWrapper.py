import gym
import copy
import math
import numpy as np

from typing import Any, List, Optional, Tuple

from gym.core import ActType, ObsType

from mcts4py.MDP import MDP
from mcts4py.Types import TState


def simulate_step(state, action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5
    polemass_length = masspole * length
    force_mag = 10.0
    tau = 0.02
    kinematics_integrator = "euler"

    x, x_dot, theta, theta_dot = state
    force = force_mag if action == 1 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if kinematics_integrator == "euler":
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
    else:
        x_dot = x_dot + tau * xacc
        x = x + tau * x_dot
        theta_dot = theta_dot + tau * thetaacc
        theta = theta + tau * theta_dot

    new_state = (x, x_dot, theta, theta_dot)
    done = (theta < -np.pi / 4 or theta > np.pi / 4 or x < -2.4 or x > 2.4)

    reward = 0 if done else 1

    return new_state, reward, done


class CartPoleMDP(MDP, gym.Wrapper):
    def __init__(self):
        super(CartPoleMDP, self).__init__(gym.make("CartPole-v1", render_mode='human'))
        self.initial = self.reset()[0]

    def initial_state(self) -> Any:
        return self.initial

    def transition(self, state: Any, action: int) -> Any:
        new_state, reward, done = simulate_step(state, action)
        return new_state

    def reward(self, previous_state: Optional[Any], action: Optional[int]) -> float:
        new_state, reward, done = simulate_step(previous_state, action)
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
