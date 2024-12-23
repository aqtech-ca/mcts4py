import gym
import copy
import time
import math
import numpy as np

from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.GymMentsSolver import MENTSSolverV1
from samples.gym.cartPole.cartPoleWrapper import CartPoleMDP


def simulate_step(state, action):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  # actually half the pole's length
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
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    if kinematics_integrator == "euler":
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
    else:  # semi-implicit euler
        x_dot = x_dot + tau * xacc
        x = x + tau * x_dot
        theta_dot = theta_dot + tau * thetaacc
        theta = theta + tau * theta_dot

    new_state = (x, x_dot, theta, theta_dot)
    done = (theta < -np.pi/4 or theta > np.pi/4 or x < -2.4 or x > 2.4)

    reward = 1 if done else 0

    return new_state, reward, done

exploration_constant= 0.5
env = CartPoleMDP()

# cp = copy.deepcopy(env)
state = env.initial_state()
print(f"Initial State: {state}")

state, reward, done, _, _  = env.step(1)
print("one: ", state)
two, reward, done, _, _  = env.step(1)
print("two: ", two)
three, reward, done, _, _  = env.step(1)
print("three: ", three)
four = env.step(1)
four, reward, done, _, _ = env.step(1)
print("four: ", four)

# tmp = gym.make("CartPole-v1", render_mode='rgb_array')
# tmp.reset()
# tmp.env.state = three
#
# tmp_state, reward, done, _, _  = tmp.step(1)

state = np.array(three)  # Example state
print("state: ", state)
action = 1  # Example action (right)
new_state, reward, done = simulate_step(state, action)
print(f"New State: {new_state}, Reward: {reward}, Done: {done}")


# solver = MENTSSolverV1(
#     mdp=env,
#     simulation_depth_limit=100,
#     exploration_constant=exploration_constant,
#     discount_factor=0.8,
#     verbose=False
# )
#
# solver.run_search(5)
# solver.display_tree(5)
# print(solver.mdp.env.state)
# action = solver.do_best_action(solver.root())
# time.sleep(1)
# solver.mdp.step(action)
# print(solver.mdp.env.state)



