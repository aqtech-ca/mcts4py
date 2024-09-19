from math import trunc

from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.GymMentsSolver import MentSolver
from samples.gym.cartPole.cartPoleWrapper import CartPoleMDP
import copy
import time



exploration_constant= 0.5
env = CartPoleMDP()

# cp = copy.deepcopy(env)
state = env.initial_state()
print(f"Initial State: {state}")

# for i in range(20):
#     state, reward, done, _, _ = env.step(1)
#
#     print(f"step: {i}, reward: {reward}, New State: {state}, done: {done}")
#     if done:
#         cart_position, cart_velocity, pole_angle, pole_velocity = state
#
#         if abs(cart_position) > 2.4:
#             print("The cart moved out of bounds!")
#         elif abs(pole_angle) > 0.2095:
#             print("The pole fell beyond 12 degrees!")
#         else:
#             print("Maximum episode length reached.")
#         break
# time.sleep(5)
# state = env.step(1)
# print("one: ", state)
# time.sleep(5)
# print("cp ", cp.initial_state())
# two = env.step(1)
# print("two: ", two)
# time.sleep(5)
# cp_state = cp.transition(state,1)
# print("cp_state", cp_state)
# print("cpppp: ", cp.env.state)

# two = env.step(1)
# state, reward, done, _, _ = env.step(1)
# print("two: ", state)
#
# print(cp.env.state)
# state2, reward2, done2, _, _ = cp.step(1)
# print(state2)


solver = MentSolver(
    mdp=env,
    simulation_depth_limit=100,
    exploration_constant=exploration_constant,
    discount_factor=0.8,
    verbose=False
)

solver.run_search(5)
solver.display_tree(5)
# print(solver.mdp.env.state)
# action = solver.do_best_action(solver.root())
# time.sleep(1)
# solver.mdp.step(action)
# print(solver.mdp.env.state)



