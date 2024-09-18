from samples.gym.cartPole.cartPoleWrapper import CartPoleMDP
import copy
import time


env = CartPoleMDP(5)

cp = copy.deepcopy(env)
state = env.initial_state()
print(f"Initial State: {state}")

print(env.spec)

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

state, reward, done, _, _ = env.step(1)
print(state)

print(cp.env.state)
time.sleep(5)
state2, reward2, done2, _, _ = cp.step(1)
print(state2)



