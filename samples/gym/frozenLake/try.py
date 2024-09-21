from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.frozenLake.frozenLakeWrapper import FrozenLakeMDP



if __name__ == "__main__":
    is_slippery = False
    exploration_constant= 0.5
    env = FrozenLakeMDP(is_slippery, map_name='8x8')

# solver = GenericSolver(
#     mdp=env,
#     simulation_depth_limit=100,
#     exploration_constant=exploration_constant,
#     discount_factor=0.8,
#     verbose=False
# )
#
# solver.run_search(5)
# solver.display_tree(5)
# state = env.initial_state()
# print(f"Initial State: {state}")

# state, reward, done, _, _ = env.step(2)
# state, reward, done, _, _ = env.step(2)
# state, reward, done, _, _ = env.step(2)
# print(f"reward: {reward}, New State: {state}, done: {done}")

#0 left
#1 down
#2 right
#3 up