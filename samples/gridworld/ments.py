from mcts4py.GenericSolver import GenericSolver
from mcts4py.Ment_Solver import MentSolver
from samples.gridworld.GridworldMDP import *
# from samples.gridworld.evaluate import *

rewards = [GridworldReward(5, 0, -0.5), GridworldReward(4, 3, 0.5)]
x_size: int = 8
y_size: int = 5
transition_probability: float = 0.8

mdp = GridworldMDP(
    x_size,
    y_size,
    rewards,
    transition_probability,
    starting_location = GridworldState(6, 2, False))

print("Initial state:")
mdp.visualize_state()

solver = MentSolver(
    mdp,
    simulation_depth_limit = 100,
    exploration_constant = 1.0,
    discount_factor = 1.0,
    verbose = False)

solver.run_search(100)
# print("\nSearch Tree:")
solver.print_tree(solver.root())
# solver.display_tree()

# solver.print_node(solver.root())

# num_trials = 1000
#
# # Evaluate performance
# average_reward, success_rate, average_steps = evaluate_performance(mdp, solver, num_trials)
#
# print(f"Average Reward: {average_reward}")
# print(f"Success Rate: {success_rate}")
# print(f"Average Steps to Goal: {average_steps}")

