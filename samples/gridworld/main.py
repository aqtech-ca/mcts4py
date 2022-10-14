from mcts4py.DeterministicSolver import *
from mcts4py.StochasticSolver import StochasticSolver
from samples.gridworld.GridworldMDP import *

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

solver = StochasticSolver(
    mdp,
    simulation_depth_limit = 100,
    exploration_constant = 1.0,
    discount_factor = 0.5,
    verbose = False)

solver.run_search(1000)

print("\nSearch Tree:")
solver.display_tree()


