from mcts4py.StatefulSolver import *
from mcts4py.GenericSolver import GenericSolver
from bunkering.BunkeringMDP import *

if __name__ == '__main__':
    mdp = BunkeringMDP(TEU=7000)

    print("Initial state:")

    solver = GenericSolver(
        mdp,
        simulation_depth_limit=100,
        exploration_constant=1.0,
        discount_factor=0.5,
        verbose=True)

    solver.run_search(1000)

    print("\nSearch Tree:")
    solver.display_tree()
