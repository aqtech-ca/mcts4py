from mcts4py.NodeClasses import *
from mdp.simple_mdp import *
from mcts4py.DeterministicSolver import *

state = "hehe" # try state representation as a string
s = StateNode(parent = None, state = state, inducing_action = "1", valid_actions = ["1", "2"] )


simpMDP = SimpleMDP(initial_state = "b")

solver = DeterministicSolver(simpMDP, verbose = True, exploration_constant = 0.9)

solver.run_tree_search(99)
solver.display_tree(True)

print(str(solver.extract_optimal_action()))