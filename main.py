from mcts4py.NodeClasses import *
from mdp.simple_mdp import *
from mcts4py.StatefulSolver import *

state = "hehe" # try state representation as a string
s = StateNode(parent = None, state = state, inducing_action = "1", valid_actions = ["1", "2"] )


simpMDP = SimpleMDP(initial_state = "b")

solver = StatefulSolver(simpMDP, verbose = True, exploration_constant = 0.9)

solver.runTreeSearch(99)
solver.displayTree(True)

print(str(solver.extractOptimalAction()))