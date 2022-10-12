# from mcts4py.NodeClasses import *
# from mdp.simple_mdp import *
# from mcts4py.StatefulSolver import *

# state = "hehe" # try state representation as a string
# s = StateNode(parent = None, state = state, inducing_action = "1", valid_actions = ["1", "2"] )


# simpMDP = SimpleMDP(initial_state = "b")

# solver = StatefulSolver(simpMDP, verbose = True, exploration_constant = 0.9)

# solver.runTreeSearch(99)
# solver.displayTree(True)

# print(str(solver.extractOptimalAction()))

from enum import Enum
from mcts4py.Types import *
from mcts4py.Nodes import *

class GridWorldAction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class GridWorldState:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"[{self.x}, {self.y}]"

root_node: StateNode[GridWorldAction, GridWorldState] = \
    StateNode(
        None,
        None,
        GridWorldState(0, 0),
        [GridWorldAction.UP, GridWorldAction.DOWN, GridWorldAction.LEFT, GridWorldAction.RIGHT],
        False)

child_node: StateNode[GridWorldAction, GridWorldState] = \
    StateNode(
        root_node,
        GridWorldAction.DOWN,
        GridWorldState(1, 0),
        [GridWorldAction.UP, GridWorldAction.DOWN, GridWorldAction.LEFT, GridWorldAction.RIGHT],
        True)

root_node.add_child(child_node)