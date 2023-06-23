import pandas as pd

from mcts4py.StatefulSolver import *
from mcts4py.MDP import *


class PuctSolver(StatefulSolver):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 discount_factor: float,
                 exploration_constant: float,
                 verbose: bool = False,
                 max_iteration: int = 1000,
                 early_stop: bool = False,
                 early_stop_condition: dict = None,
                 exploration_constant_decay=1,
                 probabilities: pd.DataFrame =None):

        super().__init__(mdp, simulation_depth_limit, discount_factor, exploration_constant, verbose, max_iteration, early_stop,
                         early_stop_condition, exploration_constant_decay)
        self.probabilities = probabilities


    def select(self, node: StateNode[TState, TAction]) -> StateNode[TState, TAction]:
        current_node = node

        while True:
            current_node.valid_actions = self.mdp.actions(current_node, current_node.n)
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            explored_actions = current_node.explored_actions()
            # This state has not been fully explored, return it
            if len(current_node.valid_actions) > len(explored_actions):
                return current_node

            # This state has been explored, select best action
            current_node = max(current_node.get_children(), key=lambda c: self.calculate_puct(c))

    def calculate_puct(self, node):
        prob = self.probabilities.loc[node]
        puct_constant = self.exploration_constant * prob
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, puct_constant)
