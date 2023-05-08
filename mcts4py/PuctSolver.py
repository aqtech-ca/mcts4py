import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.ProgressiveWideningSolver import *
from mcts4py.MDP import *
import pandas as pd

class PuctSolver(ProgressiveWideningSolver):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 discount_factor: float,
                 exploration_constant: float,
                 verbose: bool = False,
                 max_iteration:int=1000,
                 early_stop: bool = False,
                 early_stop_condition: dict = None,
                 exploration_constant_decay=1):

        self.probabilities = pd.read_csv('lr_models/probabilities.csv', index_col=0)
        self.probabilities.set_index(['port', 'refuel_amount'], inplace=True)
        self.probabilities = self.probabilities.to_dict()['prob']
        super().__init__(mdp, simulation_depth_limit, discount_factor, exploration_constant, verbose, max_iteration, early_stop,
                         early_stop_condition, exploration_constant_decay)

    def select(self, node: StateNode[TState, TAction], iteration_number=None) -> StateNode[TState, TAction]:
        current_node = node

        while True:
            current_node.valid_actions = self.mdp.actions(current_node.state, current_node.n, iteration_number,
                                                          self.max_iteration)
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            explored_actions = current_node.explored_actions()
            # This state has not been fully explored, return it
            if len(current_node.valid_actions) > len(explored_actions):
                return current_node

            # This state has been explored, select best action
            current_node = max(current_node.children, key=lambda c: self.calculate_puct(c))

    # def calculate_puct(self, node:TNode):
    #
    #     # use the loaded model to make predictions
    #
    #     try:
    #         probability = self.probabilities[node.state.port, node.inducing_action.refuel_amount].values[0]
    #     except KeyError:
    #         probability = 0.001
    #     puct_constant = self.exploration_constant*probability
    #     parentN = node.parent.n if node.parent !=None else node.n
    #     return self.calculate_uct_impl(parentN, node.n, node.reward, puct_constant)
    #
    def calculate_puct(self, node: TNode):

        # use the loaded model to make predictions
        try:
            probability = self.probabilities[node.state.port, node.inducing_action.refuel_amount]
        except KeyError:
            probability = 0.001
        puct_constant = self.exploration_constant * probability
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, puct_constant)