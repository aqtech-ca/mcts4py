import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.ProgressiveWideningSolver import *
from mcts4py.MDP import *
import pandas as pd
from joblib import load
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
                 exploration_constant_decay=1,
                 models: list =None,
                 stds: list = None):

        self.models = models
        self.stds = stds
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


        port = node.state.port
        std = self.stds[port]
        model = self.models[port]
        value = [node.state.price, node.state.fuel_amount]
        prediction = model.predict(np.array(value).reshape(1, -1))
        ref_am = node.inducing_action.refuel_amount
        if ref_am == 0:
            prob = 0.5
        elif prediction * (1 - std * 0.5) < node.inducing_action.refuel_amount <= prediction * (1 + std * 0.5):
            prob = 0.4
        elif prediction * (1 - std) < node.inducing_action.refuel_amount <= prediction * (1 + std):
            prob = 0.2
        elif prediction * (1 - std * 1.5) < node.inducing_action.refuel_amount <= prediction * (1 + std * 1.5):
            prob = 0.1
        elif prediction * (1 - std * 3) < node.inducing_action.refuel_amount <= prediction * (1 + std * 3):
            prob = 0.05
        else:
            prob = 0.01
        puct_constant = self.exploration_constant * prob
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, puct_constant)