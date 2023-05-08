import math
import random
import time

from mcts4py.DPWSolver import *
from mcts4py.MDP import *
import copy
from math import floor
from tqdm import tqdm
import numpy as np
import statistics
import graphviz
import pandas as pd


class PuctWithDPWSolver(DPWSolver):

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
                 dpw_exploration=1,
                 dpw_alpha=1):
        self.probabilities = pd.read_csv('lr_models/probabilities.csv', index_col=0)
        self.probabilities.set_index(['port', 'refuel_amount'], inplace=True)
        super().__init__(mdp, simulation_depth_limit, discount_factor, exploration_constant, verbose, max_iteration,
                         early_stop, early_stop_condition, exploration_constant_decay, dpw_exploration, dpw_alpha)


    def select(self, node: DecisionNode[TRandom, TAction], iteration_number: int = None) -> NewNode[
        TAction, TRandom]:
        current_node = node
        while True:
            if self.mdp.is_terminal(current_node.state):
                return current_node
            if isinstance(current_node, RandomNode):
                next_state = self.mdp.transition(current_node.state, current_node.inducing_action)
                if next_state in current_node.children_states:
                    current_node = current_node.child_with_specific_state(next_state)
                else:
                    current_node = self.create_node(current_node, current_node.inducing_action, next_state)
                # valid_actions = self.mdp.actions(next_state,0)
                # current_node = DecisionNode(current_node, current_node.inducing_action, next_state, valid_actions,
                #                       self.mdp.is_terminal(next_state))
            t = current_node.n + 1
            pos_actions = self.mdp.actions(current_node.state, t, dpw_alpha=self.dpw_alpha,
                                           dpw_exploration=self.dpw_exp)
            for act in pos_actions:
                if act not in current_node.explored_actions():
                    chosen_action = act
                    new_child = self.create_node(current_node, chosen_action, current_node.state)
                    chosen_child = new_child
                    return chosen_child
            try:
                current_node = max(current_node.children, key=lambda c: self.calculate_puct(c))
            except ValueError:
                if self.mdp.is_terminal(current_node.state):
                    return current_node

    def calculate_puct(self, node: TNode):

        # use the loaded model to make predictions

        try:
            probability = self.probabilities[node.state.port, node.inducing_action.refuel_amount]
        except KeyError:
            probability = 0.001
        if node.state.port == 4:
            probability = 1
        puct_constant = self.exploration_constant * probability
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, puct_constant)
