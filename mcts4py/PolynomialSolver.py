import math
import random
import time

from mcts4py.Solver import *
from mcts4py.MDP import *
import copy
from math import floor
from tqdm import tqdm
import numpy as np
import statistics
import graphviz


class PuctSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 max_iteration: int,
                 early_stop: bool,
                 max_depth: int,
                 early_stop_condition: dict = None,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        super().__init__(exploration_constant, verbose, max_iteration, early_stop, early_stop_condition)
        self.__root_node = self.create_node(None, None, mdp.initial_state())
        self.max_depth = max_depth

    def select(self, node: NewNode[TRandom, TAction], iteration_number: int = None) -> NewNode[TAction, TRandom]:
        current_node = node

        # If the node is terminal, return it
        if current_node.is_terminal or current_node.n == 0:
            return current_node
        while True:
            if isinstance(current_node, DecisionNode):
                a = floor(10 * pow(current_node.n, self._get_alpha(current_node)))
                b = floor(10 * pow(current_node.n - 1, self._get_alpha(current_node)))
                if a > b or len(current_node.children) <= 5:
                    return current_node  # decision node
                else:
                    try:
                        current_node = max(current_node.children, key=lambda c: self.calculate_puct(c))  # random node
                    except ValueError:
                        return current_node

            if isinstance(current_node, RandomNode):
                a = floor(pow(current_node.n, self._get_alpha(current_node)))
                b = floor(pow(current_node.n - 1, self._get_alpha(current_node)))
                if a == b:
                    current_node = min(current_node.children, key=lambda c: c.n)  # random node
                else:
                    return current_node  # random node

    def expand(self, node: NewNode[TState, TAction], iteration_number: int = None) -> NewNode[TState, TAction]:
        # If the node is terminal, return it
        if node.is_terminal:
            return node

        if isinstance(node, DecisionNode):
            explored_actions = node.explored_actions()
            node.valid_actions = self.mdp.widening_actions(node.state, node.n, iteration_number, self.max_iteration)
            unexplored_actions = [a for a in node.valid_actions if a not in explored_actions]
            if len(unexplored_actions) == 0:
                return node

            action_taken = random.choice(unexplored_actions)
            # new_state = self.mdp.transition(node.state, action_taken)
            return self.create_node(node, action_taken, node.state, node.n, iteration_number)

        elif isinstance(node, RandomNode):
            new_state = self.mdp.transition(node.state, node.inducing_action)
            for ch in node.children:
                if ch.state == new_state:
                    return ch
            return self.create_node(node, node.inducing_action, new_state, node.parent.n, iteration_number)

    def simulate(self, node: NewNode[TRandom, TAction], depth=0, iteration_number=None) -> float:
        reward = 0
        if depth == 0:
            current_node = copy.deepcopy(node)
            if isinstance(current_node, RandomNode):
                reward += self.mdp.reward(node.state, node.inducing_action)
            else:
                reward += self.mdp.reward(node.parent.state, node.parent.inducing_action)
        if node.is_terminal:
            return reward
        if isinstance(node, DecisionNode):
            valid_actions = self.mdp.widening_actions(node.state, node.n, iteration_number, self.max_iteration)
            random_action = random.choice(valid_actions)
            reward += self.mdp.reward(node.state, random_action) * (self.discount_factor ** depth)
            if node.is_terminal or depth >= self.simulation_depth_limit:
                return reward
            else:
                next_node = RandomNode(None, random_action, node.state, self.mdp.is_terminal(node.state))
                reward += self.simulate(next_node, depth + 1, iteration_number)
                return reward
        elif isinstance(node, RandomNode):
            next_state = self.mdp.transition(node.state, node.inducing_action)
            valid_actions = self.mdp.widening_actions(next_state, node.n, iteration_number, self.max_iteration)
            next_node = DecisionNode(None, node.inducing_action, next_state, valid_actions,
                                     self.mdp.is_terminal(next_state))
            reward += self.simulate(next_node, depth + 1, iteration_number)
            return reward

    # Utilities
    def run_search_iteration(self, iteration_number=0):
        # Selection
        root_node = self.root()
        # if iteration_number %10 ==1:
        #     print('------')
        #     print('NUMBER OF NODES: ', len(root_node.children))
        #     print('MAX: ', max([i.n for i in root_node.children]))
        #     print('MIN: ', min([i.n for i in root_node.children]))
        #     print('MODE: ', statistics.mode([i.n for i in root_node.children]))
        #     print('MEDIAN: ', statistics.median([i.n for i in root_node.children]))
        #     print('------')

        best = self.select(root_node)

        if self.verbose:
            print("Selected:")
            self.display_node(best)

        # Expansion
        expanded = self.expand(best, iteration_number)

        if self.verbose:
            print("Expanded to:")
            self.display_node(expanded)

        # Simulation
        start = time.time()
        simulated_reward = self.simulate(expanded, iteration_number=iteration_number)
        # if iteration_number % 100 == 0:
        #   print((time.time() -start)*100)

        if self.verbose:
            print(f"Simulated reward: {simulated_reward}")

        # Backpropagation
        self.backpropagate(expanded, simulated_reward)
        return simulated_reward

    def create_node(self, parent: Optional[DecisionNode[TState, TAction]], inducing_action: Optional[TAction],
                    state: TState, number_of_visits=0, iteration_number=0) -> NewNode[TRandom, TAction]:

        if isinstance(parent, DecisionNode):
            is_terminal = self.mdp.is_terminal(state)
            node = RandomNode(parent, inducing_action, state, is_terminal)
        else:  # isinstance(state, DecisionNode):
            valid_actions = self.mdp.widening_actions(state, number_of_visits, iteration_number, self.max_iteration)
            is_terminal = self.mdp.is_terminal(state)
            node = DecisionNode(parent, inducing_action, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(node)
            node.depth = parent.depth + 0.5
        else:
            node.depth = 0

        return node

    def save_tree_impl(self, depth_limit, node, indent: str, path: str):
        if node == None or node.depth > depth_limit:
            return
        with open(path, 'a') as f:
            f.write(
                f"{indent} {str(node)} (n: {node.n}, reward: {node.reward / node.n:.3f}, UCT: "
                f"{self.calculate_uct(node):.3f})")
            f.write('\n')

        children = node.children

        if len(children) == 0:
            return

        children.sort(key=lambda c: c.reward / c.n, reverse=True)

        for child in children[:-1]:
            self.save_tree_impl(depth_limit, child, self.generate_indentation(indent) + " ├", path)
        self.save_tree_impl(depth_limit, children[-1], self.generate_indentation(indent) + " └", path)

    def save_tree(self, depth_limit=10, indent="", path=f'runs/', simulation_number=0, prices='None', run_time='None'):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + f'/tree{simulation_number}.txt'
        node = self.root()
        with open(path, 'a') as f:
            f.write(f"Prices:{prices}")
            f.write('\n')
            f.write(f"Run Time:{run_time}")
            f.write('\n')
        self.save_tree_impl(depth_limit, node, indent, path)

    def calculate_puct(self, node):
        return node.reward + math.sqrt(math.pow(node.parent.n, self._get_epsilon(node.parent)) / node.n)

    def _get_alpha(self, node):
        dmax = self.max_depth
        if isinstance(node, DecisionNode):
            if not node.depth <= dmax - 1:
                raise ('Wrong Depth')
            alpha = 1 / (10 * (dmax - node.depth) - 3)
        else:
            if node.depth <= dmax - 1.5:
                alpha = 1 / (10 * (dmax - node.depth) - 3)
            elif node.depth == dmax - 0.5:
                alpha = 1
            else:
                raise ('Wrong Depth')

        return alpha

    def _get_epsilon(self, node):
        dmax = self.max_depth
        p = 8
        if isinstance(node, DecisionNode):
            if not node.depth <= dmax - 1:
                raise ('Wrong Depth')
            epsilon = (1 - (3 / (10 * (dmax - node.depth)))) * (1 / 2 * p)
        else:
            raise ('Random Node Called Epsilon which wasn\'t supposed to happen ')

        return epsilon

    def extract_optimal_action(self) -> Optional[TAction]:
        max_rn = max(self.root().children, key=lambda c: c.reward / c.n)
        return max(max_rn.children, key=lambda c: c.reward / c.n)
