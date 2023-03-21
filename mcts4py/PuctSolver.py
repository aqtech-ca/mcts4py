import math
import random
from mcts4py.Solver import *
from mcts4py.MDP import *
import copy
from math import floor
from tqdm import tqdm
import numpy as np

class PuctSolver(MCTSSolver[TAction, NewNode[TRandom, TAction]], Generic[TDecisionNode, TRandomNode, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 max_iteration: int,
                 early_stop: bool,
                 early_stop_condition: dict = None,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        super().__init__(exploration_constant, verbose, max_iteration, early_stop, early_stop_condition)
        self.__root_node = self.create_node(None, None, mdp.initial_state())

    def root(self) -> NewNode[TAction, TRandom]:
        return self.__root_node

    def select(self, node: NewNode[TRandom, TAction], iteration_number: int = None) -> NewNode[TAction, TRandom]:
        current_node = node

        # If the node is terminal, return it
        if current_node.is_terminal:
            return current_node
        if current_node.n == 0:  # The power function gets error when n=0
            return current_node
        while True:
            if isinstance(current_node, DecisionNode):
                if floor(pow(current_node.n, self._get_alpha(current_node))) > floor(
                        pow(current_node.n - 1, self._get_alpha(current_node))):
                    return current_node  # decision node
                else:
                    current_node = max(current_node.get_children(), key=lambda c: self.calculate_puct(c))  # random node

            if isinstance(current_node, RandomNode):
                if floor(pow(current_node.n, self._get_alpha(current_node)) == floor(
                        pow(current_node.n - 1, self._get_alpha(current_node)))):
                    current_node = min(current_node.parent.get_children(), key=lambda c: c.n)  # random node
                else:
                    return current_node  # random node

    def expand(self, node: NewNode[TState, TAction], iteration_number: int = None) -> NewNode[TState, TAction]:
        # If the node is terminal, return it
        if node.is_terminal:
            return node

        if isinstance(node, DecisionNode):
            explored_actions = node.explored_actions()
            node.valid_actions = self.mdp.widening_actions(node.state, node.n, iteration_number,
                                                           max_iteration=self.max_iteration)
            unexplored_actions = [a for a in node.valid_actions if a not in explored_actions]
            if len(unexplored_actions) == 0:
                raise RuntimeError("No unexplored actions available")

            action_taken = random.choice(unexplored_actions)
            # new_state = self.mdp.transition(node.state, action_taken)
            return self.create_node(node, action_taken, node.state, node.n, iteration_number)

        elif isinstance(node, RandomNode):
            new_state = self.mdp.transition(node.state, node.inducing)
            return self.create_node(node, node.inducing, new_state, node.parent.n, iteration_number)

    def simulate(self, node: NewNode[TRandom, TAction], depth=0, iteration_number=None) -> float:
        reward = 0.0
        if depth == 0:
            current_node = copy.deepcopy(node)
            while current_node != None:
                if isinstance(current_node, RandomNode):
                    reward += self.mdp.reward(node.state,
                                              node.inducing)  # * self.discount_factor ** (depth + i) this part is
                    # not necessary because it is just reflecting the previous results
                current_node = current_node.parent
        if node.is_terminal:
            return reward
        if isinstance(node, DecisionNode):
            valid_actions = self.mdp.widening_actions(node.state, node.n, iteration_number=iteration_number,
                                                      max_iteration=self.max_iteration)
            random_action = random.choice(valid_actions)
            reward += self.mdp.reward(node.state, random_action, None) * (self.discount_factor ** depth)
            if node.is_terminal or depth >= self.simulation_depth_limit:
                return reward
            else:
                next_state = self.mdp.transition(node.state, random_action)
                next_node = DecisionNode(None, random_action, next_state, valid_actions,
                                         self.mdp.is_terminal(next_state))
                reward += self.simulate(next_node, depth + 1, iteration_number)
                return reward
        elif isinstance(node, RandomNode):
            next_state = self.mdp.transition(node.state, node.inducing)
            next_node = DecisionNode(None, node.inducing, next_state, node.parent.valid_actions,
                                     self.mdp.is_terminal(next_state))
            reward += self.simulate(next_node, depth + 1, iteration_number)
            return reward

    def backpropagate(self, node: NewNode[TRandom, TAction], reward: float) -> None:
        current_state_node = node
        current_reward = reward

        while current_state_node != None:
            current_state_node.max_reward = max(current_reward, current_state_node.max_reward)
            current_state_node.reward += current_reward
            current_state_node.n += 1

            current_state_node = current_state_node.parent
            current_reward *= self.discount_factor

    # Utilities
    def run_search_iteration(self, iteration_number=0):
        # Selection
        root_node = self.root()
        best = self.select(root_node, iteration_number)

        if self.verbose:
            print("Selected:")
            self.display_node(best)

        # Expansion
        expanded = self.expand(best, iteration_number)

        if self.verbose:
            print("Expanded to:")
            self.display_node(expanded)

        # Simulation
        simulated_reward = self.simulate(expanded, iteration_number=iteration_number)

        if self.verbose:
            print(f"Simulated reward: {simulated_reward}")

        # Backpropagation
        self.backpropagate(expanded, simulated_reward)
        return simulated_reward

    def create_node(self, parent: Optional[DecisionNode[TState, TAction]], inducing: Optional[TAction],
                    state: TState, number_of_visits=0, iteration_number=0) -> NewNode[TRandom, TAction]:

        if isinstance(parent, DecisionNode):
            is_terminal = self.mdp.is_terminal(state)
            node = RandomNode(parent, inducing, state, is_terminal)
        else:  # isinstance(state, DecisionNode):
            valid_actions = self.mdp.widening_actions(state, number_of_visits, iteration_number, self.max_iteration)
            is_terminal = self.mdp.is_terminal(state)
            node = DecisionNode(parent, inducing, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(node)
            node.depth = parent.depth + 0.5
        else:
            node.depth = 0

        return node

    # def save_tree_impl(self, depth_limit, node: Optional[TNode], indent: str, path: str):
    #     if node == None or node.depth > depth_limit:
    #         return
    #     with open(path, 'a') as f:
    #         f.write(
    #             f"{indent} {str(node)} (n: {node.n}, reward: {node.reward / node.n:.3f}, UCT: "
    #             f"{self.calculate_uct(node):.3f})")
    #         f.write('\n')
    #
    #     children = node.get_children()
    #
    #     if len(children) == 0:
    #         return
    #
    #     children.sort(key=lambda c: c.reward / c.n, reverse=True)
    #
    #     for child in children[:-1]:
    #         self.save_tree_impl(depth_limit, child, self.generate_indentation(indent) + " ├", path)
    #     self.save_tree_impl(depth_limit, children[-1], self.generate_indentation(indent) + " └", path)
    #
    # def save_tree(self, depth_limit=10, indent="", path=f'runs/', simulation_number=0, prices='None', run_time='None'):
    #     import os
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     path = path + f'/tree{simulation_number}.txt'
    #     node = self.root()
    #     with open(path, 'a') as f:
    #         f.write(f"Prices:{prices}")
    #         f.write('\n')
    #         f.write(f"Run Time:{run_time}")
    #         f.write('\n')
    #     self.save_tree_impl(depth_limit, node, indent, path)

    def calculate_puct(self, node):
        return node.reward + math.sqrt(math.pow(node.parent.n, self._get_epsilon(node.parent)) / node.n)

    def _get_alpha(self, node):
        dmax = 4
        if isinstance(node, DecisionNode):
            if not node.depth <= dmax - 1:
                raise ('Wrong Depth')
            alpha = 1 / (10 * (dmax - node.depth) - 3)
        else:
            if node.depth <= dmax - 1.5:
                alpha = 3 / (10 * (dmax - node.depth) - 3)
            elif node.depth == dmax - 0.5:
                alpha = 1
            else:
                raise ('Wrong Depth')

        return alpha

    def _get_epsilon(self, node):
        dmax = 4
        p = 10
        if isinstance(node, DecisionNode):
            if not node.depth <= dmax - 1:
                raise ('Wrong Depth')
            epsilon = 1 - (3 / (10 * (dmax - node.depth)))
        else:
            raise ('Random Node Called Epsilon which wasn\'t supposed to happen ')

        return epsilon
