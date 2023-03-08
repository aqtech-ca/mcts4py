from math import sqrt, log
from mcts4py.Types import *
from mcts4py.Nodes import *
from tqdm import tqdm
from abc import ABC, abstractmethod
import numpy as np

class MCTSSolver(ABC, Generic[TAction, TNode]):
    def __init__(self, exploration_constant: float, verbose: bool = False, max_iteration=1000, early_stop=False,
                 early_stop_condition : dict=None):
        """
        :param exploration_constant:
        :param verbose:
        :param max_iteration:
        :param early_stop:
        :param early_stop_condition: Min number of iterations to run, epsilon to check simulated reward difference,
               and the number of iterations to check the stability of the latest simulated rewards
               ie: {'min_iterations': 100, 'epsilon': 0.0005, 'last_iterations_number': 50}
        """
        if early_stop is True:
            assert early_stop_condition is not None, "early_stop_condition must be provided if early_stop is True"
            self.min_iterations = early_stop_condition['min_iterations']
            self.epsilon = early_stop_condition['epsilon']
            self.last_iterations_number = early_stop_condition['last_iterations_number']
        self.max_iteration = max_iteration
        self.early_stop = early_stop
        self.exploration_constant = exploration_constant
        self.verbose = verbose

    @abstractmethod
    def root(self) -> TNode:
        raise NotImplementedError

    @abstractmethod
    def select(self, node: TNode) -> TNode:
        raise NotImplementedError

    @abstractmethod
    def expand(self, node: TNode) -> TNode:
        raise NotImplementedError

    @abstractmethod
    def simulate(self, node: TNode) -> float:
        raise NotImplementedError

    @abstractmethod
    def backpropagate(self, node: TNode, reward: float) -> None:
        raise NotImplementedError

    def run_search(self, iterations: int):
        simulated_rewards = []
        for iteration_number in tqdm(range(iterations), desc="Iterations", position=0, leave=False):
            if self.verbose:
                print(f"\nNew iteration: {iteration_number}")
                print("=======================")
            simulated_rewards.append(self.run_search_iteration(iteration_number=iteration_number))
            if self.early_stop:
                if iteration_number > self.min_iterations and self.early_stop:
                    a = np.mean(simulated_rewards[-self.last_iterations_number:])
                    if a >= simulated_rewards[-1] * (1+self.epsilon) and a <= simulated_rewards[-1] * (1-self.epsilon):
                        if self.verbose:
                            print('Iteration: ', iteration_number)
                            print("----------------- EARLY STOPPING -----------------")
                        break

    def run_search_iteration(self, iteration_number=0):
        # Selection
        root_node = self.root()
        best = self.select(root_node)

        if self.verbose:
            print("Selected:")
            self.display_node(best)

        # Expansion
        expanded = self.expand(best)

        if self.verbose:
            print("Expanded to:")
            self.display_node(expanded)

        # Simulation
        simulated_reward = self.simulate(expanded)

        if self.verbose:
            print(f"Simulated reward: {simulated_reward}")

        # Backpropagation
        self.backpropagate(expanded, simulated_reward)
        return simulated_reward

    # Utilities

    def calculate_uct(self, node: TNode) -> float:
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, self.exploration_constant)

    def calculate_uct_impl(self, parentN: TNode, n: TNode, reward: float, exploration_constant: float) -> float:
        return reward / n + exploration_constant * sqrt(log(parentN) / n)

    def extract_optimal_action(self) -> Optional[TAction]:
        return max(self.root().get_children(), key=lambda c: c.reward / c.n)

    def display_node(self, node: TNode) -> None:
        if node.parent != None:
            self.display_node(node.parent)

        if node.depth > 0:
            print("  " * (node.depth - 1) + " └ " + str(node))

    def display_tree(self, depth_limit: int = 3) -> None:
        self.display_tree_impl(depth_limit, self.root(), "")

    def display_tree_impl(self, depth_limit: int, node: Optional[TNode], indent: str) -> None:

        if node == None or node.depth > depth_limit:
            return

        print(
            f"{indent} {str(node)} (n: {node.n}, reward: {node.reward / node.n:.3f}, UCT: "
            f"{self.calculate_uct(node):.3f})")

        children = node.get_children()

        if len(children) == 0:
            return

        children.sort(key=lambda c: c.reward / c.n, reverse=True)

        for child in children[:-1]:
            self.display_tree_impl(depth_limit, child, self.generate_indentation(indent) + " ├")
        self.display_tree_impl(depth_limit, children[-1], self.generate_indentation(indent) + " └")

    def generate_indentation(self, indent: str):
        return indent.replace('├', '│').replace('└', ' ')