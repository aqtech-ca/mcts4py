from math import sqrt, log
from mcts4py.Types import *
from mcts4py.Nodes import *
from abc import ABC, abstractmethod


class MCTSSolver(ABC, Generic[TAction, TNode]):
    def __init__(self, exploration_constant: float, verbose: bool = False):
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
    def update(self, node: TNode, reward: float) -> None:
        raise NotImplementedError

    def run_search(self, iterations: int):
        for i in range(iterations):
            if self.verbose:
                print(f"\nNew iteration: {i}")
                print("=======================")
            self.run_search_iteration()

    def run_search_iteration(self):
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

        # Update
        self.update(expanded, simulated_reward)

    # Utilities

    def calculate_uct(self, node: TNode) -> float:
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, self.exploration_constant)

    def calculate_uct_impl(self, parentN: TNode, n: TNode, reward: float, exploration_constant: float) -> float:
        return reward/n + exploration_constant * sqrt(log(parentN)/n)

    def extract_optimal_action(self) -> Optional[TAction]:
        return max(self.root().get_children(), key=lambda c: c.n)

    def display_node(self, node: TNode) -> None:
        if node.parent != None:
            self.display_node(node.parent)

        if node.depth > 0:
            print("  "*(node.depth - 1) + " └ " + str(node))

    def display_tree(self, depth_limit: int = 3) -> None:
        self.display_tree_impl(depth_limit, self.root(), "")

    def display_tree_impl(self, depth_limit: int, node: Optional[TNode], indent: str) -> None:

        if node == None or node.depth > depth_limit:
            return

        print(f"{indent} {str(node)} (n: {node.n}, reward: {node.reward/node.n:.3f}, UCT: {self.calculate_uct(node):.3f})")

        children = node.get_children()

        if len(children) == 0:
            return

        for child in children[:-1]:
            self.display_tree_impl(depth_limit, child, self.generate_indentation(indent) + " ├")
        self.display_tree_impl(depth_limit, children[-1], self.generate_indentation(indent) + " └")

    def generate_indentation(self, indent: str):
        return indent.replace('├', '│').replace('└', ' ')

