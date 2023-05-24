from mcts4py.ProgressiveWideningSolver import *
from mcts4py.MDP import *


class PuctSolver(ProgressiveWideningSolver):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 discount_factor: float,
                 exploration_constant: float,
                 verbose: bool = False,
                 max_iteration: int = 1000,
                 early_stop: bool = False,
                 early_stop_condition: dict = None,
                 exploration_constant_decay=1):

        super().__init__(mdp, simulation_depth_limit, discount_factor, exploration_constant, verbose, max_iteration, early_stop,
                         early_stop_condition, exploration_constant_decay)

    def select(self, node: StateNode[TState, TAction], iteration_number=None) -> StateNode[TState, TAction]:
        current_node = node

        while True:
            if current_node.n % 50 == 49:
                current_node.valid_actions = self.mdp.actions(current_node.state, current_node.n, iteration_number,
                                                              self.max_iteration, dpw_exploration=self.dpw_exploration,
                                                              dpw_alpha=self.dpw_alpha)
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            explored_actions = current_node.explored_actions()
            # This state has not been fully explored, return it
            if len(current_node.valid_actions) > len(explored_actions):
                return current_node

            # This state has been explored, select best action

            current_node = max(current_node.children, key=lambda c: self.calculate_puct(c))

    def calculate_puct(self, node):
        if node.parent is None:
            if node.inducing_action.refuel_amount > 100:
                prob = 0.2
            elif node.inducing_action.refuel_amount > 50:
                prob = 0.3
            else:
                prob = 1
        elif node.parent.state.fuel_amount < 30 and node.inducing_action.refuel_amount > 70:
            prob = 1
        else:
            prob = 0.5

        puct_constant = self.exploration_constant * prob
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, puct_constant)
