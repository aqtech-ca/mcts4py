from mcts4py.Solver import *
from mcts4py.StatefulSolver import StatefulSolver
from mcts4py.MDP import *
import copy


class ProgressiveWideningSolver(StatefulSolver):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 discount_factor: float,
                 exploration_constant: float,
                 verbose: bool = False,
                 max_iteration: int = 1000,
                 early_stop: bool = False,
                 early_stop_condition: dict = None,
                 exploration_constant_decay: float = 1.0,
                 dpw_exploration: float = None,
                 dpw_alpha: float = None,
                 pw_action_refresh_frequency = 10
                 ):
        self.dpw_exploration = dpw_exploration
        self.dpw_alpha = dpw_alpha
        self.pw_action_refresh_frequency = pw_action_refresh_frequency
        super().__init__(mdp, simulation_depth_limit, discount_factor, exploration_constant,
                         verbose, max_iteration, early_stop, early_stop_condition, exploration_constant_decay)

    def select(self, node: StateNode[TState, TAction], iteration_number: int = None) -> StateNode[
        TState, TAction]:
        current_node = node

        while True:
            if current_node.n % self.pw_action_refresh_frequency == self.pw_action_refresh_frequency - 1:
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
            current_node = max(current_node.children, key=lambda c: self.calculate_uct(c))

    def expand(self, node: StateNode[TState, TAction], iteration_number=None) -> StateNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        explored_actions = node.explored_actions()
        unexplored_actions = [a for a in node.valid_actions if a not in explored_actions]

        if len(unexplored_actions) == 0:
            raise RuntimeError("No unexplored actions available")

        # Expand an unexplored action
        # random choice with seed
        action_taken = random.choice(unexplored_actions)

        new_state = self.mdp.transition(node.state, action_taken)
        return self.create_node(node, action_taken, new_state, node.n, iteration_number=iteration_number)

    def simulate(self, node: StateNode[TState, TAction], depth=0, iteration_number=None) -> float:
        if self.verbose:
            print("Simulation:")
        reward = 0
        if depth == 0:
            curnode = copy.copy(node)
            i = 0
            while curnode.parent is not None:
                discount = self.discount_factor ** (depth + i)
                i += 1
                reward += self.mdp.reward(curnode.parent.state, curnode.inducing_action) * discount
                curnode = curnode.parent
        if self.mdp.is_terminal(node.state):
            if self.verbose:
                print("Terminal state reached")
            return reward
        current_state = node.state
        discount = self.discount_factor ** depth
        valid_actions = self.mdp.actions(current_state, node.n, iteration_number=iteration_number,
                                         max_iteration_number=self.max_iteration, dpw_alpha=self.dpw_alpha,
                                         dpw_exploration=self.dpw_exploration,
                                         min_action=True)
        random_action = random.choice(valid_actions)
        new_state = self.mdp.transition(current_state, random_action)

        if self.mdp.is_terminal(new_state):
            reward += self.mdp.reward(current_state, random_action) * discount
            if self.verbose:
                print(f"-> Terminal state reached: {reward}")
            return reward

        ## Causing the loop to finish before all rewards are realized.

        if depth > self.simulation_depth_limit:
            reward += self.mdp.reward(current_state, random_action) * discount
            if self.verbose:
                print(f"-> Depth limit reached: {reward}")
            return reward
        next_node = ActionNode(node, random_action)
        next_node.state = new_state
        reward += self.mdp.reward(current_state, random_action) * discount
        reward += self.simulate(next_node, depth=depth + 1, iteration_number=iteration_number)
        return reward

    # Utilities
    def run_search_iteration(self, iteration_number=None):
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

    def create_node(self, parent: Optional[StateNode[TState, TAction]], inducing_action: Optional[TAction],
                    state: TState, number_of_visits=0, iteration_number=0) -> StateNode[TState, TAction]:

        valid_actions = self.mdp.actions(state, number_of_visits, iteration_number, self.max_iteration,
                                         dpw_exploration=self.dpw_exploration, dpw_alpha=self.dpw_alpha)
        is_terminal = self.mdp.is_terminal(state)
        state_node = StateNode(parent, inducing_action, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(state_node)

        return state_node
