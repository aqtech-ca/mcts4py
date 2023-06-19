from mcts4py.Solver import *
from mcts4py.MDP import *
import numpy as np


class DPWSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

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
                 dpw_exploration: float = None,
                 dpw_alpha: float = None,
                 max_random_states: float = math.inf,
                 probabilistic: bool =True,
                 pw_refresh_frequency= 10):
        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.pw_refresh_frequency = pw_refresh_frequency
        super().__init__(exploration_constant, verbose, max_iteration, early_stop, early_stop_condition,
                         exploration_constant_decay)
        self.dpw_exp = dpw_exploration
        self.dpw_alpha = dpw_alpha
        self.max_random_states = max_random_states
        self.probabilistic = probabilistic
        self.__root_node = self.create_node(None, None, mdp.initial_state())

    def root(self) -> NewNode[TAction, TRandom]:
        return self.__root_node

    def select(self, node: DecisionNode[TRandom, TAction], iteration_number: int = None) -> NewNode[
        TAction, TRandom]:
        current_node = node
        counter = 0
        while True:
            counter += 1
            if self.mdp.is_terminal(current_node.state):
                return current_node
            if isinstance(current_node, RandomNode):
                a = (current_node.n + 0.01) ** self.dpw_alpha
                kPrime = math.ceil(self.dpw_exp * a)
                if kPrime > len(current_node.children) and len(current_node.children)<self.max_random_states:
                    new_state = self.mdp.transition(current_node.state, current_node.inducing_action)
                    if new_state in [ch.state for ch in current_node.children]:
                        for ch in current_node.children:
                            if ch.state == new_state:
                                current_node = ch
                                break
                    else:
                        current_node = self.create_node(current_node, current_node.inducing_action, new_state)
                else:
                    if self.probabilistic:
                        children_visits = [ch.n for ch in current_node.children]
                        probabilities = [v / sum(children_visits) for v in children_visits]
                        current_node = np.random.choice(current_node.children, p=probabilities)
                    else:
                        current_node = max(current_node.children, key=lambda c: self.calculate_uct(c))
            if current_node.n % self.pw_refresh_frequency == self.pw_refresh_frequency - 1:
                current_node.valid_actions = self.mdp.actions(current_node.state, current_node.n + 1,
                                                              dpw_exploration=self.dpw_exp,
                                                              dpw_alpha=self.dpw_alpha)
            explored_actions = current_node.explored_actions()
            if len(explored_actions) < len(current_node.valid_actions):
                unexplored_actions = [a for a in current_node.valid_actions if a not in explored_actions]
                action_taken = random.choice(unexplored_actions)
                new_child = self.create_node(current_node, action_taken, current_node.state)
                return new_child
            try:
                current_node = max(current_node.children, key=lambda c: self.calculate_uct(c))
            except ValueError:
                if self.mdp.is_terminal(current_node.state):
                    return current_node
                else: # If the current valid actions are not producing any valid actions expand the actions space
                    current_node.valid_actions = self.mdp.actions(current_node.state, current_node.n+1,
                                                                  dpw_exploration=self.dpw_exp * counter * counter * 2,
                                                                  dpw_alpha=self.dpw_alpha * 2)

    def expand(self, node: TNode) -> TNode:
        return node

    def simulate(self, node: NewNode[TRandom, TAction], depth=0, iteration_number=None) -> float:
        reward = 0
        if depth == 0:
            current_node = node
            while current_node.parent is not None:
                reward += self.mdp.reward(current_node.parent.state, current_node.inducing_action)
                current_node = current_node.parent.parent
                if current_node is None:
                    break
        if self.mdp.is_terminal(node.state):
            return reward
        if isinstance(node, RandomNode):
            current_state =self.mdp.transition(node.state, node.inducing_action)
            if self.mdp.is_terminal(current_state):
                return reward
        else:
            current_state = node.state

        discount = self.discount_factor ** depth
        valid_actions = self.mdp.actions(current_state, node.n, dpw_exploration=self.dpw_exp, dpw_alpha=self.dpw_alpha)
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
            return reward
        next_node = ActionNode(node, random_action)
        next_node.state = new_state
        reward += self.mdp.reward(current_state, random_action) * discount
        reward += self.simulate(next_node, depth=depth + 1)
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
    def run_search_iteration(self, iteration_number=None):
        # Selection
        root_node = self.root()
        best = self.select(root_node)
        if self.verbose:
            print("Expanded to:")
            self.display_node(best)

        simulated_reward = self.simulate(best)

        if self.verbose:
            print(f"Simulated reward: {simulated_reward}")
        self.backpropagate(best, simulated_reward)
        return simulated_reward

    def create_node(self, parent: Optional[DecisionNode[TState, TAction]], inducing_action: Optional[TAction],
                    state: TState, number_of_visits=0) -> NewNode[TRandom, TAction]:

        if isinstance(parent, DecisionNode):
            is_terminal = self.mdp.is_terminal(state)
            node = RandomNode(parent, inducing_action, state, is_terminal)
        else:  # isinstance(state, DecisionNode):
            valid_actions = self.mdp.actions(state, number_of_visits, dpw_alpha=self.dpw_alpha,
                                             dpw_exploration=self.dpw_exp)
            is_terminal = self.mdp.is_terminal(state)
            node = DecisionNode(parent, inducing_action, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(node)
            node.depth = parent.depth + 0.5
        else:
            node.depth = 0

        return node
