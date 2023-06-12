import math

from mcts4py.DPWSolver import *
from mcts4py.MDP import *


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
                 dpw_alpha=1,
                 max_random_states=math.inf,
                 probabilistic=True,
                 pw_refresh_frequency=10,
                 probabilities: pd.DataFrame =None):
        super().__init__(mdp, simulation_depth_limit, discount_factor, exploration_constant, verbose, max_iteration,
                         early_stop, early_stop_condition, exploration_constant_decay, dpw_exploration, dpw_alpha, max_random_states, probabilistic, pw_refresh_frequency)
        self.probabilities = probabilities  # probabilities table has to have the node as the index column

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
                if kPrime > len(current_node.children) and len(current_node.children) < self.max_random_states:
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
                        current_node = max(current_node.children, key=lambda c: self.calculate_puct(c))
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
                current_node = max(current_node.children, key=lambda c: self.calculate_puct(c))
            except ValueError:
                if self.mdp.is_terminal(current_node.state):
                    return current_node
                else:  # If the current valid actions are not producing any valid actions expand the actions space
                    current_node.valid_actions = self.mdp.actions(current_node.state, current_node.n + 1,
                                                                  dpw_exploration=self.dpw_exp * counter * counter * 2,
                                                                  dpw_alpha=self.dpw_alpha * 2)

    def expand(self, node: TNode) -> TNode:
        return node

    def calculate_puct(self, node):
        prob = self.probabilities.loc[node]
        puct_constant = self.exploration_constant * prob
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculate_uct_impl(parentN, node.n, node.reward, puct_constant)
