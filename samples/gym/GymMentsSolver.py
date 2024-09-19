from mcts4py.Solver import *
from mcts4py.MDP import *
import numpy as np


class MentSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 verbose: bool = False,
                 epsilon: float = 0.8):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.__root_node = SoftmaxActionNode[TState, TAction](None, None, self.mdp.actions(self.mdp.initial_state()))
        self.simulate_action(self.__root_node)
        self.epsilon = epsilon

        super().__init__(exploration_constant, verbose)

    def root(self) -> SoftmaxActionNode[TState, TAction]:
        return self.__root_node

    def select(self, node: SoftmaxActionNode[TState, TAction], iteration_number=None) -> SoftmaxActionNode[
        TState, TAction]:
        if len(node.children) == 0:
            return node

        current_node = node
        self.simulate_action(node)

        while True:
            if self.mdp.is_terminal(current_node.state):
                return current_node

            current_children = current_node.children
            explored_actions = set([c.inducing_action for c in current_children])

            # find all possible actions: explore all possible children before selecting the leaf node
            if len(set(current_node.valid_actions) - explored_actions) > 0:
                return current_node

            # This state has been explored, select best action
            action = self.select_action(current_node)
            next_node = None
            for child in current_node.children:
                if child.inducing_action == action:
                    next_node = child
                    break

            if next_node is None:
                return current_node

            current_node = next_node
            self.simulate_action(current_node)

    def expand(self, node: SoftmaxActionNode[TState, TAction], iteration_number=None) -> SoftmaxActionNode[
        TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        current_children = node.children
        explored_actions = set([c.inducing_action for c in current_children])
        valid_action: set[TAction] = set(node.valid_actions)
        unexplored_actions = valid_action - explored_actions

        # Expand an unexplored action
        action_taken = random.sample(unexplored_actions, 1)[0]

        new_node = SoftmaxActionNode(node, action_taken, self.mdp.actions(node.state))
        node.add_child(new_node)

        self.simulate_action(new_node)

        return new_node

    def simulate(self, node: SoftmaxActionNode[TState, TAction], depth=0) -> float:

        if self.verbose:
            print(f"Simulation: {node.state}")

        if self.mdp.is_terminal(node.state):
            if self.verbose:
                print("Terminal state reached")
            parent = node.get_parent()
            parent_state = parent.state if parent != None else None
            if parent_state is None:
                return 0.0
            return self.mdp.reward(parent_state, node.inducing_action)

        current_state = node.state
        discount = self.discount_factor ** depth

        valid_actions = self.mdp.actions(current_state)
        random_action = random.choice(valid_actions)
        new_state = self.mdp.transition(current_state, random_action)

        reward = self.mdp.reward(current_state, random_action, ) * discount
        if self.mdp.is_terminal(new_state):
            reward = self.mdp.reward(current_state, random_action) * discount
            if self.verbose:
                print(f"-> Terminal state reached: {reward}")
            return reward

        if depth > self.simulation_depth_limit:
            reward = self.mdp.reward(current_state, random_action) * discount
            if self.verbose:
                print(f"-> Depth limit reached: {reward}")
            return reward
        next_node = SoftmaxActionNode(node, random_action, self.mdp.actions(node.state))
        next_node.state = new_state
        reward += self.simulate(next_node, depth=depth + 1)
        return reward

    def backpropagate(self, node: SoftmaxActionNode, reward: float) -> None:
        current_node = node
        parent = node.parent

        if parent is not None:
            direct_reward = self.mdp.reward(parent.state, current_node.inducing_action)
            current_reward = reward

            parent.n += 1
            parent.N_sa[current_node.inducing_action] += 1
            parent.Q_stf[current_node.inducing_action] = direct_reward + reward

            direct_reward = self.mdp.reward(parent.state, current_node.inducing_action)

            parent = parent.parent
            current_node = current_node.parent

            # Traverse the path from the current node to the root
            while parent is not None:
                # Update total visit count for the state
                parent.n += 1

                # Update visit count
                parent.N_sa[current_node.inducing_action] += 1

                # Stable log-sum-exp calculation to avoid underflow/overflow issues
                max_q = max(
                    [current_node.Q_stf[action] / self.discount_factor for action in current_node.valid_actions])
                sum_exp = np.sum(np.exp([(current_node.Q_stf[action] / self.discount_factor) - max_q for action in
                                         current_node.valid_actions]))

                # Avoid log(0) by adding a small value to the sum
                log_sum_exp = np.log(sum_exp + 1e-6) + max_q

                # Update Q_stf value
                a = current_node.inducing_action
                parent.Q_stf[a] = direct_reward + self.discount_factor * log_sum_exp
                # Move to the parent node and discount the reward

                direct_reward = self.mdp.reward(parent.state, current_node.inducing_action)
                current_reward *= self.discount_factor

                current_node = current_node.parent
                parent = parent.parent

    # Utilities

    # find all possible children
    def simulate_action(self, node: SoftmaxActionNode[TState, TAction]):
        if node.parent == None:
            initial_state = self.mdp.initial_state()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(initial_state)
            return

        if node.inducing_action == None:
            raise RuntimeError("Action was null for non-null parent")
        new_state = self.mdp.transition(node.parent.state, node.inducing_action)
        node.state = new_state
        node.valid_actions = self.mdp.actions(new_state)

        # Ensure all valid actions are initialized in Q_stf and N_sa
        for action in node.valid_actions:
            if action not in node.Q_stf:
                node.Q_stf[action] = 0.0
            if action not in node.N_sa:
                node.N_sa[action] = 0

    def select_action(self, node, tau=1.0):
        lambda_s = calculate_lambda_s(node, self.epsilon)
        pi_T = {}

        # Compute the denominator for the softmax
        softmax_denominator = sum(np.exp(node.Q_stf[act] / tau) for act in node.valid_actions)

        # Avoid division by zero
        if softmax_denominator == 0 or np.isnan(softmax_denominator):
            softmax_denominator = 1e-6

        for a in node.valid_actions:
            if a in node.Q_stf:
                try:
                    softmax_part = (1 - lambda_s) * np.exp(node.Q_stf[a] / tau) / softmax_denominator
                    softmax_part = max(softmax_part, 0.0)
                except (OverflowError, FloatingPointError):
                    softmax_part = 0.0
            else:
                softmax_part = 0.0

            uniform_part = lambda_s / len(node.valid_actions)
            pi_T[a] = softmax_part + uniform_part

        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(pi_T[a] for a in node.valid_actions)

        if total_prob == 0 or np.isnan(total_prob):
            probabilities = [1.0 / len(node.valid_actions) for _ in node.valid_actions]
        else:
            probabilities = [pi_T[a] / total_prob for a in node.valid_actions]

        # probabilities are non-negative
        probabilities = [max(p, 0.0) for p in probabilities]

        # probabilities sum to 1
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities) for _ in probabilities]

        actions = list(node.valid_actions)

        return np.random.choice(actions, p=probabilities)

    def softmax(self, Q_stf, tau=1.0):
        e_Qstf = np.exp(Q_stf / tau)
        return e_Qstf / e_Qstf.sum()

    def do_best_action(self, node: SoftmaxActionNode[TState, TAction]) -> TAction:
        return max(node.Q_stf, key=node.Q_stf.get)
