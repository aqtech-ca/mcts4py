import random
from typing import List

from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
from samples.gridworld.GridworldMDP import GridworldAction


class MentSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.__root_node = SoftmaxActionNode[TState, TAction](None, None, self.mdp.actions(self.mdp.initial_state()))
        self.simulate_action(self.__root_node)

        super().__init__(exploration_constant, verbose)

    def root(self) -> SoftmaxActionNode[TState, TAction]:
        return self.__root_node

    def select(self, node: SoftmaxActionNode[TState, TAction], iteration_number=None, epsilon=0.1) -> SoftmaxActionNode[
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
            action = self.select_action(current_node, epsilon)
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
            return self.mdp.reward(parent_state, node.inducing_action, node.state)

        current_state = node.state
        discount = self.discount_factor ** depth

        valid_actions = self.mdp.actions(current_state)
        random_action = random.choice(valid_actions)
        new_state = self.mdp.transition(current_state, random_action)

        reward = self.mdp.reward(current_state, random_action, new_state) * discount
        if self.mdp.is_terminal(new_state):
            reward = self.mdp.reward(current_state, random_action, new_state) * discount
            if self.verbose:
                print(f"-> Terminal state reached: {reward}")
            return reward

        if depth > self.simulation_depth_limit:
            reward = self.mdp.reward(current_state, random_action, new_state) * discount
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
        direct_reward = self.mdp.reward(parent, current_node.inducing_action, current_node.state)
        current_reward = reward

        parent.n += 1
        parent.N_sa[current_node.inducing_action] += 1
        parent.Q_stf[current_node.inducing_action] = direct_reward + reward

        current_node = current_node.parent
        parent = parent.parent
        direct_reward = self.mdp.reward(parent, current_node.inducing_action, current_node.state)

        # Traverse the path from the current node to the root
        while parent is not None:
            # Update total visit count for the state
            parent.n += 1

            # Update visit count
            parent.N_sa[current_node.inducing_action] += 1

            # Update Q_stf value
            a = current_node.inducing_action
            parent.Q_stf[a] = (direct_reward + self.discount_factor *
                    np.log(np.sum(np.exp([current_node.Q_stf[action] / self.discount_factor for action in current_node.valid_actions]))))
            # Move to the parent node and discount the reward
            current_node = current_node.parent
            parent = parent.parent
            direct_reward = self.mdp.reward(parent, current_node.inducing_action, current_node.state)
            current_reward *= self.discount_factor

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

    def select_action(self, node, epsilon, tau=1.0):
        lambda_s = calculate_lambda_s(node, epsilon)
        pi_T = {}

        for a in node.valid_actions:
            if a in node.Q_stf:
                softmax_part = (1 - lambda_s) * np.exp(node.Q_stf[a] / tau) / sum(
                    np.exp(node.Q_stf[act]) / tau for act in node.valid_actions)
            else:
                # Handle case where Q_stf[a] is not initialized or available
                softmax_part = 0.0

            uniform_part = lambda_s / len(node.valid_actions)
            pi_T[a] = softmax_part + uniform_part

        # Normalize probabilities to ensure they sum to 1
        total_prob = sum(pi_T[a] for a in node.valid_actions)
        probabilities = [pi_T[a] / total_prob for a in node.valid_actions]

        actions = list(node.valid_actions)
        return np.random.choice(actions, p=probabilities)

    def print_node(self, node: SoftmaxActionNode[TState, TAction], depth=0) -> None:
        indent = ' ' * (depth * 2)
        print(
            f"{indent}Node(State: {node.state}, Action: {node.inducing_action}, Q_stf: <{self.printQstf(node.Q_stf)}>, n: {node.n}, N_sa: <{self.printQstf(node.N_sa)}>)")

    def print_tree(self, node: SoftmaxActionNode[TState, TAction], depth=0) -> None:
        self.print_node(node, depth)
        for child in node.children:
            self.print_tree(child, depth + 1)

    def softmax(self, Q_stf, tau=1.0):

        e_Qstf = np.exp(Q_stf / tau)
        return e_Qstf / e_Qstf.sum()

    def printQstf(self, dics):
        res = ""
        for action, value in dics.items():
            if action == GridworldAction.UP:
                arrow = '↑'
            elif action == GridworldAction.DOWN:
                arrow = '↓'
            elif action == GridworldAction.LEFT:
                arrow = '←'
            elif action == GridworldAction.RIGHT:
                arrow = '→'
            else:
                arrow = str(action)

            res += f"'{arrow}': {value}, "

        return res

