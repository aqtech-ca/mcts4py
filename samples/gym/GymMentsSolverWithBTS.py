from mcts4py.Solver import *
from mcts4py.MDP import *


class MENTSSolverV1WithBTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 temperature: float = 1.0,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.temperature = temperature
        self.__root_node = SoftmaxActionNode[TState, TAction](None, None, self.mdp.actions(self.mdp.initial_state()))
        self.__root_node.Q_stf = {action: 0.0 for action in self.mdp.actions(self.mdp.initial_state())}
        self.simulate_action(self.__root_node)

        super().__init__(exploration_constant, verbose)

    def root(self) -> SoftmaxActionNode[TState, TAction]:
        return self.__root_node

    def select(self, node: SoftmaxActionNode[TState, TAction], iteration_number=None) -> SoftmaxActionNode[TState, TAction]:
        if len(node.children) == 0:
            return node

        current_node = node
        self.simulate_action(node)

        while True:
            if self.mdp.is_terminal(current_node.state):
                return current_node

            current_children = current_node.children
            explored_actions = set([c.inducing_action for c in current_children])

            # Explore all possible actions before selecting a leaf node
            if len(set(current_node.valid_actions) - explored_actions) > 0:
                return current_node

            # Select action based on Boltzmann policy
            action = self.select_action(current_node, tau=self.temperature)
            next_node = None
            for child in current_node.children:
                if child.inducing_action == action:
                    next_node = child
                    break

            if next_node is None:
                return current_node

            current_node = next_node
            self.simulate_action(current_node)

    def expand(self, node: SoftmaxActionNode[TState, TAction], iteration_number=None) -> SoftmaxActionNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        current_children = node.children
        explored_actions = set([c.inducing_action for c in current_children])
        valid_action: set[TAction] = set(node.valid_actions)
        unexplored_actions = valid_action - explored_actions

        if not unexplored_actions:
            return node

        # Expand an unexplored action
        action_taken = random.sample(unexplored_actions, 1)[0]

        new_node = SoftmaxActionNode(node, action_taken, self.mdp.actions(node.state))
        new_node.Q_stf = {action: 0.0 for action in new_node.valid_actions}
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
            parent_state = parent.state if parent is not None else None
            if parent_state is None:
                return 0.0
            return self.mdp.reward(parent_state, node.inducing_action)

        current_state = node.state
        discount = self.discount_factor ** depth

        valid_actions = self.mdp.actions(current_state)
        random_action = random.choice(valid_actions)
        new_state = self.mdp.transition(current_state, random_action)

        reward = self.mdp.reward(current_state, random_action) * discount
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

        next_node = SoftmaxActionNode(node, random_action, self.mdp.actions(new_state))
        next_node.Q_stf = {action: 0.0 for action in next_node.valid_actions}
        next_node.state = new_state
        reward += self.simulate(next_node, depth=depth + 1)
        return reward

    def backpropagate(self, node: SoftmaxActionNode, reward: float) -> None:
        """
        Backpropagates the reward up the tree, updating Q-values and visit counts.

        Args:
            node: The node from which to start backpropagation.
            reward: The reward obtained from the simulation.
        """
        current_node = node
        discount = self.discount_factor

        while current_node is not None:
            current_node.n += 1
            for action in current_node.valid_actions:
                if action not in current_node.Q_stf:
                    current_node.Q_stf[action] = 0.0
                if action not in current_node.N_sa:
                    current_node.N_sa[action] = 0

            if current_node.parent is not None:
                action = current_node.inducing_action
                current_node.parent.Q_stf[action] += (reward * discount)
                current_node.parent.N_sa[action] += 1

            current_node = current_node.parent
            discount *= self.discount_factor

    # Utilities

    def simulate_action(self, node: SoftmaxActionNode[TState, TAction]):
        if node.parent is None:
            initial_state = self.mdp.initial_state()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(initial_state)
        else:
            if node.inducing_action is None:
                raise RuntimeError("Action was null for non-null parent")
            new_state = self.mdp.transition(node.parent.state, node.inducing_action)
            node.state = new_state
            node.valid_actions = self.mdp.actions(new_state)

        # Initialize Q-values and visit counts for all valid actions
        for action in node.valid_actions:
            if action not in node.Q_stf:
                node.Q_stf[action] = 0.0  # Initialize Q-value to zero
            if action not in node.N_sa:
                node.N_sa[action] = 0

    def select_action(self, node, tau=1.0):
        """
        Selects an action based on the Boltzmann (softmax) policy.

        Args:
            node: The current node from which to select an action.
            tau: Temperature parameter controlling exploration-exploitation trade-off.

        Returns:
            Selected action based on the Boltzmann policy.
        """
        q_values = np.array([node.Q_stf.get(a, 0.0) for a in node.valid_actions])
        # Avoid numerical overflow by subtracting the max Q-value
        q_values = q_values - np.max(q_values)
        exp_q = np.exp(q_values / tau)
        probabilities = exp_q / np.sum(exp_q)
        actions = list(node.valid_actions)
        return np.random.choice(actions, p=probabilities)

    def softmax(self, Q_stf, tau=1.0):
        e_Qstf = np.exp(Q_stf / tau)
        return e_Qstf / e_Qstf.sum()

    def do_best_action(self, node: SoftmaxActionNode[TState, TAction]) -> TAction:
        return max(node.Q_stf, key=node.Q_stf.get)

    def update_temperature(self, decay_rate: float = 0.99):
        self.temperature *= decay_rate