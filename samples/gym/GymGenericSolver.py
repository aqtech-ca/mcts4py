import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *


class GenericSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.__root_node = ActionNode[TState, TAction](None, None)
        self.simulate_action(self.__root_node)

        super().__init__(exploration_constant, verbose)

    def root(self) -> ActionNode[TState, TAction]:
        return self.__root_node

    def select(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        if len(node.children) == 0:
            return node

        current_node = node
        self.simulate_action(node)

        while True:
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            current_children = current_node.children
            explored_actions = set([c.inducing_action for c in current_children])

            # This state has not been fully explored, return it
            if len(set(current_node.valid_actions) - explored_actions) > 0:
                return current_node

            # This state has been explored, select best action
            current_node = max(current_children, key=lambda c: self.calculate_uct(c))
            self.simulate_action(current_node)

    def expand(self, node: ActionNode[TState, TAction], iteration_number=None) -> ActionNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        current_children = node.children
        explored_actions = set([c.inducing_action for c in current_children])
        valid_action: set[TAction] = set(node.valid_actions)
        unexplored_actions = valid_action - explored_actions

        # Expand an unexplored action
        action_taken = random.sample(unexplored_actions, 1)[0]

        new_node = ActionNode(node, action_taken)
        node.add_child(new_node)
        self.simulate_action(new_node)

        return new_node

    def simulate(self, node: ActionNode[TState, TAction], depth=0) -> float:

        if self.verbose:
            print("Simulation:")

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
        next_node = ActionNode(node, random_action)
        next_node.state = new_state
        reward += self.simulate(next_node, depth=depth + 1)
        return reward

    def backpropagate(self, node: ActionNode[TState, TAction], reward: float) -> None:
        current_node = node
        current_reward = reward

        while current_node != None:
            current_node.max_reward = max(current_reward, current_node.max_reward)
            current_node.reward += current_reward
            current_node.n += 1

            current_node = current_node.parent
            current_reward *= self.discount_factor

    # Utilities

    def simulate_action(self, node: ActionNode[TState, TAction]):
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

    def do_best_action(self, node: ActionNode[TState,TAction]) -> TAction:

        best_child = max(node.children, key=lambda c: c.reward / c.n if c.n > 0 else float('-inf'))

        return best_child.inducing_action
