import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *

class StatefulSolver(MCTSSolver[TAction, StateNode[TState, TAction]], Generic[TState, TAction]):

    def __init__(self,
        mdp: MDP[TState, TAction],
        simulation_depth_limit: int,
        exploration_constant: float,
        discount_factor: float,
        verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        self.__root_node = self.create_node(None, None, mdp.initialState())

        super().__init__(exploration_constant, verbose)

    def root(self) -> StateNode[TState, TAction]:
        return self.__root_node

    def select(self, node: StateNode[TState, TAction]) -> StateNode[TState, TAction]:
        current_node = node

        while True:
            # If the node is terminal, return it
            if self.mdp.isTerminal(current_node.state):
                return current_node

            explored_actions = current_node.explored_actions()
            # This state has not been fully explored, return it
            if len(current_node.valid_actions) > len(explored_actions):
                return current_node

            # This state has been explored, select best action
            return max(current_node.get_children(), key=lambda c: self.calculate_uct(c))

    def expand(self, node: StateNode[TState, TAction]) -> StateNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.isTerminal(node.state):
            return node

        explored_actions = node.explored_actions()
        unexploredActions = [a for a in node.valid_actions if a not in explored_actions]

        if len(unexploredActions) == 0:
            raise RuntimeError("No unexplored actions available")

        # Expand an unexplored action
        action_taken = random.choice(unexploredActions)

        new_state = self.mdp.transition(node.state, action_taken)
        return self.create_node(node, action_taken, new_state)

    def simulate(self, node: StateNode[TState, TAction]) -> float:
        if self.verbose:
            print("Simulation:")

        if node.is_terminal:
            if self.verbose:
                print("Terminal state reached")
            parent = node.get_parent()
            parent_state = parent.state if parent != None else None
            return self.mdp.reward(parent_state, node.inducing_action, node.state)

        depth = 0
        current_state = node.state
        discount = self.discount_factor

        while True:
            valid_actions = self.mdp.actions(current_state)
            random_action = random.choice(valid_actions)
            new_state = self.mdp.transition(current_state, random_action)

            if self.mdp.isTerminal(new_state):
                reward = self.mdp.reward(current_state, random_action, new_state) * discount
                if self.verbose:
                    print(f"-> Terminal state reached: {reward}")
                return reward

            current_state = new_state
            depth += 1
            discount *= self.discount_factor

            if depth > self.simulation_depth_limit:
                reward = self.mdp.reward(current_state, random_action, new_state) * discount
                if self.verbose:
                    print("-> Depth limit reached: " + str(reward))
                return reward

    def update(self, node: StateNode[TState, TAction], reward: float) -> None:
        current_state_node = node
        current_reward = reward

        while current_state_node != None:
            current_state_node.max_reward = max(current_reward, current_state_node.max_reward)
            current_state_node.reward += current_reward
            current_state_node.n += 1

            current_state_node = current_state_node.parent
            current_reward *= self.discount_factor

    # Utilities

    def create_node(self, parent: Optional[StateNode[TState, TAction]], inducing_action: Optional[TAction], state: TState) -> StateNode[TState, TAction]:
        valid_actions = self.mdp.actions(state)
        is_terminal = self.mdp.isTerminal(state)
        state_node = StateNode(parent, inducing_action, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(state_node)

        return state_node

