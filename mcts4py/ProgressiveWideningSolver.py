import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
import copy


class ProgressiveWideningSolver(MCTSSolver[TAction, StateNode[TState, TAction]], Generic[TState, TAction]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 exploration_constant: float,
                 discount_factor: float,
                 max_iteration: int,
                 early_stop: bool,
                 early_stop_condition: dict=None,
                 verbose: bool = False):

        self.mdp = mdp
        self.simulation_depth_limit = simulation_depth_limit
        self.discount_factor = discount_factor
        super().__init__(exploration_constant, verbose, max_iteration, early_stop, early_stop_condition)
        self.__root_node = self.create_node(None, None, mdp.initial_state())



    def root(self) -> StateNode[TState, TAction]:
        return self.__root_node

    def select(self, node: StateNode[TState, TAction], iteration_number: int = None) -> StateNode[TState, TAction]:
        current_node = node

        while True:
            current_node.valid_actions = self.mdp.widening_actions(node.state, node.n, iteration_number,
                                                                   self.max_iteration)
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            explored_actions = current_node.explored_actions()
            # This state has not been fully explored, return it
            if len(current_node.valid_actions) > len(explored_actions):
                return current_node

            # This state has been explored, select best action
            current_node = max(current_node.get_children(), key=lambda c: self.calculate_uct(c))

    def expand(self, node: StateNode[TState, TAction], iteration_number: int = None) -> StateNode[TState, TAction]:
        # If the node is terminal, return it
        if self.mdp.is_terminal(node.state):
            return node

        explored_actions = node.explored_actions()
        unexplored_actions = [a for a in node.valid_actions if a not in explored_actions]

        if len(unexplored_actions) == 0:
            raise RuntimeError("No unexplored actions available")

        # Expand an unexplored action
        action_taken = random.choice(unexplored_actions)

        new_state = self.mdp.transition(node.state, action_taken)
        return self.create_node(node, action_taken, new_state, node.n, iteration_number)

    def simulate(self, node: StateNode[TState, TAction], depth=0, iteration_number=None) -> float:
        if self.verbose:
            print("Simulation:")
        reward = 0
        if depth == 0:
            temp_node = copy.copy(node)
            i = 0
            while temp_node.parent != None:
                discount = self.discount_factor ** (depth + i)
                i += 1
                reward += self.mdp.reward(temp_node.parent.state, temp_node.inducing_action, temp_node.state) * discount
                temp_node = temp_node.parent
        if self.mdp.is_terminal(node.state):
            if self.verbose:
                print("Terminal state reached")
            parent = node.get_parent()
            parent_state = parent.state if parent != None else None
            # reward += self.mdp.reward(parent_state, node.inducing_action, node.state) # ALREADY INCLUDED UPPER
            return reward

        current_state = node.state
        discount = self.discount_factor ** depth
        valid_actions = self.mdp.widening_actions(current_state, node.n, iteration_number=iteration_number,
                                                  max_iteration_number=self.max_iteration)
        random_action = random.choice(valid_actions)
        new_state = self.mdp.transition(current_state, random_action)

        if self.mdp.is_terminal(new_state):
            reward += self.mdp.reward(current_state, random_action, new_state) * discount
            if self.verbose:
                print(f"-> Terminal state reached: {reward}")
            return reward

        ## Causing the loop to finish before all rewards are realized.

        if depth > self.simulation_depth_limit:
            reward += self.mdp.reward(current_state, random_action, new_state) * discount
            if self.verbose:
                print(f"-> Depth limit reached: {reward}")
            return reward
        next_node = ActionNode(node, random_action)
        next_node.state = new_state
        reward += self.mdp.reward(current_state, random_action, new_state) * discount
        reward += self.simulate(next_node, depth=depth + 1, iteration_number=iteration_number)
        return reward

    def backpropagate(self, node: StateNode[TState, TAction], reward: float) -> None:
        current_state_node = node
        current_reward = reward

        while current_state_node != None:
            current_state_node.max_reward = max(current_reward, current_state_node.max_reward)
            current_state_node.reward += current_reward
            current_state_node.n += 1

            current_state_node = current_state_node.parent
            current_reward *= self.discount_factor

    # Utilities
    def run_search_iteration(self, iteration_number=0):
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

        valid_actions = self.mdp.widening_actions(state, number_of_visits, iteration_number, self.max_iteration)
        is_terminal = self.mdp.is_terminal(state)
        state_node = StateNode(parent, inducing_action, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(state_node)

        return state_node


    def save_tree_impl(self, depth_limit, node: Optional[TNode], indent: str, path: str):
        if node == None or node.depth > depth_limit:
            return
        with open(path, 'a') as f:
            f.write(
                f"{indent} {str(node)} (n: {node.n}, reward: {node.reward / node.n:.3f}, UCT: "
                f"{self.calculate_uct(node):.3f})")
            f.write('\n')

        children = node.get_children()

        if len(children) == 0:
            return

        children.sort(key=lambda c: c.reward / c.n, reverse=True)

        for child in children[:-1]:
            self.save_tree_impl(depth_limit, child, self.generate_indentation(indent) + " ├", path)
        self.save_tree_impl(depth_limit, children[-1], self.generate_indentation(indent) + " └", path)

    def save_tree(self, depth_limit=10, indent="", path=f'runs/', simulation_number=0, prices='None', run_time='None'):
        import os
        if not os.path.exists(path):
            os.makedirs(path)
        path = path + f'/tree{simulation_number}.txt'
        node = self.root()
        with open(path, 'a') as f:
            f.write(f"Prices:{prices}")
            f.write('\n')
            f.write(f"Run Time:{run_time}")
            f.write('\n')
        self.save_tree_impl(depth_limit, node, indent, path)