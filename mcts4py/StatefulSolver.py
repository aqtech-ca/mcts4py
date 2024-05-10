import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *

random = random.Random(0)
import copy
import inspect


accepts_arguments = lambda func, num_args: len(inspect.signature(func).parameters) == num_args


class StatefulSolver(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

    def __init__(self,
                 mdp: MDP[TState, TAction],
                 simulation_depth_limit: int,
                 discount_factor: float,
                 exploration_constant: float,
                 verbose: bool = False,
                 max_iteration: int = 1000,
                 early_stop: bool = False,
                 early_stop_condition: dict = None,
                 exploration_constant_decay = 1,
                 value_function_estimator_callback = None,
                 alpha_value = 0.5,
                 value_clipping: bool = False,
                 value_function_upper_estimator_callback = None,
                 value_function_lower_estimator_callback = None):
        self.mdp = mdp
        self.discount_factor = discount_factor
        self.simulation_depth_limit = simulation_depth_limit
        self.value_function_estimator_callback = value_function_estimator_callback
        self.alpha_value = alpha_value
        self.value_clipping = value_clipping
        self.value_function_upper_estimator_callback = value_function_upper_estimator_callback
        self.value_function_lower_estimator_callback = value_function_lower_estimator_callback
        
        super().__init__(exploration_constant, verbose, max_iteration,
                         early_stop, early_stop_condition, exploration_constant_decay)
        self.__root_node = self.create_node(None, None, mdp.initial_state())

    def root(self) -> StateNode[TState, TAction]:
        return self.__root_node

    def select(self, node: StateNode[TState, TAction]) -> StateNode[TState, TAction]:
        current_node = node

        while True:
            current_node.valid_actions = self.mdp.actions(current_node.state)
            # If the node is terminal, return it
            if self.mdp.is_terminal(current_node.state):
                return current_node

            explored_actions = current_node.explored_actions()
            # This state has not been fully explored, return it
            if len(current_node.valid_actions) > len(explored_actions):
                return current_node

            # This state has been explored, select best action
            current_node = max(current_node.children, key=lambda c: self.calculate_uct(c))

    def expand(self, node: StateNode[TState, TAction], iteration_number = None) -> StateNode[TState, TAction]:
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
        return self.create_node(node, action_taken, new_state, node.n)
    
    def simulate(self, node: StateNode[TState, TAction], mc_sim_iter = 10) -> float:
        if self.verbose:
            print("Simulation:")

        if node.is_terminal:
            if self.verbose:
                print("Terminal state reached")
            parent = node.get_parent()
            parent_state = parent.state if parent != None else None
            return self.mdp.reward(parent_state, node.inducing_action, node.state)
        
        use_value_approx = np.random.uniform(0, 1) > self.alpha_value
        if self.value_function_estimator_callback is None:
            sim_reward, trajectory_history = self.simulate_by_simulation(node, mc_sim_iter = mc_sim_iter)
        else:
            # value_function_estimator_callback() will receive a StateNode object.
            if use_value_approx:
                sim_reward = self.value_function_estimator_callback(node)
            else:
                sim_reward, trajectory_history = self.simulate_by_simulation(node, mc_sim_iter = mc_sim_iter)
        
        if self.value_clipping and self.value_function_lower_estimator_callback is not None:
            lower_est = self.value_function_lower_estimator_callback(node)
            sim_reward = np.max([sim_reward, self.value_function_lower_estimator_callback(node)])
        elif self.value_clipping and self.value_function_upper_estimator_callback is not None:
            if any(param for param in inspect.signature(self.value_function_upper_estimator_callback).parameters.values() if param.name == 'trajectory_history'):
                sim_reward = np.min([sim_reward, self.value_function_upper_estimator_callback(node, trajectory_history = trajectory_history)])
            else:
                sim_reward = np.min([sim_reward, self.value_function_upper_estimator_callback(node)])
        
        return sim_reward    
    
    def simulate_by_simulation(self, node, mc_sim_iter = 10):
        depth = 0
        current_state = node.state
        discount = self.discount_factor
        
        trajectory_history = []
        reward_history = []
        for i in range(mc_sim_iter):
            state_history = [current_state]
            while True:
                valid_actions = self.mdp.actions(current_state)
                random_action = random.choice(valid_actions)
                new_state = self.mdp.transition(current_state, random_action)
                state_history.append(new_state)

                if self.mdp.is_terminal(new_state):
                    reward = self.mdp.reward(current_state, random_action, new_state) * discount
                    if self.verbose:
                        print(f"-> Terminal state reached: {reward}")
                    reward_history.append(reward)
                    trajectory_history.append(state_history)
                    break

                current_state = new_state
                depth += 1
                discount *= self.discount_factor

                # statefulsolver, state should have a terminal check, in the state itself (ie last port in the schedule)
                if depth > self.simulation_depth_limit:
                    reward = self.mdp.reward(current_state, random_action, new_state) * discount
                    if self.verbose:
                        print(f"-> Depth limit reached: {reward}")
                    reward_history.append(reward)
                    trajectory_history.append(state_history)
                    break
        expected_reward = np.mean(reward_history)
        return expected_reward, trajectory_history

    def backpropagate(self, node: StateNode[TState, TAction], reward: float) -> None:
        current_state_node = node
        current_reward = reward

        while current_state_node != None:
            current_state_node.max_reward = max(current_reward, current_state_node.max_reward)
            current_state_node.reward += current_reward
            current_state_node.n += 1

            current_state_node = current_state_node.parent
            current_reward *= self.discount_factor

    def create_node(self, parent: Optional[StateNode[TState, TAction]], inducing_action: Optional[TAction],
                    state: TState, number_of_visits=0) -> StateNode[TState, TAction]:
        
        valid_actions = self.mdp.actions(state)
        is_terminal = self.mdp.is_terminal(state)
        state_node = StateNode(parent, inducing_action, state, valid_actions, is_terminal)

        if parent != None:
            parent.add_child(state_node)

        return state_node
