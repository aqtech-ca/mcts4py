import random
from mcts4py.Types import *
from mcts4py.Solver import *
from mcts4py.MDP import *
from mcts4py.QValueEstimator import *

random = random.Random(0)
import copy
import inspect


accepts_arguments = lambda func, num_args: len(inspect.signature(func).parameters) == num_args


class StatefulSolverMENTS(MCTSSolver[TAction, NewNode[TRandom, TAction], TRandom], Generic[TState, TAction, TRandom]):

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
                 value_function_lower_estimator_callback = None,
                 lambda_temp_callback=exponential_decay):
        self.mdp = mdp
        self.discount_factor = discount_factor
        self.simulation_depth_limit = simulation_depth_limit
        self.value_function_estimator_callback = value_function_estimator_callback
        self.alpha_value = alpha_value
        self.value_clipping = value_clipping
        self.value_function_upper_estimator_callback = value_function_upper_estimator_callback
        self.value_function_lower_estimator_callback = value_function_lower_estimator_callback

        self.q_estimator = QValueEstimator(alpha=alpha_value, lambda_temp_callback=lambda_temp_callback)

        self.ments_value_tracker = []
        
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
            action_probs_dict = self.q_estimator.get_softmax_prob_multinom(node.state, current_node.valid_actions)
            _, action_index = self.q_estimator.draw_from_multinomial(action_probs_dict)
            current_node = current_node.children[action_index]
            pass
            
            # current_node = max(current_node.children, key=lambda c: self.calculate_uct(c))

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

            state_children_iter = current_state_node.children
            current_state_node.n += 1

            ### Check this over
            # get all taken actions to update q table
            all_poss_actions = [sc.inducing_action for sc in state_children_iter]
            action_visit_dict = {}
            
            for visited_action in all_poss_actions:
                for state_child in state_children_iter:
                    if state_child.inducing_action == visited_action:
                        if repr(visited_action) not in action_visit_dict:
                            action_visit_dict[repr(visited_action)] = 1
                        else:
                            action_visit_dict[repr(visited_action)] += 1
            
            reward_to_go_dict = {}
            for visited_action in all_poss_actions:
                for state_child in state_children_iter:
                    if state_child.inducing_action == visited_action:
                        reward_to_go_term = state_child.n / action_visit_dict[repr(state_child.inducing_action)] * self.q_estimator.get_state_value(state_child.state)
                        if repr(visited_action) not in reward_to_go_dict:
                            reward_to_go_dict[repr(visited_action)] = {"reward_to_go": reward_to_go_term, "action_obj": visited_action}
                        else:
                            reward_to_go_dict[repr(visited_action)]["reward_to_go"] += reward_to_go_term
            
            # Q and Value Update: ### Check this over...
            for action_repr in reward_to_go_dict.keys():
                self.q_estimator.update_q_value(current_state_node.state, reward_to_go_dict[action_repr]["action_obj"], reward, reward_to_go_dict[action_repr]["reward_to_go"], discount_factor=self.discount_factor)
            self.q_estimator.update_state_value(current_state_node.state, all_poss_actions)
            current_state_node.ments_value = self.q_estimator.get_state_value(current_state_node.state)

            if current_state_node.parent is None:
                self.ments_value_tracker.append(current_state_node.ments_value)

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
