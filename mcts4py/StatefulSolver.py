from mcts4py.NodeClasses import *
from mcts4py.Solver import *
import random

# just implement the stateful solver
class StatefulSolver(Solver):

    def __init__(self,
        mdp,
        discount_factor = 0.9,
        simulation_depth_limit = 5,
        verbose = False,
        exploration_constant = 0.4):

        self.mdp = mdp
        self.discount_factor = discount_factor
        self.simulation_depth_limit = simulation_depth_limit
        self.verbose = verbose
        self.root_node = self.createNode(None, None, mdp.initialState())
        self.exploration_constant = exploration_constant

    def root(self):
        return self.root_node

    # forcibly expore all actions from root:
    # def initialExploration(self):
    #     root_node = self.root()
    #     children = root_node.getChildren(None)

    #     node = root_node

    #     for action_taken in self.mdp.actions(root_node.state):
    #         new_state = self.mdp.transition(root_node.state, action_taken)
    #         new_node = self.createNode(root_node, action_taken, new_state)

    #         self.backpropagate(new_node, 0.9)
    #     return root_node

    def select(self, node: ActionNode):
        # if len(node.getChildren()) == 0:
        #     return node

        current_node = node

        while True:
            if self.mdp.isTerminal(current_node.state):
                return None

            current_children = current_node.getChildren(None)
            explored_actions = [x.inducing_action for x in current_children]

            valid_actions = self.mdp.actions(current_node.state)
            if len(list(set(valid_actions) - set(explored_actions))) > 0:
                return current_node

            current_children_list = list(current_children)

            # if np.random.uniform() < 0.7:
            #     return random.choice(current_children_list)

            if len(current_children) > 1:
                max_ind = np.argmax([self.calculateUCT(a) for a in current_children])
                current_node = current_children_list[max_ind] # throw null
            else:
                return current_node


    def expand(self, node):
        if self.mdp.isTerminal(node.state):
            return node

        valid_actions = self.mdp.actions(node.state)
        inducing_actions = [x.inducing_action for x in node.getChildren(None)]
        unexplored_actions = list(set(set(valid_actions) - set(inducing_actions)))

        if len(unexplored_actions) < 1:
            return None

        ind = np.random.choice(len(unexplored_actions), 1, replace = False)[0]
        action_taken = unexplored_actions[ind]

        new_state = self.mdp.transition(node.state, action_taken)
        new_node = self.createNode(node, action_taken, new_state)
        return new_node

    def simulate(self, node):
        print("Run simulation")

        if node is None:
            return None


        if self.mdp.isTerminal(node.state):
            print("Terminal state reached!")
            parent_state = node.parent.state if node.parent != None else None
            if parent_state is not None:
                return self.mdp.reward(parent_state, node.inducing_action, node.state)
            else:
                return 0.0

        depth = 0
        current_state = node.state
        discount = self.discount_factor

        while True:
            #### HACKY ISSUE ####
            if current_state is None:
                break

            valid_actions = self.mdp.actions(current_state)
            random_action = np.random.choice(valid_actions, 1)[0]

            # #### HACKY FIX ####
            # temp_state = self.mdp.transition(current_state, random_action)
            # while temp_state is None:
            #     temp_state = self.mdp.transition(current_state, random_action)
            # new_state = temp_state

            new_state = self.mdp.transition(current_state, random_action)

            if self.mdp.isTerminal(new_state):
                reward = self.mdp.reward(current_state, random_action, new_state) * discount
                return reward

            current_state = new_state
            depth += 1
            discount *= self.discount_factor

            if depth > self.simulation_depth_limit:
                reward = self.mdp.reward(current_state, random_action, new_state) * discount
                if self.verbose:
                    print("-> Depth limit reached: " + str(reward))

                return reward
        return 0.0

    def backpropagate(self, node, reward):

        if node is None:
            return None

        current_state_node = node
        current_reward = reward

        while True:

            current_state_node.max_reward = max([current_reward, current_state_node.max_reward])
            current_state_node.reward = current_reward
            current_state_node.n += 1

            if current_state_node.parent != None:
                current_state_node = current_state_node.parent
                current_reward *= self.discount_factor
            else:
                break

        # return True

    # Utilities

    def createNode(self, parent, inducing_action, state): # return state node
        valid_actions = self.mdp.actions(state)
        is_terminal = self.mdp.isTerminal(state)
        state_node = StateNode(parent = parent, inducing_action = inducing_action, state = state, valid_actions = valid_actions, is_terminal = is_terminal)

        if parent != None:
            parent.addChild(state_node) # parent?.addChild(stateNode) kotlin version

        return state_node

