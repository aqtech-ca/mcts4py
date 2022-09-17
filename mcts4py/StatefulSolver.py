from mcts4py.NodeClasses import *
from mcts4py.Solver import *

# just implement the stateful solver
class StatefulSolver(Solver):

    def __init__(self, 
        mdp,
        discount_factor = 1.0,
        simulation_depth_limit = 200,
        verbose = False,
        exploration_constant = 0.9):

        self.mdp = mdp
        self.discount_factor = discount_factor
        self.simulation_depth_limit = simulation_depth_limit
        self.verbose = verbose
        self.root_node = self.createNode(None, None, mdp.initialState())
        self.exploration_constant = exploration_constant

    def root(self):
        return self.root_node
    
    def select(self, node: ActionNode):
        # if len(node.getChildren()) == 0:
        #     return node

        current_node = node
        # self.simulateActions(node)

        while True:
            if self.mdp.isTerminal(current_node.state):
                return current_node
            
            current_children = current_node.getChildren(None)
            explored_actions = [x.inducing_action for x in current_children]

            valid_actions = self.mdp.actions(current_node.state)
            if len(list(set(valid_actions) - set(explored_actions))) > 0:
                return current_node

            max_ind = np.argmax([self.calculateUCT(a) for a in current_children]) 
            current_node = list(current_children)[max_ind] # throw null

            # self.simulateActions(self.current_node)
    
    
    def expand(self, node):
        if self.mdp.isTerminal(node.state):
            return node
        
        valid_actions = self.mdp.actions(node.state)
        inducing_actions = [x.getChildren(None) for x in node.getChildren(None)]
        unexplored_actions = list(set(set(valid_actions) - set(inducing_actions)))

        ind = np.random.choice(len(unexplored_actions), 1, replace = False)[0]
        action_taken = unexplored_actions[ind]

        new_state = self.mdp.transition(node.state, action_taken)
        new_node = self.createNode(node, action_taken, new_state)
        return new_node
    
    def simulate(self, node):
        print("Run simulation")

        if self.mdp.isTerminal(node.state):
            print("Terminal state reached!")
            parent_state = node.parent.state if node.parent != None else None
            return self.mdp.reward(parent_state, node.inducing_action, node.state)

        depth = 0
        current_state = node.state
        discount = self.discount_factor

        while True:
            valid_actions = self.mdp.actions(current_state)
            random_action = np.random.choice(valid_actions, 1)
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
        return 1.0
    
    def backpropagate(self, node, reward):
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
    def simulateActions(self, node: ActionNode):
        parent = node.parent

        if parent == None:
            initial_state = self.mdp.initialState()
            node.state = initial_state
            node.valid_actions = self.mdp.actions(initial_state)
            return True
        
        parent_state = parent.state
        parent_action = node.inducing_action if node.inducing_action != None else None

        state = self.mdp.transition(parent_state, parent_action)
        node.state = state
        node.valid_actions = self.mdp.actions(state)

        return True
    
    def createNode(self, parent, inducing_action, state): # return state node
        valid_actions = self.mdp.actions(state)
        is_terminal = self.mdp.isTerminal(state)
        state_node = StateNode(parent = parent, inducing_action = inducing_action, state = state, valid_actions = valid_actions, is_terminal = is_terminal)

        if parent != None:
            parent.addChild(state_node) # parent?.addChild(stateNode) kotlin version

        return state_node

