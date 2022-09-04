from mcts4py.NodeClasses import *
from mcts4py.Solver import *

# just implement the stateful solver
class StatefulSolver(Solver):

    def __init__(self, 
        mdp,
        discount_factor = 1.0,
        simulation_depth_limit = 200,
        verbose = False):

        self.mdp = mdp
        self.discount_factor = discount_factor
        self.simulation_depth_limit = simulation_depth_limit
        self.verbose = verbose
        self.root_node = self.createNode(None, None, mdp.initialState())

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
            
            current_children = self.current_node.getChildren()
            explored_actions = [x.inducingAction for x in current_children]

            if len(list(set(self.current_node.valid_actions) - set(explored_actions))) > 0:
                return self.current_node

            self.current_node = np.max([self.calculateUCT(a) for a in current_children]) # throw null

            self.simulateActions(self.current_node)
    
    
    def expand(self, node):
        if self.mdp.isTerminal(node.state):
            return node
        
        inducing_actions = [n.get_children() for n in node.get_children()]
        unexplored_actions = [c.inducing_actions for c in list(set(node.valid_actions) - set(inducing_actions))]

        ind = np.random.choice(len(unexplored_actions), 1, replace = False)
        action_taken = unexplored_actions[ind]

        new_node = ActionNode(node, action_taken)
        node.addChild(new_node)
        self.simulateActions(new_node)

        return node
    
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
        parent_action = node.inducingAction() if node.inducingAction() != None else None

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

