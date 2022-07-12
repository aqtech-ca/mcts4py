from NodeClasses import *

# just implement the stateful solver

class StatefulSolver():
    def __init__(self, 
        mdp,
        value = 0.0,
        root_node = StateNode(parent = None, inducingAction = None)):

        self.value = value
        self.mdp = mdp

    
    def select(self, node: ActionNode):

        if len(node.getChildren()) == 0:
            return node

        current_node = node
        self.simulateActions(node)

        while True:
            if self.mdp.isTerminal(self.current_node.state):
                return self.current_node
            
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
        return 1.0
    
    def backpropagate(self, iters):
        return True
    
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
