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
        return node
    
    def expand(self, node):
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
