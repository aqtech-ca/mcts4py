from enum import Enum
from Node import *
import numpy as np

# Sample Enum definition,
# class Event(Enum):
#     DEMAND_ARRIVAL = 0
#     SUPPLY_ARRIVAL = 1
#     SUPPLIER_ON = 2
#     SUPPLIER_OFF = 3
#     NO_EVENT = 4

class Node():
    
    def __init__(self, 
        parent = None,
        inducingAction = None):

        self.parent = parent
        self.inducingAction = inducingAction

        self.depth = 0 if parent == None else parent.depth + 1

        self.n = 0
        self.reward = 0.0
        self.max_reward = 0.0

    def addChild(self):
        raise NotImplementedError("Please Implement this method")

    def getChildren(self):
        raise NotImplementedError("Please Implement this method")


# Stateful action node
class ActionNode(Node):

    def __init__(self, 
        value,
        parentState = None,
        children = []):

        self.value = value
        self.parentState = parentState
        self.children = children

        def addChild(self, child):
            self.children.append(child)

        def getChildren(self, state):
            return filter(lambda x: x.parentState == state, self.children)

# Stateful state node
class StateNode(Node):

    def __init__(self, 
        value,
        inducingAction = None,
        valid_actions = []):

        self.value = value
        self.inducingAction = inducingAction
        self.children = valid_actions

        def addChild(self, child):
            self.children.append(child)

        def getChildren(self, action):
            return filter(lambda x: x.inducingAction == action, self.children)

