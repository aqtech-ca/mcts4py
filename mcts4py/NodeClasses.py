from enum import Enum
# from Node import *
import numpy as np
from abc import ABC, abstractmethod


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
        inducing_action = None):

        self.parent = parent
        self.inducing_action = inducing_action

        self.depth = 0 if parent == None else parent.depth + 1

        self.n = 0
        self.reward = 0.0
        self.max_reward = 0.0

    @abstractmethod
    def addChild(self):
        # pass
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def getChildren(self):
        # pass
        raise NotImplementedError("Please Implement this method")

# Stateful action node
class ActionNode(Node):
    def __init__(self, 
        value,
        parent = None,
        children = []):

        self.value = value
        self.parent = parent
        self.children = children

    def addChild(self, child):
        self.children.append(child)

    def getChildren(self, action):
        if action == None:
            return self.children
        else:
            return filter(lambda x: x.inducing_action == action, self.children) 
        # return filter(lambda x: x.parentState == state, self.children)
    
    def __str__(self):
        return 'Action: {}, Max Reward: {}'.format(str(self.inducing_action), str(self.max_reward)) 

# Stateful state node
# class StateNode(Node):

    def __init__(self, 
        value,
        state,
        inducing_action = None,
        valid_actions = [],
        is_terminal = False):

        self.value = value
        self.inducing_action = inducing_action
        self.children = valid_actions
        self.state = state

    def addChild(self, child):
        if child.inducing_action == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing_action in self.children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self.children[repr(child.inducing_action)] = child

    def getChildren(self, action):
        if action == None:
            return self.children.values()
        else:
            child = self.children[action]
            if child == None:
                return []
            else:
                return [child]
    
    def exploredActions(self):
        return self.children.keys()
    
    def __str__(self):
        return 'State: {}, Max Reward: {}'.format(str(self.state), str(self.max_reward)) 
    
    # fun exploredActions(): Collection<ActionType> {
    #     return children.keys
    # }

