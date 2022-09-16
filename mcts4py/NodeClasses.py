from enum import Enum
# from Node import *
import numpy as np
from abc import ABC, abstractmethod
from typing import TypeVar, Generic


# Sample Enum definition,
# class Event(Enum):
#     DEMAND_ARRIVAL = 0
#     SUPPLY_ARRIVAL = 1
#     SUPPLIER_ON = 2
#     SUPPLIER_OFF = 3
#     NO_EVENT = 4

ActionType = TypeVar('ActionType')
StateType = TypeVar('StateType')

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
        raise NotImplementedError("Please Implement addChild method")

    @abstractmethod
    def getChildren(self):
        raise NotImplementedError("Please Implement getChildren method")

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
class StateNode():

    def __init__(self,
        parent = None,
        inducing_action = None,
        state = None,
        valid_actions = [],
        is_terminal = False):

        self.parent = parent
        self.inducing_action = inducing_action

        self.depth = 0 if parent == None else parent.depth + 1

        self.n = 0
        self.reward = 0.0
        self.max_reward = 0.0

        self.inducing_action = inducing_action
        self.children = dict()
        self.state = state
        self.is_terminal = is_terminal

    def addChild(self, child):
        if child.inducing_action == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing_action in self.children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self.children[repr(child.inducing_action)] = child

    def getChildren(self, action):
        if action == None:
            return self.children.values() # return states
        else:
            if action in self.children:
                return [self.children[action]] # returns a [state]
            else:
                return []
    
    def exploredActions(self):
        return self.children.keys()
    
    def __str__(self):
        return 'State: {}, Max Reward: {} '.format(str(self.state), str(self.max_reward)) 
    
    # fun exploredActions(): Collection<ActionType> {
    #     return children.keys
    # }

