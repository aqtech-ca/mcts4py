from mcts4py.NodeClasses import *
from abc import ABC, abstractmethod
import numpy as np
from operator import attrgetter
import typing

# Solver abstract class
class Solver():
    def __init__(self, exploration_constant = 0.9, verbose = False):
        self.exploration_constant = exploration_constant
        self.verbose = verbose

    @abstractmethod
    def root(self):
        return 
        # raise NotImplementedError("Please Implement root")
    
    @abstractmethod
    def select(self):
        raise NotImplementedError
    
    @abstractmethod
    def expand(self):
        raise NotImplementedError
    
    @abstractmethod
    def simulate(self):
        raise NotImplementedError
    
    @abstractmethod
    def backpropagate(self):
        raise NotImplementedError
    
    def runTreeSearch(self, iters: int):
        for i in range(iters):
            if self.verbose:
                print("New iteration: " + str(i))
                print("=======================")
            self.runtTreeSearchIteration()
    
    def runtTreeSearchIteration(self):
        root_node = self.root()
        best = self.select(root_node)

        if self.verbose:
            print("Expanding")
            self.displayNode(best)
        
        expanded = self.expand(best)
        simulated_reward = self.simulate(expanded)

        print("simulated reward: " + str(simulated_reward))

        self.backpropagate(expanded, simulated_reward)
    
    def calculateUCT(self, node):
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculateUCTLongForm(parentN, node.n, node.reward, self.exploration_constant)
    
    def calculateUCTLongForm(parentN, n, reward, exploration_constant):
        return reward/n + exploration_constant * np.sqrt(np.log(parentN )/n)


    def extractOptinalAction(self):
        if self.root().getChildren() != None:
            max_c = max(self.root().getChildren(), key=attrgetter('n'))
            return max_c
        else:
            return None

    def displayNode(self, node):
        if node.parent != None:
            self.displayNode(node.parent)
        
        if node.depth > 0:
            print(" " * (node.depth - 1)*2 + " └")
        
        print(str(node))
    
    def displayTree(self, depth_limit: int = 3):
        self.displayTreeLongForm(depth_limit, self.root(), "")
    
    def displayTreeLongForm(self, depth_limit: int, node: typing.Union[Node, None], indent: str):
        if node == None:
            return None
        
        if node.depth > depth_limit:
            return None

        line = str(indent) + str(node) + 'n: {}, reward: {}, UCT: {}'.format(str(node.n), str(node.reward), str(self.calculateUCT(node)) ) 
        print(line)

        children = node.getChildren()

        if None in children:
            return None
        
        for child in children[:-1]:
            self.displayTreeLongForm(depth_limit, child, self.generateIndent() + " ├")
        self.displayTreeLongForm(depth_limit, children[-1], self.generateIndent() + " └")

    def generateIndent(self, indent: str):
        return indent.replace('├', '│').replace('└', ' ')

