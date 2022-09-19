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
        # self.mdp.reset() # no reset in the kotlin version
        root_node = self.root()
        best = self.select(root_node)

        if self.verbose:
            print("Expanding")
            self.displayNode(best)

        if best.state is None:
            print("hehe")
        
        expanded = self.expand(best)
        simulated_reward = self.simulate(expanded)

        print("simulated reward: " + str(simulated_reward))

        self.backpropagate(expanded, simulated_reward)
    
    def calculateUCT(self, node):
        parentN = node.parent.n if node.parent != None else node.n
        return self.calculateUCTLongForm(parentN, node.n, node.reward, self.exploration_constant)
    
    def calculateUCTLongForm(self, parentN, n, reward, exploration_constant):
        return reward/n + exploration_constant * np.sqrt(np.log(parentN )/n)
    
    def extractOptimalAction(self):
        if self.root().getChildren(None) != None:
            visit_counts = [x.n for x in self.root().getChildren(None)]
            max_i = np.argmax(visit_counts) # max(self.root().getChildren(), key=attrgetter('n'))
            inducing_actions = [x.inducing_action for x in self.root().getChildren(None)]
            return inducing_actions[max_i]
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
            return 
        
        if node.depth > depth_limit:
            return None

        line = str(indent) + str(node.state) + 'n: {}, reward: {}, UCT: {}'.format(str(node.n), str(node.reward), str(self.calculateUCT(node))) 
        print(line)

        children = node.getChildren(None)

        # cc = list(children)

        # if cc == []:
        #     print("hehe")

        if None in children:
            return 
        
        child_states = list(children)

        ccc = child_states[:-1]
        for child in child_states[:-1]:
            indent = self.generateIndent(indent) + " ├"
            self.displayTreeLongForm(depth_limit, child, indent)
        
        self.displayTreeLongForm(depth_limit, child_states[-1], self.generateIndent(indent) + " └")

        # if len(child_states) > 1:
        #     self.displayTreeLongForm(depth_limit, child_states[-1], self.generateIndent(indent) + " └")

    def generateIndent(self, indent: str):
        return indent.replace('├', '│').replace('└', ' ')

