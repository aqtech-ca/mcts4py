from abc import ABC, abstractmethod

class MDP(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transition(self):
        pass

    @abstractmethod
    def reward(self):
        pass

    @abstractmethod
    def initialState(self):
        pass
    
    @abstractmethod
    def actions(self):
        pass

    @abstractmethod
    def isTerminal(self):
        pass

