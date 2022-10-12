from abc import ABC, abstractmethod

class MDP(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def transition(self, state, action): # return a state
        pass

    @abstractmethod
    def reward(self, previous_state, action, state) -> float:
        pass

    @abstractmethod
    def initialState(self): # return a state
        pass

    @abstractmethod
    def actions(self, state) -> bool:
        raise NotImplementedError

    @abstractmethod
    def isTerminal(self, state) -> iter:
        pass

