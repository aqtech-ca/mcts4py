from abc import ABC, abstractmethod
from typing import Generic, Optional

from mcts4py.Types import TAction, TState

class MDP(ABC, Generic[TState, TAction]):

    @abstractmethod
    def transition(self, state: TState, action: TAction) -> TState:
        raise NotImplementedError

    @abstractmethod
    def reward(self, previous_state: Optional[TState], action: Optional[TAction], state: TState) -> float:
        raise NotImplementedError

    @abstractmethod
    def initial_state(self) -> TState:
        raise NotImplementedError

    @abstractmethod
    def actions(self, state: TState) -> list[TAction]:
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: TState) -> bool:
        raise NotImplementedError

