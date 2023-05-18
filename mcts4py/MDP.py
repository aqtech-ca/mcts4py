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

class MDPWidening(MDP):
    @abstractmethod
    def widening_actions(self, state: TState, number_of_visits: int, iteration_number: int, max_iteration_number: int) -> list[TAction]:
        """
        If you want to apply progressive widening, you need to define how to progressively widen the actions
        :return:
        """
        pass