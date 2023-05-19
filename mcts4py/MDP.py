from abc import ABC, abstractmethod
from typing import Generic, Optional

from mcts4py.Types import TAction, TState, TRandom


class MDP(ABC, Generic[TState, TAction]):

    @abstractmethod
    def transition(self, state: TState, random: TRandom) -> TState:
        raise NotImplementedError

    @abstractmethod
    def reward(self, previous_state: Optional[TState], action: Optional[TAction]) -> float:
        raise NotImplementedError

    @abstractmethod
    def initial_state(self) -> TState:
        raise NotImplementedError

    # @abstractmethod
    # def actions(self, state: TState) -> list[TAction]:
    #     raise NotImplementedError

    @abstractmethod
    def is_terminal(self, state: TState) -> bool:
        raise NotImplementedError

    @abstractmethod
    def actions(self, state: TState, state_visit, iteration_number=0, max_iteration_number=0, dpw_exploration=1,
                dpw_alpha=1, min_action = False) -> list[TAction]:
        raise NotImplementedError
