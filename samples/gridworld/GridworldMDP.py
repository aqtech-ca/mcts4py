from __future__ import annotations
from enum import Enum
from mcts4py.MDP import *
import random
import numpy as np
from mcts4py.StatefulSolver import *

class GridworldAction(Enum):
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    def __str__(self) -> str:
        if self == GridworldAction.UP:
            return "↑"
        elif self == GridworldAction.DOWN:
            return "↓"
        elif self == GridworldAction.LEFT:
            return "←"
        elif self == GridworldAction.RIGHT:
            return "→"


class GridworldPosition():
    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, GridworldPosition):
            return __o.x == self.x and __o.y == self.y
        return False

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __hash__(self) -> int:
        return hash(self.__str__())


class GridworldReward(GridworldPosition):
    def __init__(self, x: int, y: int, value: float):
        super().__init__(x, y)
        self.value = value


class GridworldState(GridworldPosition):
    def __init__(self, x: int, y: int, is_terminal: bool):
        super().__init__(x, y)
        self.is_terminal = is_terminal

    def is_neighbour_valid(self, action: GridworldAction, x_size: int, y_size: int) -> bool:
        if action == GridworldAction.UP:
            return self.x > 0
        elif action == GridworldAction.DOWN:
            return self.x < x_size-1
        elif action == GridworldAction.LEFT:
            return self.y > 0
        elif action == GridworldAction.RIGHT:
            return self.y < y_size-1

    def resolve_neighbour(self, action: GridworldAction, x_size: int, y_size: int) -> Optional[GridworldState]:
        if self.is_neighbour_valid(action, x_size, y_size):
            if action == GridworldAction.UP:
                return GridworldState(self.x-1, self.y, False)
            elif action == GridworldAction.DOWN:
                return GridworldState(self.x+1, self.y, False)
            elif action == GridworldAction.LEFT:
                return GridworldState(self.x, self.y-1, False)
            elif action == GridworldAction.RIGHT:
                return GridworldState(self.x, self.y+1, False)
        else:
            return None


class GridworldMDP(MDP[GridworldState, GridworldAction]):

    def __init__(self,
        x_size: int,
        y_size: int,
        rewards: list[GridworldReward],
        transition_probability: float,
        starting_location: GridworldState = GridworldState(0, 0, False)):

        self.x_size = x_size
        self.y_size = y_size
        self.rewards = rewards
        self.transition_probability = transition_probability
        self.starting_location = starting_location

    def initial_state(self) -> GridworldState:
        return self.starting_location

    def is_terminal(self, state: GridworldState) -> bool:
        for r in self.rewards:
            if state == r:
                return True
        return False

    def reward(self, previous_state: Optional[GridworldState], action: Optional[GridworldAction], state: GridworldState) -> float:
        for r in self.rewards:
            if state == r:
                return r.value
        return 0.0

    def transition(self, state: GridworldState, action: GridworldAction) -> GridworldState:
        if state.is_terminal:
            return state

        for r in self.rewards:
            if state == r:
                return r

        # Resolve neighbour and return current state if neighbour is out of bounds
        target_neighbour = state.resolve_neighbour(action, self.x_size, self.y_size)
        if target_neighbour is None:
            return state

        # Transition to target neighbour with transition probability
        if np.random.uniform() < self.transition_probability:
            return target_neighbour
        else:
            remaining_actions = [a for a in self.actions(state) if a != action]
            non_target_neighbours = [state.resolve_neighbour(a, self.x_size, self.y_size) for a in remaining_actions]

            if len(non_target_neighbours) > 0:
                return random.choice(non_target_neighbours)
            else:
                raise Exception("No valid neighbours exist")

    def actions(self, state: GridworldState) -> list[GridworldAction]:
        if isinstance(state, StateNode):
            state = state.state

        return [a for a in GridworldAction if state.is_neighbour_valid(a, self.x_size, self.y_size)]

    # Utilities

    def visualize_state(self) -> None:
        state_array = np.full([self.x_size, self.y_size], "-")
        state_array[self.starting_location.x][self.starting_location.y] = "S"

        for r in self.rewards:
            if r.value >= 0.0:
                state_array[r.x][r.y] = "*"
            else:
                state_array[r.x][r.y] = "X"

        for row in state_array:
            print(row)



