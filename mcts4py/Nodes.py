from abc import ABC, abstractmethod
from typing import Generic, MutableMapping, Optional, TypeVar
from mcts4py.Types import TAction, TState, TRandom
import math

TNode = TypeVar("TNode", bound="Node")


class Node(ABC, Generic[TAction]):

    def __init__(self: TNode,
        parent: Optional[TNode] = None,
        inducing_action: Optional[TAction] = None):

        self.inducing_action = inducing_action
        self.depth = 0 if parent is None else parent.depth + 1
        self.n = 0
        self.reward = 0.0
        self.max_reward = 0.0

    @abstractmethod
    def get_parent(self: TNode) -> Optional[TNode]:
        raise NotImplementedError

    @abstractmethod
    def add_child(self: TNode, child: TNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_children(self: TNode) -> list[TNode]:
        raise NotImplementedError

    @abstractmethod
    def get_children_of_action(self: TNode, action: TAction) -> list[TNode]:
        raise NotImplementedError


TStateNode = TypeVar("TStateNode", bound="StateNode")


class StateNode(Generic[TState, TAction], Node[TAction]):

    def __init__(self,
        parent: Optional[TStateNode] = None,
        inducing_action: Optional[TAction] = None,
        state: Optional[TState] = None,
        valid_actions: list[TAction] = [],
        is_terminal: bool = False):

        self.parent = parent
        self.children: MutableMapping[TAction, TStateNode] = dict()
        self.state = state
        self.valid_actions = valid_actions
        self.is_terminal = is_terminal

        super().__init__(parent, inducing_action)

    def get_parent(self: TStateNode) -> Optional[TStateNode]:
        return self.parent

    def add_child(self: TStateNode, child: TStateNode) -> None:
        if child.inducing_action == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing_action in self.children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self.children[child.inducing_action] = child

    def get_children(self: TStateNode) -> list[TStateNode]:
        return list(self.children.values())

    def get_children_of_action(self: TStateNode, action:TAction) -> list[TStateNode]:
        if action in self.children:
            return [self.children[action]]
        else:
            return []

    def explored_actions(self) -> list[TAction]:
        return list(self.children.keys())

    def __str__(self):
        return f"State: {self.state} Inducing Action: {self.inducing_action}"


TActionNode = TypeVar("TActionNode", bound="ActionNode")


class ActionNode(Generic[TState, TAction], Node[TAction]):

    def __init__(self,
        parent: Optional[TActionNode] = None,
        inducing_action: Optional[TAction] = None):

        self.parent = parent
        self.__children: list[TActionNode] = []
        self.__state: Optional[TState] = None
        self.__valid_actions: Optional[list[TAction]] = None

        super().__init__(parent, inducing_action)

    @property
    def state(self) -> TState:
        if self.__state == None:
            raise RuntimeError(f"Simulation not run at depth: {self.depth}")
        return self.__state

    @state.setter
    def state(self, value: TState) -> None:
        self.__state = value

    @property
    def valid_actions(self) -> list[TAction]:
        if self.__valid_actions == None:
            raise RuntimeError(f"Simulation not run")
        return self.__valid_actions

    @valid_actions.setter
    def valid_actions(self, value: list[TAction]) -> None:
        self.__valid_actions = value

    def get_parent(self: TActionNode) -> Optional[TActionNode]:
        return self.parent

    def add_child(self: TActionNode, child: TActionNode) -> None:
        self.__children.append(child)

    def get_children(self: TActionNode) -> list[TActionNode]:
        return self.__children

    def get_children_of_action(self: TActionNode, action: TAction) -> Optional[TActionNode]:
        return [child for child in self.__children if child.inducing_action == action]

    def __str__(self):
        return f"Action: {self.inducing_action}"


TRandomNode = TypeVar("TRandomNode", bound="RandomNode")
TDecisionNode = TypeVar("TDecisionNode", bound="DecisionNode")

class NewNode(ABC, Generic[TRandom, TAction]):
    def __init__(self, parent, inducing, state=None,
                 is_terminal: bool = False):
        self.inducing = inducing
        self.parent = parent
        self._state = state
        self.is_terminal = is_terminal
        self.n = 0
        self.max_reward = -math.inf
        self.reward = 0.0
        self._depth = 0.0

    @property
    def __parent(self: TNode) -> Optional[TNode]:
        return self.parent

    @property
    def state(self) -> TState:
        if self._state == None:
            raise RuntimeError(f"Simulation not run at depth: {self.depth}")
        return self._state

    @state.setter
    def state(self, value: TState) -> None:
        self._state = value


    def get_parent(self: TNode) -> Optional[TNode]:
        raise self.parent

    def add_child(self: TNode, child) -> None:
        if child.inducing == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing in self.children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self.children[child.inducing] = child

    def get_children(self: TNode) -> list[TNode]:
        return list(self.children.values())

    @property
    def depth(self) -> float:
        return self._depth

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth = value




class RandomNode(Generic[TAction, TRandom, TDecisionNode], NewNode[TAction, TRandom]):

    def __init__(self,
                 parent: Optional[TStateNode] = None,
                 inducing: Optional[TAction] = None,
                 state: Optional[TState] = None,
                 is_terminal: bool = False):
        self.children: MutableMapping[TAction, TRandom] = dict()
        super().__init__(parent, inducing, state, is_terminal)

    def get_parent(self: TRandomNode) -> Optional[TDecisionNode]:
        return self.parent

    def add_child(self: TRandomNode, child: TDecisionNode) -> None:
        if child.inducing == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing in self.children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self.children[child.inducing] = child

    def get_children(self: TRandomNode) -> list[TDecisionNode]:
        return list(self.children.values())

    def __str__(self):
        return f"Action: {self.parent}"

    # def __eq__(self, other):
    #     return (self.price == other.price) and (self.state == other.state)


class DecisionNode(Generic[TAction, TRandom], NewNode[TAction, TRandom]):

    def __init__(self,
                 parent: Optional[TRandomNode] = None,
                 inducing: Optional[TRandom] = None,
                 state: Optional[TState] = None,
                 valid_actions: Optional[list[TAction]] = None,
                 is_terminal: bool = False):
        self.parent = parent
        self.children: MutableMapping[TAction, TStateNode] = dict()
        self._valid_actions: Optional[list[TAction]] = valid_actions
        super().__init__(parent, inducing, state, is_terminal)

    def explored_actions(self) -> list[TAction]:
        return list(self.children.keys())

    def get_parent(self: TDecisionNode) -> Optional[TRandomNode]:
        return self.parent

    def add_child(self: TDecisionNode, child: TRandomNode) -> None:
        if child.inducing == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing in self.children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self.children[child.inducing] = child

    def get_children(self: TDecisionNode) -> list[TRandomNode]:
        return list(self.children.values())

    @property
    def valid_actions(self) -> list[TAction]:
        if self._valid_actions == None:
            raise RuntimeError(f"Simulation not run")
        return self._valid_actions

    @valid_actions.setter
    def valid_actions(self, value: list[TAction]) -> None:
        self._valid_actions = value


    def __str__(self):
        return f"Action: {self.inducing}, Price: {self.price}"

    def __eq__(self, other):
        if other is None:
            return False
        return (self.state == other.state)
