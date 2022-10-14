from abc import ABC, abstractmethod
from typing import Generic, MutableMapping, Optional, TypeVar
from mcts4py.Types import TAction, TState


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