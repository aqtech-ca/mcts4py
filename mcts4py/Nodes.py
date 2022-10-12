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
    def get_child_of_action(self: TNode, action: TAction) -> Optional[TNode]:
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

    def get_child_of_action(self: TStateNode, action:TAction) -> Optional[TStateNode]:
        if action in self.children:
            return self.children[action]
        else:
            return None

    def explored_actions(self) -> list[TAction]:
        return list(self.children.keys())

    def __str__(self):
        return f"State: {self.state} Inducing Action: {self.inducing_action}"

