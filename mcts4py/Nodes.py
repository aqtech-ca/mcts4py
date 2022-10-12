from abc import ABC, abstractmethod
from typing import Generic, MutableMapping, Optional, TypeVar
from mcts4py.Types import TAction, TState

TNode = TypeVar("TNode", bound="Node")

class Node(ABC, Generic[TAction]):

    def __init__(self: TNode,
        parent: Optional[TNode] = None,
        inducing_action: Optional[TAction] = None):

        self.parent: Optional[TNode] = parent
        self.inducing_action: Optional[TAction] = inducing_action

        self.depth: int = 0 if parent is None else parent.depth + 1

        self.n: int = 0
        self.reward: float = 0.0
        self.max_reward: float = 0.0

    @abstractmethod
    def add_child(self: TNode, child: TNode) -> None:
        raise NotImplementedError()

    @abstractmethod
    def get_children(self: TNode, action: Optional[TAction] = None) -> list[TNode]:
        raise NotImplementedError()

TStateNode = TypeVar("TStateNode", bound="StateNode")

class StateNode(Generic[TState, TAction], Node[TAction]):

    def __init__(self,
        parent: Optional[TStateNode] = None,
        inducing_action: Optional[TAction] = None,
        state: Optional[TState] = None,
        valid_actions: list[TAction] = [],
        is_terminal: bool = False):

        self.children: MutableMapping[TAction, TStateNode] = dict()
        self.state: Optional[TState] = state
        self.valid_actions: list[TAction] = valid_actions
        self.is_terminal: bool = is_terminal

        super().__init__(parent, inducing_action)

    def add_child(self: TStateNode, child: TStateNode) -> None:
        if child.inducing_action == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing_action in self.children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self.children[child.inducing_action] = child

    def get_children(self: TStateNode, action: Optional[TAction]) -> list[TStateNode]:
        if action == None:
            return list(self.children.values())
        else:
            if action in self.children:
                return [self.children[action]]
            else:
                return []

    def exploredActions(self) -> list[TAction]:
        return list(self.children.keys())

    def __str__(self):
        return "State: {}, Max Reward: {} ".format(str(self.state), str(self.max_reward))

