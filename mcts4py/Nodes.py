import math
import random
import time
from abc import ABC, abstractmethod
from typing import Generic, MutableMapping, Optional, TypeVar
from mcts4py.Types import TAction, TState, TRandom

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
        self._children: MutableMapping[TAction, TStateNode] = dict()
        self._state = state
        self.valid_actions = valid_actions
        self.is_terminal = is_terminal

        super().__init__(parent, inducing_action)

    def get_parent(self: TStateNode) -> Optional[TStateNode]:
        return self.parent

    def add_child(self: TStateNode, child: TStateNode) -> None:
        if child.inducing_action == None:
            raise Exception("Inducing action must be set on child")
        if child.inducing_action in self._children.keys():
            raise Exception("A child with the same inducing action has already been added")
        self._children[child.inducing_action] = child

    @property
    def children(self: TStateNode) -> list[TStateNode]:
        return list(self._children.values())

    @property
    def name(self):
        if self.inducing_action is not None:
            return f'{self.inducing_action.name}_{self._state.port}'
        else:
            return f'None_{self._state.port}'

    @property
    def state(self) -> TState:
        return self._state

    @state.setter
    def state(self, value: TState) -> None:
        self._state = value

    def get_children_of_action(self: TStateNode, action:TAction) -> list[TStateNode]:
        if action in self._children:
            return [self._children[action]]
        else:
            return []

    def explored_actions(self) -> list[TAction]:
        return list(self._children.keys())

    def __str__(self):
        return f"State: {self._state} Inducing Action: {self.inducing_action}"


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

    @property
    def children(self: TActionNode) -> list[TActionNode]:
        return self.__children

    def get_parent(self: TActionNode) -> Optional[TActionNode]:
        return self.parent

    def add_child(self: TActionNode, child: TActionNode) -> None:
        self.__children.append(child)

    def get_children_of_action(self: TActionNode, action: TAction) -> Optional[TActionNode]:
        return [child for child in self.__children if child.inducing_action == action]

    def __str__(self):
        return f"Action: {self.inducing_action}"


TRandomNode = TypeVar("TRandomNode", bound="RandomNode")
TDecisionNode = TypeVar("TDecisionNode", bound="DecisionNode")


class NewNode(ABC, Generic[TRandom, TAction]):
    def __init__(self, parent, inducing, state=None,
                 is_terminal: bool = False):
        self.inducing_action = inducing
        self._parent = parent
        self._state = state
        self.is_terminal = is_terminal
        self.n = 0
        self.max_reward = -math.inf
        self.reward = 0.0
        self._depth = 0.0

    @property
    def parent(self) -> Optional[TNode]:
        return self._parent

    @parent.setter
    def parent(self, value: TNode) -> None:
        self._parent = value

    @property
    def state(self) -> TState:
        if self._state == None:
            raise RuntimeError(f"Simulation not run at depth: {self.depth}")
        return self._state

    @state.setter
    def state(self, value: TState) -> None:
        self._state = value

    def add_child(self: TNode, child: TNode) -> None:
        raise NotImplementedError

    @property
    def children(self):
        raise NotImplementedError

    @property
    def depth(self) -> float:
        return self._depth

    @depth.setter
    def depth(self, value: float) -> None:
        self._depth = value

    def __eq__(self, other: TNode):
        if other is None:
            return False
        return (self.state == other.state) and (self.inducing == other.inducing) and (self._parent == other.parent)

    def __str__(self):
        return f"Node: {self.state} Inducing Action: {self.inducing}"

    @property
    def name(self):
        if self.inducing_action is None:
            return f'None_{self._state.port}'
        return f'{int(self.state.fuel_amount)}_{self.inducing_action.name}_{self.state.port}'


class RandomNode(Generic[TAction, TRandom, TDecisionNode], NewNode[TAction, TRandom]):

    def __init__(self,
                 parent: Optional[TStateNode] = None,
                 inducing: Optional[TAction] = None,
                 state: Optional[TState] = None,
                 is_terminal: bool = False):
        self._children: list[TDecisionNode] = []
        self._children_states: list =[]
        super().__init__(parent, inducing, state, is_terminal)

    @property
    def children(self: TRandomNode) -> list[TDecisionNode]:
        return self._children

    @property
    def seed(self):
        return self._seed

    def add_child(self: TRandomNode, child: TDecisionNode) -> None:
        if child.inducing_action == None:
            raise Exception("Inducing action must be set on child")
        if child.state in [ch.state for ch in self._children]: # We dont check inducing actions because they are always the same with the parent. since the inducing action of a decision node is the same as its parent.
            raise Exception("A child with the same state has already been added")
        self._children.append(child)
        self._children_states.append(child.state)

    @property
    def children_states(self):
        return self._children_states

    def child_with_specific_state(self, state):
        for child in self.children:
            if child.state == state:
                return child

    def __str__(self):
        return f'Inducing: {self.inducing}, State: {self.state}, Seed: {self._seed}'

    # def __eq__(self, other):
    #     return (self.price == other.price) and (self.state == other.state)


class DecisionNode(Generic[TAction, TRandom], NewNode[TAction, TRandom]):

    def __init__(self,
                 parent: Optional[TRandomNode] = None,
                 inducing: Optional[TRandom] = None,
                 state: Optional[TState] = None,
                 valid_actions: Optional[list[TAction]] = None,
                 is_terminal: bool = False):
        self._children: list[TRandomNode] = []
        self._valid_actions: Optional[list[TAction]] = valid_actions
        super().__init__(parent, inducing, state, is_terminal)

    @property
    def children(self: TDecisionNode) -> list[TRandomNode]:
        return self._children

    def explored_actions(self) -> list[TAction]:
        return [ch.inducing_action for ch in self._children]

    def add_child(self: TDecisionNode, child: TRandomNode) -> None:
        if child.inducing_action == None:
            raise Exception("Inducing action must be set on child")
        if child.state in [ch.state for ch in self._children] and child.inducing_action in[ch.inducing_action for ch in self.children]:
            raise Exception("A child with the same state has already been added")
        self._children.append(child)

    @property
    def valid_actions(self) -> list[TAction]:
        if self._valid_actions == None:
            raise RuntimeError(f"Simulation not run")
        return self._valid_actions

    @valid_actions.setter
    def valid_actions(self, value: list[TAction]) -> None:
        self._valid_actions = value

    def __str__(self):
        return f'Inducing: {self.inducing_action}, State: {self.state}'


class SoftmaxActionNode((ActionNode[TState, TAction])):
    def __init__(self, parent=None, inducing_action=None, valid_actions=None):
        super().__init__(parent, inducing_action)
        if valid_actions is None:
            valid_actions = []
        self.Q_stf = {}  # Dictionary to store Q_stf(s, a) for each action a
        self.N_sa = {}  # Dictionary to store N(s, a) for each action a
        self.valid_actions = valid_actions if valid_actions is not None else []

        for action in self.valid_actions:
            self.Q_stf[action] = 0.0
            self.N_sa[action] = 0


# Softmax Node
def calculate_lambda_s(node, epsilon):
    num_actions = len(node.valid_actions)
    n = node.n
    if n > 0:
        lambda_s = epsilon * num_actions / math.log(n + 1)
    else:
        # when N_s is zero
        lambda_s = epsilon * num_actions
    return lambda_s

