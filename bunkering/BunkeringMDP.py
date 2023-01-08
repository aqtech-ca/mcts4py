from __future__ import annotations

import itertools
import math
from aenum import Enum, extend_enum
from mcts4py.MDP import *
import random
import numpy as np

KAPPA = 10
KAPPA2 = 100
MAX_FUEL_SIZE = 150
INITIAL_SPEED = 20  # KNOTS
# DISTANCE IS IN MILES
MAX_SPEED = 30  # KNOTS
POSSIBLE_FUEL_ACTIONS = [round(MAX_FUEL_SIZE / i, 0) for i in range(1, 10)]
POSSIBLE_FUEL_ACTIONS.append(0)
POSSIBLE_SPEED_ACTIONS = [round(MAX_SPEED / i, 2) for i in range(1, 10)]
POSSIBLE_ACTIONS = [(x, y) for x in POSSIBLE_SPEED_ACTIONS for y in POSSIBLE_FUEL_ACTIONS]


class PortEnums(Enum):
    Qingdao = 0
    Shanghai = 1
    Ningbo = 2
    Yantian = 3
    CaiMap = 4
    Singapore = 5
    Jeddah = 6
    Malta = 7
    LeHavre = 8
    Antwerp = 9
    Rotterdam = 10
    Hamburg = 11


class Port():
    def __init__(self, value, arrival_time=0):
        self.portEnum = PortEnums(value)
        self.port = self.portEnum.name
        self.arrival_time = arrival_time

    def update_arrival_time(self, arrival_time):
        self.arrival_time = arrival_time

    def is_terminal(self):
        if self.portEnum == PortEnums.Hamburg:
            return True
        else:
            return False

    def next_port(self):
        if self.portEnum != PortEnums.Hamburg:
            return Port(self.portEnum.value + 1)
        else:
            return None

    def distance_to_next_port(self):
        if self.portEnum == PortEnums.Qingdao:
            return 367
        if self.portEnum == PortEnums.Shanghai:
            return 87
        if self.portEnum == PortEnums.Ningbo:
            return 917
        if self.portEnum == PortEnums.Yantian:
            return 1085
        if self.portEnum == PortEnums.CaiMap:
            return 775
        if self.portEnum == PortEnums.Singapore:
            return 4685
        if self.portEnum == PortEnums.Jeddah:
            return 1908
        if self.portEnum == PortEnums.Malta:
            return 2538
        if self.portEnum == PortEnums.LeHavre:
            return 244
        if self.portEnum == PortEnums.Antwerp:
            return 144
        if self.portEnum == PortEnums.Rotterdam:
            return 341
        if self.portEnum == PortEnums.Hamburg:
            return None

    def arrival_allowance_time(self):
        if self.portEnum == PortEnums.Qingdao:
            return 0
        if self.portEnum == PortEnums.Shanghai:
            return 28
        if self.portEnum == PortEnums.Ningbo:
            return 35
        if self.portEnum == PortEnums.Yantian:
            return 112
        if self.portEnum == PortEnums.CaiMap:
            return 202
        if self.portEnum == PortEnums.Singapore:
            return 266
        if self.portEnum == PortEnums.Jeddah:
            return 656
        if self.portEnum == PortEnums.Malta:
            return 815
        if self.portEnum == PortEnums.LeHavre:
            return 1026
        if self.portEnum == PortEnums.Antwerp:
            return 1046
        if self.portEnum == PortEnums.Rotterdam:
            return 1058
        if self.portEnum == PortEnums.Hamburg:
            return 1086


class BunkeringAction(Enum):
    pass

    def speed(self):
        return self.value[0]

    def refuel_amount(self):
        return self.value[1]


def setBunkeringActionAttributes():
    for i, action in enumerate(POSSIBLE_ACTIONS):
        name = f'A_{action[0]}_{action[1]}'
        extend_enum(BunkeringAction, name, action)


class BunkeringPosition():
    def __init__(self, port: Port, fuel_amount) -> None:
        self.port = port
        self.fuel_amount = fuel_amount

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, BunkeringPosition):
            return __o.port == self.port and __o.fuel_amount == self.fuel_amount
        return False

    def __str__(self) -> str:
        return f'{self.port}, {self.fuel_amount}'


class BunkeringReward(BunkeringPosition):
    def __init__(self, port: Port, fuel_amount, arrival_time, min_fuel_allowance, kappa=KAPPA, kappa2=KAPPA2):
        super().__init__(port, fuel_amount)
        self.arrival_time = arrival_time
        self.kappa = kappa
        self.kappa2 = kappa2
        self.min_fuel_allowance = min_fuel_allowance
        self.reward = 0

    def calculate_reward(self):
        if self.port.arrival_allowance_time() >= self.port.arrival_time:
            pass
        else:
            self.reward -= (self.port.arrival_time - self.port.arrival_allowance_time()) * self.kappa

        if self.fuel_amount < self.min_fuel_allowance:
            self.reward -= (self.min_fuel_allowance - self.fuel_amount) * self.kappa2
        if self.fuel_amount <= 0:
            self.reward -= math.inf


class BunkeringState(BunkeringPosition):
    def __init__(self, port: Port, fuel_amount, speed):
        super().__init__(port, fuel_amount)

        self.is_terminal = port.is_terminal()
        self.speed = speed

    def is_action_valid(self, action: BunkeringAction, current_fuel_size) -> bool:
        if action.speed() <= MAX_SPEED and action.refuel_amount() + current_fuel_size <= MAX_FUEL_SIZE and \
                not self.is_terminal:
            return True
        else:
            return False

    def resolve_action(self, action: BunkeringAction, current_fuel_size) -> Optional[
        BunkeringState]:
        if self.is_action_valid(action, current_fuel_size):
            return BunkeringState(port=self.port.next_port(), fuel_amount=current_fuel_size + action.refuel_amount(),
                                  speed=action.speed())
        return None


class BunkeringMDP(MDP[BunkeringState, BunkeringAction]):
    def __init__(self, TEU, minimum_allowable_fuel_percentage: float = 0.9):
        self.minimum_allowable_fuel_percentage = minimum_allowable_fuel_percentage
        self.starting_state = BunkeringState(Port(0, arrival_time=0), fuel_amount=MAX_FUEL_SIZE, speed=INITIAL_SPEED)
        self.current_fuel = MAX_FUEL_SIZE
        self.TEU = TEU
        self.k1 = None
        self.k2 = None
        setBunkeringActionAttributes()

    def is_terminal(self, state: BunkeringState) -> bool:
        return state.is_terminal

    def __set_k_coefficients(self):
        if self.TEU <= 1000:
            self.k1 = 0.004476
            self.k2 = 6.17
        elif self.TEU <= 2000:
            self.k1 = 0.004595
            self.k2 = 16.42
        elif self.TEU <= 3000:
            self.k1 = 0.004501
            self.k2 = 29.28
        elif self.TEU <= 4000:
            self.k1 = 0.006754
            self.k2 = 32.23
        elif self.TEU <= 5000:
            self.k1 = 0.006732
            self.k2 = 55.84
        elif self.TEU <= 6000:
            self.k1 = 0.007297
            self.k2 = 71.4
        else:
            self.k1 = 0.006705
            self.k2 = 87.71

    def __fuel_consumption_function(self, distance, speed):  # tons per day
        if self.k1 is None:
            self.__set_k_coefficients()
        consumption_per_day = self.k1 * pow(speed, 3) + self.k2
        total_consumption = (distance / speed) / 24
        return total_consumption

    def initial_state(self) -> BunkeringState:
        return self.starting_state

    def reward(self, previous_state: Optional[BunkeringState], action: Optional[BunkeringAction],
               state: BunkeringState) -> float:
        br = BunkeringReward(previous_state.port, self.current_fuel,
                              self.arrival_time(previousPort=previous_state.port, speed=action.speed()),
                              self.minimum_allowable_fuel_percentage)
        br.calculate_reward()
        return br.reward

    def arrival_time(self, previousPort: Port, speed):
        return previousPort.arrival_time + previousPort.distance_to_next_port() / speed

    def transition(self, state: BunkeringState, action: BunkeringAction) -> BunkeringState:
        if state.is_terminal:
            return state

        fuel_consumption = self.__fuel_consumption_function(state.port.distance_to_next_port(), action.speed())
        current_fuel = state.fuel_amount - fuel_consumption
        target_state = state.resolve_action(action, current_fuel)
        if target_state is None:
            return state
        else:
            self.current_fuel = target_state.fuel_amount
            target_state.port.update_arrival_time(self.arrival_time(state.port, state.speed))
            return target_state

    def actions(self, state: BunkeringState) -> list[BunkeringAction]:
        actions = list()
        for a in BunkeringAction:
            if state.is_action_valid(a, self.current_fuel):
                actions.append(a)
        return actions
