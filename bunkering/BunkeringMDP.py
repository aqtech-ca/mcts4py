from __future__ import annotations

import math
from enum import Enum
from mcts4py.MDP import *
import random
import numpy as np

KAPPA=10
KAPPA2 = 100

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
    def __init__(self, value, arrival_time=None):
        self.portEnum = PortEnums(value)
        self.port = self.portEnum.name
        self.arrival_time = arrival_time

    def __update_arrival_time(self, arrival_time):
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

class BunkeringAction():
    def __init__(self, refuel_amounts, speed):
        self.refuel_amounts = refuel_amounts
        self.speed = speed

    def __str__(self) -> str:
        return (f'Refuel Amount : {self.refuel_amounts}, Speed: {self.speed} ')


class BunkeringPosition():
    def __init__(self, port: Port, fuel_amount) -> None:
        self.port= port
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
            self.reward -= (self.port.arrival_time - self.port.arrival_allowance_time())*self.kappa

        if self.fuel_amount < self.min_fuel_allowance:
            self.reward -= (self.min_fuel_allowance - self.fuel_amount)*self.kappa2
        if self.fuel_amount <=0:
            self.reward -= math.inf

class BunkeringState(BunkeringPosition):
    def __init__(self, port: Port, fuel_amount):
        super().__init__(port, fuel_amount)

        self.is_terminal = port.is_terminal()

    def is_action_valid(self, action: BunkeringAction, max_speed, current_fuel_size, max_fuel_size) -> bool:
        if action.speed <= max_speed and action.refuel_amounts + current_fuel_size <= max_fuel_size and \
                self.port.is_terminal():
            return True
        else:
            return False

    def resolve_action(self, action: BunkeringAction, max_speed, current_fuel_size, max_fuel_size) -> Optional[
        BunkeringState]:
        if self.is_action_valid(action, max_speed, current_fuel_size, max_fuel_size):
            return BunkeringState(port=self.port.next_port(), fuel_amount=self.fuel_amount + action.refuel_amounts)
        return None


class BunkeringMDP(MDP[BunkeringState, BunkeringAction]):
    def __init__(self, max_fuel_size: float, max_speed: float, reward_function, fuel_consumption_function,
                 minimum_allowable_fuel_percentage: float = 0.2,
                 initial_state: BunkeringState = BunkeringState(Port(0, arrival_time=0), 100, 15)):
        self.max_fuel_size = max_fuel_size
        self.max_speed = max_speed
        self.reward_function = reward_function
        self.fuel_consumption_function = fuel_consumption_function
        self.minimum_allowable_fuel_percentage = minimum_allowable_fuel_percentage
        self.initial_state = initial_state

    def arrival_time(self, previousPort: Port, speed):
