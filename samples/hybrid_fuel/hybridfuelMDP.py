from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Tuple
import random
from mcts4py.MDP import *
import numpy as np

TState = TypeVar('TState')
TAction = TypeVar('TAction')
TRandom = TypeVar('TRandom')

GAS_MILEAGE = 10 # dist per unit of gas
ELECTRIC_MILEAGE = 10 #dist per unit of electricity
LOW_MILEAGE = 1 #dist per unit of gas or anything

GAS_CAPACITY = 20
ELECTRIC_CAPACITY = 40
REGEN_BATTERY_INC = 2

RESOURCE_INC = 2
TIME_STEPS = 50

MC_ITER = 100

class VehicleState:
    """
    Represents the state of the hybrid vehicle.
    
    Attributes:
        fuel (float): Current fuel level.
        battery (float): Current battery level.
        time_step (int): Current time increment in the simulation.
        initial_fuel (float): Initial fuel level for reference.
        initial_battery (float): Initial battery level for reference.
    """
    

    def __init__(self, fuel: float, battery: float, time_step: int = 0, initial_fuel: float = 0, initial_battery: float = 0, scenario='gas_efficient'):
        self.fuel = fuel
        self.battery = battery
        self.time_step = time_step
        self.initial_fuel = initial_fuel
        self.initial_battery = initial_battery
        self.scenario = scenario

    def __repr__(self):
        return f"VehicleState(fuel={self.fuel}, battery={self.battery}, time_step={self.time_step}, scenario={self.scenario})"

class VehicleAction():
    def __init__(self, gas: float, electricity: float):
        self.gas = gas
        self.electricity = electricity
    
    def __str__(self):
        return f"VehicleAction(gas={self.gas:.3f}, electricity={self.electricity:.3f})"


class HybridVehicleMDP(MDP[VehicleState, str]):
    def __init__(self, max_fuel: float, max_battery: float):
        self.max_fuel = max_fuel
        self.max_battery = max_battery
        self.scenarios = ['gas_efficient', 'electric_efficient', 'regenerative_braking']
        self.gas_mileage = GAS_MILEAGE
        self.electric_mileage = ELECTRIC_MILEAGE
        self.scenario = 'gas_efficient'
    
    def initial_state(self) -> VehicleState:
        scenario = random.choice(self.scenarios)  # Randomly choose a scenario
        return VehicleState(fuel=self.max_fuel, battery=self.max_battery, initial_fuel=self.max_fuel, initial_battery=self.max_battery, scenario=scenario)

    def transition(self, state: VehicleState, action: VehicleAction) -> VehicleState:
        fuel, battery = state.fuel, state.battery
        
        # set consumption rates: todo
        fuel = np.clip(fuel - action.gas, 0, np.Inf)  # Consume gas
        battery = np.clip(battery - action.electricity, 0, np.Inf)  # Consume electricity

        if state.scenario == 'regenerative_braking':
            battery = min(self.max_battery, battery + REGEN_BATTERY_INC)  # Regen brake adds battery
        
        state.scenario = random.choice(self.scenarios)  # Randomly choose a scenario
        
        return VehicleState(fuel=fuel, 
                            battery=battery, 
                            time_step=state.time_step + 1,
                            initial_fuel=state.initial_fuel, 
                            initial_battery=state.initial_battery, 
                            scenario=state.scenario)

    def reward(self, previous_state: Optional[VehicleState], action: Optional[str]) -> float:
        if not previous_state or not action:
            return 0
        
        gas_usage = min(previous_state.fuel, action.gas)
        electric_usage = min(previous_state.battery, action.electricity)

        if previous_state.scenario == 'gas_efficient':
            gas_mileage = GAS_MILEAGE
            electric_mileage = LOW_MILEAGE
            return gas_mileage*gas_usage + electric_mileage*electric_usage 
        elif previous_state.scenario == 'electric_efficient':
            gas_mileage = LOW_MILEAGE
            electric_mileage = ELECTRIC_MILEAGE
            return gas_mileage*gas_usage + electric_mileage*electric_usage
        else: # regen braking
            return 0.0
        
    def actions(self, state: VehicleState, state_visit=0, iteration_number=0, max_iteration_number=0,
                dpw_exploration=1, dpw_alpha=1, min_action=False) -> List[str]:
        available_actions = []
        if state.fuel > 0:
            available_actions.append("use_gas")
        if state.battery > 0:
            available_actions.append("use_electricity")
        if state.fuel < self.max_fuel:
            available_actions.append("regenerate")
        return available_actions

    def is_terminal(self, state: VehicleState) -> bool:
        return state.fuel <= 0 and state.battery <= 0

