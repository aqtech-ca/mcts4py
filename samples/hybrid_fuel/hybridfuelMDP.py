from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Tuple
import random
from mcts4py.MDP import *
import numpy as np
from samples.hybrid_fuel.hybrid_state_definitions import *

TState = TypeVar('TState')
TAction = TypeVar('TAction')
TRandom = TypeVar('TRandom')


def random_logic(state: VehicleState) -> VehicleAction:
    lower_gas = 0
    upper_gas = min(state.fuel, RESOURCE_INC)
    gas_amount = random.uniform(lower_gas, upper_gas)

    lower_electric = 0
    upper_electric = min(state.battery, RESOURCE_INC-gas_amount)

    # electric_amount = random.uniform(lower_electric, upper_electric)
    electric_amount = upper_electric
    action = VehicleAction(gas=gas_amount, electricity=electric_amount)

    # Quick renormalize

    return action

def greedy_logic(state: VehicleState) -> VehicleAction:
    if state.scenario == "gas_efficient" or state.battery == 0:
        gas_amount = RESOURCE_INC
        electric_amount = 0
        if 0 <= state.fuel < RESOURCE_INC:
            gas_amount = state.fuel
            if state.battery > 0:
                electric_amount = min(state.battery, RESOURCE_INC-gas_amount)
    elif state.scenario == "electric_efficient" or state.fuel == 0:
        gas_amount = 0
        electric_amount = RESOURCE_INC
        if 0 <= state.battery < RESOURCE_INC:
            electric_amount = state.battery
            if state.battery > 0:
                gas_amount = min(state.fuel, RESOURCE_INC-electric_amount)
    else:
        return random_logic(state)
    # Default action if no efficient scenario or insufficient resources
    return VehicleAction(gas=gas_amount, electricity=electric_amount)


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
                            time_remaining = state.time_remaining - 1,
                            scenario=state.scenario)

    def reward(self, previous_state: Optional[VehicleState], action: Optional[str], next_state: Optional[VehicleState]) -> float:
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
        gas_min = 0.0
        electricity_min = 0.0

        action_greedy = greedy_logic(state)

        alternative_regime = "electric_efficient" if state.scenario =="gas_efficient" else "gas_efficient"
        action_alternative = greedy_logic(VehicleState(fuel=state.fuel, battery=state.battery, scenario=alternative_regime))

        if 0 <= state.fuel < RESOURCE_INC:
            gas_max = state.fuel
            electricity_min = RESOURCE_INC - gas_max
        else:
            gas_max = RESOURCE_INC
        
        if 0 <= state.battery < RESOURCE_INC:
            electricity_max = state.battery
            gas_min = RESOURCE_INC - electricity_max
        else:
            electricity_max = RESOURCE_INC

        available_actions = [# VehicleAction(gas=gas_min, electricity=electricity_min),
                            action_greedy,
                            action_alternative,
                            VehicleAction(gas=0.0, electricity=0.0)]
                            # VehicleAction(gas=gas_max, electricity=electricity_min),
                            # VehicleAction(gas=gas_min, electricity=electricity_max)]
        
        return available_actions

    def is_terminal(self, state: VehicleState) -> bool:
        return state.time_remaining == 0


