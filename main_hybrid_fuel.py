from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List, Tuple
import random
from mcts4py.MDP import *

TState = TypeVar('TState')
TAction = TypeVar('TAction')
TRandom = TypeVar('TRandom')

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
        return f"VehicleAction(gas={self.gas}, electricity={self.electricity})"


class HybridVehicleMDP(MDP[VehicleState, str]):
    def __init__(self, max_fuel: float, max_battery: float):
        self.max_fuel = max_fuel
        self.max_battery = max_battery
        self.scenarios = ['gas_efficient', 'electric_efficient', 'regenerative_braking']
    
    def initial_state(self) -> VehicleState:
        scenario = random.choice(self.scenarios)  # Randomly choose a scenario
        return VehicleState(fuel=self.max_fuel, battery=self.max_battery, initial_fuel=self.max_fuel, initial_battery=self.max_battery, scenario=scenario)

    def transition(self, state: VehicleState, action: VehicleAction) -> VehicleState:
        fuel, battery = state.fuel, state.battery
        if state.scenario == 'gas_efficient' and fuel > 0:
            fuel -= min(1, action.gas)  # Consume gas
        elif state.scenario == 'electric_efficient' and battery > 0:
            battery -= min(1, action.electricity)  # Consume electricity
        elif state.scenario == 'regenerative_braking':
            battery = min(self.max_battery, battery + 0.5)  # Regen brake adds battery
        
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
        if previous_state.scenario == 'gas_efficient':
            return 5*action.gas  # Assuming gas provides 5 units of distance.
        elif previous_state.scenario == 'electric_efficient':
            return 10*action.electricity
        elif action == "regenerate":
            return 0  # Negative reward for not moving but recharging.
        return 0

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

# Example Usage:
mdp = HybridVehicleMDP(max_fuel=10, max_battery=5)
initial_state = mdp.initial_state()
print("Initial state:", initial_state)
print("Actions:", mdp.actions(initial_state))
print("Transition:", mdp.transition(initial_state, action=VehicleAction(gas=1, electricity=0)))

if __name__ == '__main__':
    # unittest.main()
    
    # Initialize MDP and initial state
    mdp = HybridVehicleMDP(max_fuel=10, max_battery=5)
    initial_state = mdp.initial_state()

    # Simulate trajectory
    current_state = initial_state
    trajectory = [current_state]
    rewards = []

    for t in range(20):
       
        available_actions = mdp.actions(current_state)
        
        if available_actions:
            random_fuel_choice = random.choice([0, 1])  # Randomly pick an available action
            if random_fuel_choice == 0:
                action = VehicleAction(gas=1, electricity=0)
            else:
                action = VehicleAction(gas=0, electricity=1)
            
            next_state = mdp.transition(current_state, action)
            reward = mdp.reward(current_state, action)
            rewards.append(reward)

            print(f"Step {t}: Action={action}, State={next_state}", f"Reward={reward}")
            trajectory.append(next_state)
            current_state = next_state
        else:
            print(f"Step {t}: No available actions, terminating early.")
            break