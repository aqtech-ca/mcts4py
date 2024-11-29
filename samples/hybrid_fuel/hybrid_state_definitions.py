import numpy as np

GAS_MILEAGE = 30  # dist per unit of gas
ELECTRIC_MILEAGE = 40  # dist per unit of electricity
LOW_MILEAGE = 10  # dist per unit of gas or anything

GAS_CAPACITY = 20
ELECTRIC_CAPACITY = 20
REGEN_BATTERY_INC = 5

RESOURCE_INC = 5
TIME_STEPS = 16

MYOPIC_THRESH = TIME_STEPS - 1

MC_ITER = 99
MCTS_IERS = 99

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
    

    def __init__(self, fuel: float, battery: float, time_step: int = 0, initial_fuel: float = 0, initial_battery: float = 0, scenario='gas_efficient', time_remaining=TIME_STEPS):
        self.fuel = fuel
        self.battery = battery
        self.time_remaining = time_remaining
        self.scenario = scenario

    def __repr__(self):
        return f"VehicleState(fuel={self.fuel}, battery={self.battery}, time_remaining={self.time_remaining}, scenario={self.scenario})"

class VehicleAction():
    def __init__(self, gas: float, electricity: float):
        self.gas = gas
        self.electricity = electricity
    
    def __str__(self):
        return f"VehicleAction(gas={self.gas:.3f}, electricity={self.electricity:.3f})"
