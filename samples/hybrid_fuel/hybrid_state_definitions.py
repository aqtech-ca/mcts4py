import numpy as np
import random

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

NOISE_VALUES = [-2, -1, 0, 1, 2] 
MARKOVIAN_TRANS_DIC = {
        "gas_efficient": {
            "gas_efficient": 0.4,
            "electric_efficient": 0.4,
            "regenerative_braking": 0.2
        },
        "electric_efficient": {
            "gas_efficient": 0.4,
            "electric_efficient": 0.4,
            "regenerative_braking": 0.2
        },
        "regenerative_braking": {
            "gas_efficient": 0.4,
            "electric_efficient": 0.4,
            "regenerative_braking": 0.2
        }
    }
DEFAULT_SCENARIO = "gas_efficient"

OBSERVED_MILEAGE = {
    "gas_efficient": {
        "gas": GAS_MILEAGE + random.choice(NOISE_VALUES),
        "electric": LOW_MILEAGE + random.choice(NOISE_VALUES)
    },
    "electric_efficient": {
        "gas": LOW_MILEAGE + random.choice(NOISE_VALUES),
        "electric": ELECTRIC_MILEAGE + random.choice(NOISE_VALUES)
    },
    "regenerative_braking": {
        "gas": 0.0, # LOW_MILEAGE + random.choice(NOISE_VALUES),
        "electric": 0.0 # LOW_MILEAGE + random.choice(NOISE_VALUES)
    }
}

class VehicleState:
    def __init__(self, fuel: float, battery: float, scenario: str = None, gas_mileage: float = None, electricity_mileage: float = None, time_remaining=TIME_STEPS):
        self.fuel = fuel
        self.battery = battery
        self.time_remaining = time_remaining
        self.gas_mileage = gas_mileage
        self.electricity_mileage = electricity_mileage

        self.scenario = scenario
        if self.gas_mileage is not None and self.electricity_mileage is not None:
            self.scenario = "gas_efficient" if self.gas_mileage > self.electricity_mileage else "electric_efficient"
            if self.gas_mileage == 0.0 and self.electricity_mileage == 0.0:
                self.scenario = "regenerative_braking"

    def __repr__(self):
        return f"VehicleState(fuel={self.fuel}, battery={self.battery}, time_remaining={self.time_remaining}, scenario={self.scenario})"

class VehicleAction():
    def __init__(self, gas: float, electricity: float):
        self.gas = gas
        self.electricity = electricity
    
    def __str__(self):
        return f"VehicleAction(gas={self.gas:.3f}, electricity={self.electricity:.3f})"
