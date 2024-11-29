import numpy as np
import random
from typing import List, Optional
from tqdm import tqdm
from samples.hybrid_fuel.hybridfuelMDP import *
import pickle

# Constants for the problem
gas_mileage = GAS_MILEAGE  # Example value
electric_mileage = ELECTRIC_MILEAGE  # Example value
low_mileage = LOW_MILEAGE  # Example value
regen_battery_inc = REGEN_BATTERY_INC  # Example value
time_horizon = TIME_STEPS  # Time horizon for the MDP
max_resources = GAS_CAPACITY + ELECTRIC_CAPACITY # Global resource constraint


# Sub-MDP solver
class SubMDPSolver:
    def __init__(self, max_fuel, max_battery, horizon):
        self.max_fuel = max_fuel
        self.max_battery = max_battery
        self.horizon = horizon

        # Precompute actions for all possible (fuel, battery) pairs
        self.precomputed_actions = self._precompute_actions()

    def _precompute_actions(self):
        """Precompute all possible actions for each (fuel, battery) state."""
        actions = {}
        for fuel in range(int(self.max_fuel) + 1):
            for battery in range(int(self.max_battery) + 1):
                actions[(fuel, battery)] = [
                    VehicleAction(gas, electricity)
                    for gas in range(fuel + 1)
                    for electricity in range(battery + 1)
                ]
        return actions

    def solve(self, state: VehicleState):
        """
        Solve the sub-MDP for a specific state, returning a value function table, policy table, and cumulative reward.
        """
        value_table = np.zeros((self.horizon + 1, int(self.max_fuel) + 1, int(self.max_battery) + 1))
        policy_table = np.full((self.horizon, int(self.max_fuel) + 1, int(self.max_battery) + 1), None, dtype=object)

        for t in tqdm(range(self.horizon - 1, -1, -1), desc="Solving Sub-MDP"):
            for fuel in range(int(self.max_fuel) + 1):
                for battery in range(int(self.max_battery) + 1):
                    # Use precomputed actions
                    actions = self.precomputed_actions[(fuel, battery)]
                    max_value = 0
                    best_action = None
                    best_mileage = None

                    for action in actions:
                        next_fuel = max(0, fuel - action.gas)
                        next_battery = max(0, min(self.max_battery, battery - action.electricity + regen_battery_inc))

                        # Hidden Markov Model setup for mileage
                        transition_probabilities = {
                            "gas_efficient": 0.4,
                            "electric_efficient": 0.4,
                            "regenerative_braking": 0.2
                        }
                        noise_values = [-2, -1, 0, 1, 2]  # Fixed discrete noise values
                        observed_mileage = {
                            "gas_efficient": {
                                "gas": gas_mileage + random.choice(noise_values),
                                "electric": low_mileage + random.choice(noise_values)
                            },
                            "electric_efficient": {
                                "gas": low_mileage + random.choice(noise_values),
                                "electric": electric_mileage + random.choice(noise_values)
                            },
                            "regenerative_braking": {
                                "gas": low_mileage / random.choice(noise_values),
                                "electric": low_mileage + random.choice(noise_values)
                            }
                        }

                        # Sample a hidden state based on transition probabilities
                        hidden_state = random.choices(
                            list(transition_probabilities.keys()),
                            weights=transition_probabilities.values()
                        )[0]

                        # Use the observed mileage for the hidden state
                        mileage = observed_mileage[hidden_state]

                        # Calculate immediate reward based on observed mileage
                        immediate_reward = (
                            mileage["gas"] * action.gas +
                            mileage["electric"] * action.electricity
                        )

                        next_value = value_table[t + 1, next_fuel, next_battery]
                        value = immediate_reward + next_value

                        if value > max_value:
                            max_value = value
                            best_action = action
                            best_mileage = mileage

                    value_table[t, fuel, battery] = max_value
                    policy_table[t, fuel, battery] = {
                        "action": best_action,
                        "mileage": best_mileage
                    }

        # Return the total reward for the initial state
        initial_reward = value_table[0, int(state.fuel), int(state.battery)]
        return value_table, policy_table, initial_reward

# Global solver combining sub-MDP solutions
class HybridVehicleMDPSolver:
    def __init__(self, hybrid_mdp: "HybridVehicleMDP", global_resources: int, horizon: int):
        self.hybrid_mdp = hybrid_mdp
        self.global_resources = global_resources
        self.horizon = horizon
        self.sub_mdp_solvers = [SubMDPSolver(hybrid_mdp.max_fuel, hybrid_mdp.max_battery, horizon)]

    def solve(self):
        """
        Solve the global MDP by combining sub-MDP solutions under resource constraints.
        """
        resource_allocations = [0] * len(self.sub_mdp_solvers)
        total_rewards = []  # Store rewards for logging
        policy_tables = []  # Store policy tables for each sub-MDP

        for t in tqdm(range(self.horizon), desc="Solving Global MDP"):
            # Compute marginal utilities
            marginal_utilities = []
            for i, solver in enumerate(self.sub_mdp_solvers):
                state = self.hybrid_mdp.initial_state()
                value_table, policy_table, reward = solver.solve(state)  # Get policy table
                policy_tables.append(policy_table)  # Collect policy table
                marginal_utility = value_table[t, int(state.fuel), int(state.battery)]
                marginal_utilities.append((marginal_utility, i))
                if t == 0:  # Log initial reward for the first iteration
                    total_rewards.append(reward)

            # Sort tasks by marginal utility and allocate resources greedily
            marginal_utilities.sort(reverse=True, key=lambda x: x[0])
            remaining_resources = self.global_resources

            for utility, i in marginal_utilities:
                if remaining_resources <= 0:
                    break
                allocation = min(remaining_resources, max_resources // len(marginal_utilities))
                resource_allocations[i] += allocation
                remaining_resources -= allocation
        
        
        policy_dic = {}
        # Output policy tables
        for i, policy_table in enumerate(policy_tables):
            print(f"Policy Table for Sub-MDP {i}:")
            for t in range(policy_table.shape[0]):
                print(f"Time {t}:")
                for fuel in range(policy_table.shape[1]):
                    for battery in range(policy_table.shape[2]):
                        policy_entry = policy_table[t, fuel, battery]
                        if policy_entry is not None and policy_entry["action"] is not None:
                            action = policy_entry["action"]
                            mileage = policy_entry["mileage"]
                            print(f"  Fuel: {fuel}, Battery: {battery} -> Action: Gas={action.gas}, Electricity={action.electricity}, Mileage={mileage}")
                            policy_dic[repr((t, mileage, fuel, battery))] = action
        # Save policy tables as a .pkl file
        with open("saved_models/hybrid_policy_tables.pkl", "wb") as f:
            pickle.dump(policy_dic, f)
        print("Policy tables saved to hybrid_policy_tables.pkl")

        print("Total Rewards for Sub-MDPs:", total_rewards)  # Output cumulative rewards
        global_reward = sum(total_rewards)
        print("Global Reward:", global_reward)

        return resource_allocations

# Example usage
if __name__ == "__main__":
    hybrid_mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
    global_solver = HybridVehicleMDPSolver(hybrid_mdp, global_resources=GAS_CAPACITY+ELECTRIC_CAPACITY, horizon=TIME_STEPS)
    resource_allocations = global_solver.solve()
    print("Resource Allocations:", resource_allocations)
