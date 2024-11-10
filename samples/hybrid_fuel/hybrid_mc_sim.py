from samples.hybrid_fuel.hybridfuelMDP import *


def random_policy(state: VehicleState) -> VehicleAction:
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

def greedy_policy(state: VehicleState) -> VehicleAction:
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
        return random_policy(state)
    # Default action if no efficient scenario or insufficient resources
    return VehicleAction(gas=gas_amount, electricity=electric_amount)

def sim_hybrid_mdp(mdp: HybridVehicleMDP, time_steps: int = 200, policy=random_policy, verbose=False):
    
    initial_state = mdp.initial_state()

    # Simulate trajectory
    current_state = initial_state
    trajectory = [current_state]
    rewards = []

    for t in range(time_steps):
        available_actions = mdp.actions(current_state)
        
        if available_actions:
            
            action = policy(current_state)

            
            reward = mdp.reward(current_state, action)
            rewards.append(reward)
            if verbose: print(f"Step {t}: Action={action}, \n State={current_state}, \n Reward={reward:.3f} \n ---")

            next_state = mdp.transition(current_state, action)

            
            trajectory.append(next_state)
            current_state = next_state
        else:
            print(f"Step {t}: No available actions, terminating early.")
            break
    
    return trajectory, rewards