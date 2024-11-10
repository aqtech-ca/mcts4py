from samples.hybrid_fuel.hybridfuelMDP import *
from samples.hybrid_fuel.hybrid_mc_sim import *

# Example Usage:
mdp = HybridVehicleMDP(max_fuel=10, max_battery=5)
initial_state = mdp.initial_state()
print("Initial state:", initial_state)
print("Actions:", mdp.actions(initial_state))
print("Transition:", mdp.transition(initial_state, action=VehicleAction(gas=1, electricity=0)))

if __name__ == '__main__':    
    # Initialize MDP and initial state
    mc_simple_pol = []
    mc_random_pol = []
    for i in range(MC_ITER):
        mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
        trajectory_simple_pol, rewards_simple_pol = sim_hybrid_mdp(mdp, time_steps=TIME_STEPS, policy=simple_policy, verbose=False)
        mc_simple_pol.append(sum(rewards_simple_pol))

        mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
        trajectory_random_pol, rewards_random_pol = sim_hybrid_mdp(mdp, time_steps=TIME_STEPS, verbose=False)
        mc_random_pol.append(sum(rewards_random_pol))

    print(f"sum of rewards - simple policy: {np.mean(mc_simple_pol)}")
    print(f"sum of rewards - random_policy: {np.mean(mc_random_pol)}")