from samples.hybrid_fuel.hybrid_mc_sim import *
from mcts4py.GenericSolver import GenericSolver
from mcts4py.StatefulSolver import StatefulSolver

# Example Usage:
# mdp = HybridVehicleMDP(max_fuel=10, max_battery=5)
# initial_state = mdp.initial_state()
# print("Initial state:", initial_state)
# print("Actions:", mdp.actions(initial_state))
# print("Transition:", mdp.transition(initial_state, action=VehicleAction(gas=1, electricity=0)))

# mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
# trajectory_simple_pol, rewards_simple_pol = sim_hybrid_mdp(mdp, time_steps=TIME_STEPS, policy=greedy_policy, verbose=True)

if __name__ == '__main__':    
    # Initialize MDP and initial state

    mc_simple_pol = []
    mc_random_pol = []
    mc_mcts_pol = []
    mc_policy_dic = []
    for i in tqdm(range(MC_ITER)):
        mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
        trajectory_simple_pol, rewards_simple_pol = sim_hybrid_mdp(mdp, time_steps=TIME_STEPS, policy=greedy_policy, verbose=False)
        mc_simple_pol.append(sum(rewards_simple_pol))

        mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
        trajectory_random_pol, rewards_random_pol = sim_hybrid_mdp(mdp, time_steps=TIME_STEPS, verbose=False)
        mc_random_pol.append(sum(rewards_random_pol))

        mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
        trajectory_random_pol, rewards_random_pol = sim_hybrid_mdp(mdp, time_steps=TIME_STEPS, policy=mcts_policy, verbose=False)
        mc_mcts_pol.append(sum(rewards_random_pol))

        mdp = HybridVehicleMDP(max_fuel=GAS_CAPACITY, max_battery=ELECTRIC_CAPACITY)
        trajectory_random_pol, rewards_random_pol = sim_hybrid_mdp(mdp, time_steps=TIME_STEPS, policy=policy_lookup, verbose=False)
        mc_policy_dic.append(sum(rewards_random_pol))
    
    print(f"sum of rewards - greedy policy: {np.mean(mc_simple_pol)}")
    print(f"sum of rewards - random_policy: {np.mean(mc_random_pol)}")
    print(f"sum of rewards - mcts_policy: {np.mean(mc_mcts_pol)}")
    print(f"sum of rewards - policy_dic: {np.mean(mc_policy_dic)}")

    
    