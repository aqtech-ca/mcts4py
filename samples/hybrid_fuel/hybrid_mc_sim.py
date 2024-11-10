from samples.hybrid_fuel.hybridfuelMDP import *
from mcts4py.StatefulSolver import *
from samples.hybrid_fuel.hybrid_state_definitions import *

def random_policy(state: VehicleState, mdp: HybridVehicleMDP) -> VehicleAction:
    return random_logic(state)

def greedy_policy(state: VehicleState, mdp: HybridVehicleMDP) -> VehicleAction:
    return greedy_logic(state)

def mcts_policy(state: VehicleState, mdp) -> VehicleAction:

    if state.scenario == "regenerative_braking":
        return VehicleAction(gas=0.0, electricity=0.0)
    
    if state.fuel == 0.0:
        electricity_amount = min(state.battery, RESOURCE_INC)
        return VehicleAction(gas=0.0, electricity=electricity_amount)
    
    if state.battery == 0.0:
        gas_amount = min(state.fuel, RESOURCE_INC)
        return VehicleAction(gas=gas_amount, electricity=0.0)

    solver = StatefulSolver(
        mdp,
        simulation_depth_limit = state.time_remaining,
        exploration_constant = 999.0,
        discount_factor = 0.99,
        verbose = False)
    
    solver.run_search(MCTS_IERS)

    # print("\nSearch Tree:")
    # solver.display_tree()
    # print(solver.calculate_uct(solver.root()))

    nodes_from_root = solver.root().children
    # for node in nodes_from_root:
    #     print(node)
    #     print(node.state)
    #     print(node.inducing_action)
    #     print(node.parent)
    #     print("-----")

    max_uct_node = max(nodes_from_root, key=lambda node: solver.calculate_uct(node))
    ucts = [solver.calculate_uct(node) for node in nodes_from_root]
    # print(max_uct_node)
    # print(solver.calculate_uct(nodes_from_root[0]))
    # print(solver.calculate_uct(nodes_from_root[1]))
    # print(solver.calculate_uct(nodes_from_root[2]))

    return max_uct_node.inducing_action

# simulation function
def sim_hybrid_mdp(mdp: HybridVehicleMDP, time_steps: int = 200, policy=random_policy, verbose=False):
    
    initial_state = mdp.initial_state()

    # Simulate trajectory
    current_state = initial_state
    trajectory = [current_state]
    rewards = []

    for t in range(time_steps):
        available_actions = mdp.actions(current_state)
        
        if available_actions:
            
            action = policy(current_state, mdp=mdp)
            reward = mdp.reward(current_state, action, current_state) # hacky here
            rewards.append(reward)
            if verbose: print(f"Step {t}: Action={action}, \n State={current_state}, \n Reward={reward:.3f} \n ---")

            next_state = mdp.transition(current_state, action)

            trajectory.append(next_state)
            current_state = next_state
        else:
            print(f"Step {t}: No available actions, terminating early.")
            break
    
    return trajectory, rewards

