from samples.gym.GymMentsSolver import MentSolver
from samples.gym.GymMentsSolverWithBTS import MentSolverWithBTS
from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.frozenLake.frozenLakeWrapper import FrozenLakeMDP


def evaluate_solver(solver_class, iterations_list, trials=100, is_slippery=False):
    exploration_constant = 0.5
    results = []
    print(f"trials: {trials}, exploration_constant: {exploration_constant},  is_slippery: {is_slippery}")

    for iterations in iterations_list:
        success_count = 0
        total_rewards = 0
        total_steps = 0

        for trial in range(trials):
            gym_mdp = FrozenLakeMDP(is_slippery)
            solver = solver_class(
                mdp=gym_mdp,
                simulation_depth_limit=100,
                exploration_constant=exploration_constant,
                discount_factor=0.8,
                verbose=False
            )

            total_reward = 0
            step_count = 0
            max_steps = 100
            done = False

            while not done and step_count < max_steps:
                solver.run_search(iterations)
                best_action = solver.do_best_action(solver.root())
                state, reward, done, _, _ = gym_mdp.step(best_action)

                act = ''
                if best_action == 0:
                    act = '←'
                elif best_action == 1:
                    act = '↓'
                elif best_action == 2:
                    act = '→'
                elif best_action == 3:
                    act = '↑'
                # print(f"move from {gym_mdp.initial}, action: {act}, state: {state}, reward: {reward}, done: {done}")
                total_reward += reward
                step_count += 1

                gym_mdp.initial = state
                solver = solver_class(
                    mdp=gym_mdp,
                    simulation_depth_limit=100,
                    exploration_constant=exploration_constant,
                    discount_factor=0.8,
                    verbose=False
                )

                if state == 15:
                    # print(f"iterations: {iterations} steps: {step_count} trial: {trial} state: {state}")
                    total_steps += step_count
                    success_count += 1
                    break

            total_rewards += total_reward

        avg_reward = total_rewards / trials
        avg_steps = total_steps / success_count if success_count != 0 else -1
        success_rate = success_count / trials

        results.append({
            "iterations": iterations,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
        })

        print({
            "iterations": iterations,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "success_rate": success_rate,
        })

    return results


if __name__ == "__main__":
    iterations_list = [1, 2, 3, 5, 10, 20, 30, 50, 100, 150, 200, 500, 1000, 1500, 2000, 2500]

    # print("running MENTS with BTS")
    # bts_results = evaluate_solver(MentSolverWithBTS, iterations_list, is_slippery=True)
    # print("done")

    # print("running UCT")
    # uct_results = evaluate_solver(GenericSolver, iterations_list, is_slippery=False)
    # print("done")

    print("running METNS")
    ments_results = evaluate_solver(MentSolver, iterations_list, is_slippery=False)
    print("done")
