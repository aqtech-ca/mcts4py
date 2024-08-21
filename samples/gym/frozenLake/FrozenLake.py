from samples.gym.GymMentsSolver import MentSolver
from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.frozenLake.frozenLakeWrapper import FrozenLakeMDP


def evaluate_solver(solver_class, iterations_list, trials=1):
    results = []

    for iterations in iterations_list:
        success_count = 0
        total_rewards = 0
        total_steps = 0

        for trial in range(trials):
            gym_mdp = FrozenLakeMDP()
            solver = solver_class(
                mdp=gym_mdp,
                simulation_depth_limit=100,
                exploration_constant=1.0,
                discount_factor=0.99,
                verbose=False
            )

            total_reward = 0
            step_count = 0
            max_steps = 100
            done = False

            while not done and step_count < max_steps:
                solver.run_search(iterations)
                best_action = solver.do_best_action(solver.root())
                state, reward, done, _, _ = gym_mdp.env.step(best_action)
                # print(f"move from {gym_mdp.initial}, action: {best_action}, state: {state}, reward: {reward}, done: {done}")
                total_reward += reward
                step_count += 1

                gym_mdp.initial = state
                solver = solver_class(
                    mdp=gym_mdp,
                    simulation_depth_limit=100,
                    exploration_constant=1.0,
                    discount_factor=0.99,
                    verbose=False
                )

                total_rewards += total_reward
                total_steps += step_count
                if state == 15:
                    success_count += 1

        avg_reward = total_rewards / trials
        avg_steps = total_steps / trials
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
    iterations_list = [1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000]

    print("running UCT")
    uct_results = evaluate_solver(GenericSolver, iterations_list)

    print("running METNS")
    ments_results = evaluate_solver(MentSolver, iterations_list)

    # print("UCT Results:", uct_results)
    # print("MENTS Results:", ments_results)
