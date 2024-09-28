from samples.gym.GymMentsSolver import MentSolver
from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.mountainCar.mountainCarWrapper import MountainCarWrapper


def evaluate_solver(solver_class, iterations_list, trials=10):
    exploration_constant = 0.5
    results = []
    print(f"trials: {trials}, exploration_constant: {exploration_constant}")

    for iterations in iterations_list:
        success_count = 0
        total_rewards = 0
        total_steps = 0

        for trial in range(trials):
            gym_mdp = MountainCarWrapper()
            solver = solver_class(
                mdp=gym_mdp,
                simulation_depth_limit=100,
                exploration_constant=exploration_constant,
                discount_factor=0.99,
                verbose=False
            )

            state = gym_mdp.initial_state()
            total_reward = 0
            step_count = 0
            done = False
            max_step = 4000
            trunc = False

            while not done and step_count < max_step:
                solver.run_search(iterations)
                best_action = solver.do_best_action(solver.root())
                state, reward, done, _, _ = gym_mdp.step(best_action)
                total_reward += reward
                step_count += 1

                gym_mdp.current_state = state
                solver = solver_class(
                    mdp=gym_mdp,
                    simulation_depth_limit=100,
                    exploration_constant=1.0,
                    discount_factor=0.99,
                    verbose=False
                )

                if done:
                    total_steps += step_count
                    success_count += 1
                    break

            print(f"Trial {trial}: Steps = {step_count}, Reward = {total_reward}")

            total_rewards += total_reward
            total_steps += step_count
            if gym_mdp.is_terminal(state):
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
    iterations_list = [1, 2, 3, 5, 10, 50, 100, 500, 1000, 1500, 2000]

    print("running UCT")
    uct_results = evaluate_solver(GenericSolver, iterations_list)
    print("UCT Results:", uct_results)

    # print("running MENTS")
    # ments_results = evaluate_solver(MentSolver, iterations_list)
    # print("MENTS Results:", ments_results)

