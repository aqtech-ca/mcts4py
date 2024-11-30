import time
import copy

from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.GymMentsSolver import MentSolver
from samples.gym.cartPole.cartPoleWrapper import CartPoleMDP


def evaluate_solver(solver_class, iterations_list, goals, trials=10):
    exploration_constant = 0.1
    results = []

    for goal in goals:
        print(f"goal: {goal}, exploration_constant: {exploration_constant}")

        for iterations in iterations_list:
            success_count = 0
            total_step_count = 0
            total_reward_sum = 0
            total_duration = 0

            for trial in range(trials):
                start = time.time()
                gym_mdp = CartPoleMDP()
                solver = solver_class(
                    mdp=gym_mdp,
                    simulation_depth_limit=100,
                    exploration_constant=exploration_constant,
                    discount_factor=0.99,
                    verbose=False
                )

                total_reward = 0
                step_count = 0

                while step_count < goal:
                    solver.run_search(iterations)
                    best_action = solver.do_best_action(solver.root())
                    state, reward, done, _, _ = gym_mdp.env.step(best_action)

                    total_reward += reward
                    step_count += 1

                    gym_mdp.initial = state
                    solver = solver_class(
                        mdp=gym_mdp,
                        simulation_depth_limit=100,
                        exploration_constant=exploration_constant,
                        discount_factor=0.99,
                        verbose=False
                    )

                    if done:
                        cart_position, cart_velocity, pole_angle, pole_velocity = state
                        if solver.verbose:
                            if abs(cart_position) > 2.4:
                                print(f"trial: {trial + 1}, The cart moved out of bounds!")
                            elif abs(pole_angle) > 0.2095:
                                print(f"trial: {trial + 1}, The pole fell beyond 12 degrees!")
                        break
                    elif step_count >= goal:
                        if solver.verbose:
                            print(f"Trial {trial + 1}: success {step_count} steps")
                        success_count += 1
                        break

                duration = time.time() - start
                total_step_count += step_count
                total_reward_sum += total_reward
                total_duration += duration

                gym_mdp.close()
            avg_step_count = total_step_count / trials
            avg_success = success_count / trials
            avg_duration = total_duration / trials

            results.append({
                "iterations": iterations,
                "avg_step_count": avg_step_count,
                "avg_success": avg_success,
                "avg_duration": avg_duration,
            })
            print({
                "iterations": iterations,
                "avg_step_count": avg_step_count,
                "avg_success": avg_success,
                "avg_duration": avg_duration,
            })
            if success_count == 10:
                break

    return results


if __name__ == "__main__":
    iterations_list = [1, 2, 3, 5, 10, 50, 100, 500, 1000, 1500, 2000, 2500]
    goals = [10, 20, 50, 100, 150, 200, 250, 300]

    print("running UCT")
    uct_results = evaluate_solver(GenericSolver, iterations_list, goals)

    print("running MENTS")
    ments_results = evaluate_solver(MentSolver, iterations_list, goals)
