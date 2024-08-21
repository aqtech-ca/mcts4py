import time

from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.GymMentsSolver import MentSolver
from samples.gym.cartPole.cartPoleWrapper import CartPoleMDP


def evaluate_solver(solver_class, iterations_list, time_limit):
    start = time.time()
    results = []

    for iterations in iterations_list:

        gym_mdp = CartPoleMDP(time_limit)
        solver = solver_class(
            mdp=gym_mdp,
            simulation_depth_limit=100,
            exploration_constant=1.0,
            discount_factor=0.99,
            verbose=False
        )

        state = gym_mdp.initial_state()
        total_reward = 0
        step_count = 0
        max_steps = 100

        while not solver.mdp.is_terminal(state, step_count) and step_count < max_steps:
            solver.run_search(iterations)
            best_action = solver.do_best_action(solver.root())
            state, reward, done, _, _ = gym_mdp.env.step(best_action)
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

            if done:
                print("pole fell")
                break

        duration = time.time() - start
        results.append({
            "iterations": iterations,
            "step_count": step_count,
            "total_reward": total_reward,
            "duration": duration,
        })

        print({
            "iterations": iterations,
            "step_count": step_count,
            "total_reward": total_reward,
            "duration": duration,
        })

    return results


if __name__ == "__main__":
    iterations_list = [1, 2, 3, 5, 10, 50, 100, 500, 1000, 5000]

    print("running UCT")
    uct_results = evaluate_solver(GenericSolver, iterations_list, 30)

    print("running MENTS")
    ments_results = evaluate_solver(MentSolver, iterations_list, 30)
