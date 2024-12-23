from stable_baselines3 import DQN
from samples.gym.GymGenericSolver import GenericSolver
from samples.gym.GymMentsSolver import MENTSSolverV1
from samples.gym.cartPole.cartPoleWrapper import CartPoleMDP
import gymnasium


def benchmark_baseline():
    env = gymnasium.make('CartPole-v1')
    model = DQN("MlpPolicy", env, verbose=1)
    total_rewards_baseline = 0
    for _ in range(100):
        done = False
        while not done:
            action = model(obs[None])[0]
            obs, reward, done, _, _ = env.step(action)
            total_rewards_baseline += reward
    print(f"Baseline Total Reward: {total_rewards_baseline}")


# Benchmark MENTS Solver
def benchmark_ments():
    env = CartPoleMDP()
    ments_solver = MENTSSolverV1(env, simulation_depth_limit=100, exploration_constant=0.5, discount_factor=0.99)
    total_rewards_ments = 0
    for _ in range(100):
        done = False
        while not done:
            ments_solver.run_search(100)
            action = ments_solver.do_best_action(ments_solver.root())
            obs, reward, done, _, _ = env.step(action)
            total_rewards_ments += reward
    print(f"MENTS Solver Total Reward: {total_rewards_ments}")


if __name__ == "__main__":
    benchmark_baseline()
    benchmark_ments()
