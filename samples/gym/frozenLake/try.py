import gym
import copy

from samples.gym.frozenLake.frozenLakeWrapper import FrozenLakeMDP


class CustomFrozenLakeEnv(gym.RewardWrapper):
    def __init__(self, env):
        super(CustomFrozenLakeEnv, self).__init__(env)

    def reward(self, reward):

        print(self.env.unwrapped.s)

        if self.env.unwrapped.s in [5, 7, 11, 12]:
            return -10
        elif self.env.unwrapped.s == 15:
            return 10
        else:
            return -0.01


is_slippery = False
env = FrozenLakeMDP(is_slippery)

state = env.initial_state()
print(f"Initial State: {state}")

print(env.reward(0, 1))
state, reward, _, _, _ = env.step(1)
print(f"reward: {reward}, New State: {state}")
