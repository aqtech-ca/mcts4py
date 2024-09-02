import gym
import copy


class RevertibleFrozenLakeEnv(gym.Wrapper):
    def __init__(self, env_name):
        super(RevertibleFrozenLakeEnv, self).__init__(gym.make(env_name, is_slippery=False))
        self.state_history = []

    def reset(self, **kwargs):
        obs = super(RevertibleFrozenLakeEnv, self).reset(**kwargs)
        self.state_history = [(copy.deepcopy(obs), self.env.np_random.bit_generator.state)]
        return obs

    def step(self, action):
        obs, reward, done, trunc, info = super(RevertibleFrozenLakeEnv, self).step(action)
        self.state_history.append((copy.deepcopy(obs), self.env.np_random.bit_generator.state))
        return obs, reward, done, trunc, info

    def revert_step(self):
        if len(self.state_history) > 1:
            self.state_history.pop()  # Remove the last state
            obs, rng_state = self.state_history[-1]  # Get the previous state and RNG state
            self.env.np_random.bit_generator.state = rng_state
            return obs
        else:
            raise Exception("No previous state to revert to")


# Example usage:
lake = gym.make('FrozenLake-v1', is_slippery=False)
lake.reset()

state, reward, done, trunc, _ = lake.step(1)
print(f"state: {state}, reward: {reward}, done: {done}, trunc: {trunc}")
state, reward, done, trunc, _ = lake.step(2)
print(f"state: {state}, reward: {reward}, done: {done}, trunc: {trunc}")
state, reward, done, trunc, _ = lake.step(3)
print(f"state: {state}, reward: {reward}, done: {done}, trunc: {trunc}")
state, reward, done, trunc, _ = lake.step(0)
print(f"state: {state}, reward: {reward}, done: {done}, trunc: {trunc}")
