import gym
import numpy as np
from numpy.random import PCG64, Generator


class CustomInitialStateEnv(gym.Wrapper):
    def __init__(self, env_name, initial_state=None):
        super(CustomInitialStateEnv, self).__init__(gym.make(env_name))
        self.initial_state = initial_state

    def reset(self, **kwargs):

        obs = super(CustomInitialStateEnv, self).reset(**kwargs)

        if self.initial_state is not None:
            self.env.state = self.initial_state
            obs = np.array(self.initial_state)

        return obs


# # Example initial state for CartPole (assuming 4-dimensional state)
# initial_state = [0.0, 0.0, 0.1, 0.0]  # position, velocity, angle, angular velocity
#
# # Create the custom environment with the initial state
# env = CustomInitialStateEnv('CartPole-v1', initial_state=initial_state)
#
# # Reset the environment (this will set the initial state to the defined one)
# print(env.reset())
#
#
# # Example initial state for CartPole (assuming 4-dimensional state)
# initial_state = [0.555, 0.0333, 0.1, 0.0]  # position, velocity, angle, angular velocity
#
# # Create the custom environment with the initial state
# env = CustomInitialStateEnv('CartPole-v1', initial_state=initial_state)
# print(env.reset())



lake = gym.make("FrozenLake-v1", render_mode="human", is_slippery=False)
lake.reset()
state, reward, done, trunc, _ = lake.step(1)
print(f"state: {state}, reward: {reward}, done: {done}, trunc: {trunc}")

state, reward, done, trunc, _ = lake.step(1)

print(f"state: {state}, reward: {reward}, done: {done}, trunc: {trunc}")

