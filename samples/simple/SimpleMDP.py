from mcts4py.MDP import *
import numpy as np

class SimpleMDP(MDP):

    def __init__(self, initial_state = "a"):
        self.initial_counter = 100
        self.counter = self.initial_counter
        self.initial_state = initial_state

    def transition(self, state, action):
        self.counter -= 1
        if action == "1":
            if state == "a" and np.random.uniform(0, 1) > 0.5:
                return "b"
            elif state == "b" and np.random.uniform(0, 1) > 0.5:
                return "a"
        if action == "2":
            if state == "a" and np.random.uniform(0, 1) > 0.5:
                return "b"
            elif state == "b" and np.random.uniform(0, 1) > 0.5:
                return "a"

        return state

    def reward(self, previous_state, action, state):
        if action == "1":
            if previous_state == "a" and state == "b":
                return 1.5
            if previous_state == "b" and state == "a":
                return 1.49
            if previous_state == state:
                return 1.52
        if action == "2":
            if previous_state == "a" and state == "b":
                return 1.5
            if previous_state == "b" and state == "a":
                return 1.51
            if previous_state == state:
                return 1.49
        return 0.0

    def initialState(self):
        return self.initial_state

    def actions(self, state):
        return ["1", "2"]

    def isTerminal(self, state):
        return False
        if state == "b":
            return True
        else:
            return False
        # if self.counter < 1:
        #     return True
        # else:
        #     return False

    def reset(self):
        self.counter = self.initial_counter
        self.state = self.initialState()


