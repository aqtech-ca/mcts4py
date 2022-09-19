from mcts4py.MDP import *
import numpy as np
import random


# reward object
# x, y, value

class GridworldState():
    def __init__(self, x, y, is_terminal = True):
        self.x = x
        self.y = y
        self.is_terminal = is_terminal
    
    def isNeighbourValid(self, action, x_size, y_size):
        if action == "UP" and self.y == y_size - 1:
            return False
        if action == "DOWN" and self.y == 0:
            return False
        if action == "RIGHT" and self.x == x_size - 1:
            return False
        if action == "LEFT" and self.x == 0:
            return False
        return True
    
    def resolveNeighbour(self, action, x_size, y_size):
        if action == "UP":
            if self.y == y_size - 1:
                return None
            else:
                return GridworldState(self.x, self.y+1, False)
        if action == "RIGHT":
            if self.x == x_size - 1:
                return None
            else:
                return GridworldState(self.x + 1, self.y, False)
        if action == "DOWN":
            if self.y == 0:
                return None
            else:
                return GridworldState(self.x, self.y - 1, False)
        if action == "LEFT":
            if self.x == 0:
                return None
            else:
                return GridworldState(self.x - 1, self.y, False)
    
    def __str__(self):
        return "[" + str(self.x) + ", " + str(self.y) + "]"


class GridworldReward():
    def __init__(self, x, y, value):
        self.state = GridworldState(x, y)
        self.value = value


class GridworldMDP(MDP):

    def __init__(self, 
        x_size = 4,
        y_size = 5,
        rewards = [GridworldReward(0, 0, 1)],
        transition_probability = 0.8,
        starting_location = (0, 0)):

        self.x_size = x_size
        self.y_size = y_size
        self.rewards = rewards
        self.transition_probability = transition_probability
        self.starting_location = starting_location 

        self.all_actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def initialState(self):
        return self.starting_location

    
    def isTerminal(self, state):
        for r in self.rewards:
            if r.state == state:
                return True
        return False
    
    def visualizeState(self):
        arr = np.zeros([self.x_size, self.y_size], dtype = str)
        state_array = np.full_like(arr, "-")
        state_array[self.starting_location[0], self.starting_location[1]] = "A"

        for r in self.rewards:
            if r.value > 0:
                state_array[r.x, r.y] = "*"
            if r.value < 0:
                state_array[r.x, r.y] = "X"

        print(state_array)

        return None
    
    # state represented as repr([x, y])
    def reward(self, previous_state, action, state):
        for r in self.rewards:
            if repr([r.state.x, r.state.y]) == repr([state.x, state.y]):
                return r.value 
        return 0.0
    
    def transition(self, state, action):
        if state.is_terminal:
            return state
        
        term_states = [repr([r.state.x, r.state.y]) for r in self.rewards]
        if repr([state.x, state.y]) in term_states:
            return state
        
        # reward states?

        target_neighbour = state.resolveNeighbour(action, self.x_size, self.y_size)
        if target_neighbour is None:
            return state
        
        if np.random.uniform() < self.transition_probability:
            return target_neighbour
        else:
            non_target_neighbours = []
            remaining_actions = list(set(self.all_actions) - set([action]))
            for a in remaining_actions:
                possible_neighbour = state.resolveNeighbour(a, self.x_size, self.y_size)
                non_target_neighbours.append(possible_neighbour)
            
            if len(non_target_neighbours) > 0:
                return random.choice(non_target_neighbours)
            else:
                raise Exception("No valid neighbours exist")
        
        return state

    def actions(self, state):        
        return [a for a in self.all_actions if state is not None and state.isNeighbourValid(a, self.x_size, self.y_size)]





