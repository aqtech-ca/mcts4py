from mcts4py.NodeClasses import *
from mdp.GridWorldMDP import *
from mcts4py.StatefulSolver import *

rewards = [GridworldReward(5, 0, -20), GridworldReward(4, 3, 99.0)]
x_size = 8
y_size = 5
tp = 0.8

reward_locations = []

for r in rewards:
    reward_locations.append((r.state.x, r.state.y))

gwMDP = GridworldMDP(x_size, y_size, rewards, tp, GridworldState(6, 2, False))

######


gwMDP.visualizeState(gwMDP.initialState())
next_state = gwMDP.transition(gwMDP.initialState(), "DOWN")
# # print("\n")
# # gwMDP.visualizeState(next_state)

solver = StatefulSolver(gwMDP, verbose = True)
solver.runTreeSearch(99)

solver.displayTree(True)
# gwMDP.visualizeState(gwMDP.initialState())
# print(str(solver.extractOptimalAction()))

# voting_states = []
# for i in range(15):
#     solver = StatefulSolver(gwMDP, verbose = False)
#     solver.runTreeSearch(99)
#     voting_states.append(str(solver.extractOptimalAction()))

# gwMDP.visualizeState(gwMDP.initialState())
# print(voting_states)
# print(max(set(voting_states), key=voting_states.count))


