from mcts4py.GenericSolver import GenericSolver
from samples.simple_american_call_binomial import AmMDP

AmMDP1 = AmMDP(K=80,S=80,T=50,r=0)

solver = GenericSolver(
    AmMDP1,
    simulation_depth_limit = 70,
    exploration_constant = 1.0,
    discount_factor = 1,
    verbose = True)  

solver.run_search(999)

solver.display_tree()