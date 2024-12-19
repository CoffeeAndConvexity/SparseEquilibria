from university_patrolling_game import generate_university_patrolling_game
from sparse_lgsg_solver import SparseLGSGSolver
import sys
import numpy as np


def run_solver(game, K):
    solver = SparseLGSGSolver(game, K)
    solver.solve()
    solver_val = solver.get_objective()
    return solver_val

def university_exp_solver(depth, seed, K): 
	
    sp_minimizer = [7, 9, 14, 19, 29, 35, 36, 39]
    high_value_targets = [13, 30, 32, 37, 40]

    # generate payoff matrix and space structure from patrolling game
    utility_matrix, _, paths_minimizer, game = generate_university_patrolling_game(seed_num=seed,depth=depth,plot_setting=False,starting_points_minimizer = sp_minimizer,high_value_targets = high_value_targets)
    utility_matrix = np.transpose(utility_matrix)

    obj_val = run_solver(game, K) 
    print(depth, seed, K, obj_val)

if __name__== "__main__":
    try:
        depth = int(sys.argv[1])
        seed = int(sys.argv[2])
        K = int(sys.argv[3])
        university_exp_solver(depth, seed, K)
    except KeyboardInterrupt:
        sys.exit(130)
        
