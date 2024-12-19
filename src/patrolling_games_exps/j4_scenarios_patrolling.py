from university_patrolling_game import generate_university_patrolling_game
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithms.mip import mip
from algorithms.cartesian_mip import cartesian_mip
from sys import argv

depth = int(argv[1])
seed = int(argv[2])
k = int(argv[3])
scenario = int(argv[4])

if scenario == 0:
	high_value_targets = [9,13,37,35] 
	sp_minimizer = [2,4,10,24,33] 
elif scenario == -1:
	high_value_targets = None
	sp_minimizer = None
elif scenario == 1:
	sp_minimizer = [1, 22, 27, 33, 36, 37, 39]
	high_value_targets = [1, 20, 21, 38]
elif scenario == 2:
	sp_minimizer = [0, 12, 13, 29, 38]
	high_value_targets = [7, 19, 29, 36, 38]
elif scenario == 3:
	sp_minimizer = [6, 8, 16, 18, 36]
	high_value_targets = [2, 9, 14, 18, 30, 40]
elif scenario == 4:
	sp_minimizer = [9, 11, 21, 25, 33, 36, 37, 41]
	high_value_targets = [3, 19, 32, 33]
elif scenario == 5:
	sp_minimizer = [7, 9, 14, 19, 29, 35, 36, 39]
	high_value_targets = [13, 30, 32, 37, 40]
else:
	assert False, "Unknown scenario: " + str(scenario)

matrix, _, paths_minimizer, game = generate_university_patrolling_game(seed,depth,plot_setting=False,starting_points_minimizer = sp_minimizer,high_value_targets = high_value_targets)

num_rows, num_cols = matrix.shape
print(f"Size of the game matrix: {num_rows} rows and {num_cols} columns")

sp_indexes = {sp:[] for sp in sp_minimizer}
for idx,path in enumerate(paths_minimizer):
	sp_indexes[path[2]].append(idx)


