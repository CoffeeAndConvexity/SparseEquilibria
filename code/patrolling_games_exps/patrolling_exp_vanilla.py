from patrolling_games_exps.university_patrolling_game import generate_university_patrolling_game
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithms.mip import mip
from sys import argv
import time
import csv
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

def run_mip(utility_matrix, K):
	tic = time.time()
	mip_val = mip(utility_matrix, K)
	tac = time.time()
	return tac-tic, mip_val

def save_output(seed, K, output_csv_path, run_time, obj_val):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([K, seed, run_time, obj_val])

def university_exp_patrolling(depth, seed, output_dir: Path):
	
	sp_minimizer = [7, 9, 14, 19, 29, 35, 36, 39]
	high_value_targets = [13, 30, 32, 37, 40]

	# generate payoff matrix and space structure from patrolling game
	utility_matrix, _, paths_minimizer, game = generate_university_patrolling_game(seed_num=seed,depth=depth,plot_setting=False,starting_points_minimizer = sp_minimizer,high_value_targets = high_value_targets)
	utility_matrix = np.transpose(utility_matrix)
	num_rows, num_columns = utility_matrix.shape
	
	output_csv_path = output_dir / "vanilla_mip_output.csv"

	# run vanilla mip
	for K in range (1, 21):
		run_time, obj_val = run_mip(utility_matrix, K)
		save_output(seed, K, output_csv_path, run_time, obj_val)
    
	run_time, obj_val = run_mip(utility_matrix, num_rows)
	save_output(seed, num_rows, output_csv_path, run_time, obj_val)

if __name__== "__main__":
    try:
        depth = int(sys.argv[1])
        seed = int(sys.argv[2])
        output_dir = Path(sys.argv[3]).resolve()
        university_exp_patrolling(depth, seed, output_dir)
    except KeyboardInterrupt:
        sys.exit(130)
        
        