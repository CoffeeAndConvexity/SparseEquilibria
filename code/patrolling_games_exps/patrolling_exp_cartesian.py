from university_patrolling_game import generate_university_patrolling_game
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithms.cartesian_mip import cartesian_mip
from sys import argv
import time
import csv
import numpy as np


def run_cartesian_mip(utility_matrix, S_hat, K):
    tic = time.time()
    obj_val = cartesian_mip(utility_matrix, S_hat, K)
    tac = time.time()
    return tac-tic, obj_val

def save_output(depth, seed, K, output_csv_path, run_time, obj_val):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([depth, K, seed, run_time, obj_val])

def structure_space(sp_minimizer, paths_minimizer):
	sp_indexes = {sp:[] for sp in sp_minimizer}
	for idx,path in enumerate(paths_minimizer):
		sp_indexes[path[2]].append(idx)
	return sp_indexes


def university_exp_patrolling_cartesian(depth, seed, output_dir: Path):
      
    sp_minimizer = [7, 9, 14, 19, 29, 35, 36, 39]
    high_value_targets = [13, 30, 32, 37, 40]

    # generate patrolling game
    utility_matrix, _, paths_minimizer, game = generate_university_patrolling_game(seed_num=seed,depth=depth,plot_setting=False,starting_points_minimizer = sp_minimizer,high_value_targets = high_value_targets)
    utility_matrix = np.transpose(utility_matrix)

    # structure the space
    sp_indexes = structure_space(sp_minimizer, paths_minimizer)
    s_hat = [[sp_indexes[s] for s in sp_indexes]]

    output_csv_path = output_dir / "cartesian_mip_output.csv"

    # run and save
    for K in range(1, len(sp_minimizer)+1):
        run_time, obj_val = run_cartesian_mip(utility_matrix, s_hat, [K])
        save_output(depth, seed, K, output_csv_path, run_time, obj_val)

if __name__== "__main__":
    try:
        depth = int(sys.argv[1])
        seed = int(sys.argv[2])
        output_dir = Path(sys.argv[3]).resolve()
        university_exp_patrolling_cartesian(depth, seed, output_dir)
    except KeyboardInterrupt:
        sys.exit(130)