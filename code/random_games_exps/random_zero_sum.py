from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithms.mip import mip
import time
import csv
import numpy as np
import os


def run_mip(utility_matrix, support_size):
    tic = time.time()
    obj_val = mip(utility_matrix, support_size)
    tac = time.time()
    return tac-tic, obj_val

def test_single_mult_milp_exp(matrix_size, support_size, i, output_dir: Path):
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    output_csv_path = output_dir / f"test_cluster_random_zero_sum_{matrix_size}.csv"

    seed_num = seeds[i]
    rng = np.random.default_rng(seed_num)
    utility_matrix = rng.integers(10, 101, size=(matrix_size, matrix_size))
    utility_matrix = np.transpose(utility_matrix)
    # run mip    
    t, obj_val = run_mip(utility_matrix, support_size)

    # save to csv
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([support_size, i, t, obj_val])

    print(obj_val)

if __name__== "__main__":
    # run python random_zero_sum.py from bash with following parameters 
    # {support_size} {game_number} {output_dir} 
    try:
        matrix_size = int(sys.argv[1])
        support_size = int(sys.argv[2])
        game_number = int(sys.argv[3])
        output_dir = Path(sys.argv[4]).resolve()
        test_single_mult_milp_exp(matrix_size, support_size, game_number, output_dir)
    except KeyboardInterrupt:
        sys.exit(130)
        
