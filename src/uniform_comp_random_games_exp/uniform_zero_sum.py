from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithms.uniform_mip import uniform_mip
# import time
import csv
import numpy as np
import os


def run_uniform_mip(utility_matrix, K):
    obj_val = uniform_mip(utility_matrix, support_size)
    return obj_val

def uniform_mip_exp(matrix_size, support_size, i, output_dir: Path):
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    output_csv_path = output_dir / f"uniform_zero_sum_{matrix_size}.csv"

    seed_num = seeds[i]
    rng = np.random.default_rng(seed_num)
    utility_matrix = rng.integers(10, 101, size=(matrix_size, matrix_size))

    # run uniform mip    
    obj_val = run_uniform_mip(utility_matrix, K=support_size)
    # save to csv
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([support_size, i, obj_val])


if __name__== "__main__":
    try:
        matrix_size = int(sys.argv[1])
        support_size = int(sys.argv[2])
        game_number = int(sys.argv[3])
        output_dir = Path(sys.argv[4]).resolve()
        uniform_mip_exp(matrix_size, support_size, game_number, output_dir)
    except KeyboardInterrupt:
        sys.exit(130)
        
