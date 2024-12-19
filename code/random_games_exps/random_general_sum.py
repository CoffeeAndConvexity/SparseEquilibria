from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithms.multiple_milp import multiple_milp
from algorithms.single_milp import single_milp
import time
import csv
import sys
import numpy as np
import os


def run_single_milp(A, B, k):
    tic = time.time()
    obj_val = single_milp(A, B, support_size=k)
    tac = time.time()
    return tac - tic, obj_val

def ensure_opposite_signs(A, B):
    assert A.shape == B.shape
    sign_A = np.sign(A)
    sign_B = np.sign(B)
    mask = sign_A == sign_B
    B[mask] *= -1
    return B

coeffs = [-0.8, -0.6, -0.4, -0.2]
A_seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
B_seeds = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

def run_and_save(A, B, support_size, output_csv_path, i, coeff):

    run_time, obj_val = run_single_milp(A, B, support_size)  

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([coeff, support_size, i, run_time, obj_val])


def random_general_sum(matrix_size, support_size, i, output_dir: Path):

    output_csv_path = output_dir / f"random_gen_sum_single_milp_{matrix_size}.csv"

    RANGE = (-50, 51)

    # generate random payoff matrix A
    A_seed_num = A_seeds[i]
    A_rng = np.random.default_rng(A_seed_num)
    A = A_rng.integers(*RANGE, size=(matrix_size, matrix_size))

    # generate random payoff matrix B negatively correlated     
    B_seed_num = B_seeds[i]
    B_rng = np.random.default_rng(B_seed_num)
    B = B_rng.integers(*RANGE, size=(matrix_size, matrix_size))
    B = ensure_opposite_signs(A, B)

    run_and_save(A, B, support_size, output_csv_path, i, coeff='None')

    # generate B with corr coeff 
    for c in coeffs:

        K = 85
        noise_seed = int(-100 * c)
        noise_rng = np.random.default_rng(noise_seed)
        N = noise_rng.integers(-K, K+1, size=A.shape)

        B = c * A + N
        # E(AB) = E ( c * A^2 + N * A) = c * E( A^2 ) = c * Var(A) same sign as c
        
        corr_coeff = np.corrcoef(A.flatten(), B.flatten())[0, 1] # element wise correlation coefficient 
        
        run_and_save(A, B, support_size, output_csv_path, i, coeff=corr_coeff)


if __name__== "__main__":
    # run python random_general_sum.py with the following parameters 
    # {support_size} {game_number} {output_dir} 
    try:
        matrix_size = int(sys.argv[1])
        support_size = int(sys.argv[2])
        game_number = int(sys.argv[3])
        output_dir = Path(sys.argv[4]).resolve()
        random_general_sum(matrix_size, support_size, game_number, output_dir)
    except KeyboardInterrupt:
        sys.exit(130)
  