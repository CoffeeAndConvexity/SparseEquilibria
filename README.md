# SparseEquilibria
This repository contains Python scripts for computing $k$-sparse commitments in two-player games where one player is restricted to
mixed strategies with support size at most $k$, as studied in our paper. The efficiency of our proposed methods is demonstrated empirically using randomly generated normal-form games and settings based on security applications.

# Main Algorithms 
1. `mip.py`: computes $k$-sparse Nash equilibirum for normal-form two-player zero-sum games where the row player has support size at most $k$.
2. `multiple_milp.py` and `single_milp.py`: compute $k$-sparse Stackelberg equilibirum for normal-form two-player general-sum games where the row player has support size at most $k$. `multiple_milp.py` solves one Mixed Integer Linear Program (MILP) for each action of the column player, while `single_milp.py` combines everything into a single large MILP. 
3. `cartesian_mip.py`: computes sparse Nash equilibirum for scenarios where sparsity is imposed in a structured manner: there are multiple constraint sets, each containing action sets and having a specified sparsity constraint.


# Applications

## Randomly generated normal-form games 
Random zero-sum and general-sum games are the simplest of the implemented games. These are normal-form games where payoffs are square matrices that are randomly generated. Zero-sum games are solved using the proposed mip algorithm. In general-sum games, we impose the condition that corresponding entries in the payoff matrices of the two players have opposite signs. These games are solved using the multiple_milp and single_milp algorithms.

## Patrolling games (Pursuit-Evasion)
Our Pursuit-Evasion game is designed on the Columbia University campus path network and models a finite-horizon scenario where a defender (pursuer) tries to capture an attacker (evader). Each player starts at one of several potential initial locations on campus corresponding to vertices, and selects a path of specific length (also known as depth); the evader is apprehended if and only if the chosen paths share an edge. 

This game is generated using the script `university_patrolling_game.py`. This script was created by *Cerny et al. Layered Graph Security Games (2024)*. This generator supports the use of real-world maps via the `osmnx` library. 
We solve this game using the mip algorithm in `patralling_exp_vanilla.py`, and using the cartesian_mip in `patrolling_exp_structured`. We also solve the game using the *lgsg solver* from Cerny et al. in `lgsg_solver_exp`. 

# Dependencies
- Python 
- `numpy`: for numerical computations. Install via:
```
pip install numpy
```
- `osmnx`: for generating real-world map data. Install via:
```
pip install osmnx
```

# Citation
If you use this repository, please cite our paper:
```
@inproceedings{afiouni2025commitment,
 author = {Salam Afiouni and Jakub Cerny and Chun Kai Ling and Christian Kroer},
 booktitle = {AAAI},
 date = {2025-02},
 note = {},
 title = {Commitment to Sparse Strategies in Two-Player Games},
 year = {2025}
}
```



