from src.worlds.world import StreetMapWorld
from src.games.bomb_planting_game import BombPlantingGame
from src.games.binary_pursuit_game import BinaryPursuitGame
from src.games.layered_game import LayeredGame
from src.util.layered_graph import find_all_paths_efficient, get_neighboring_nodes, path_clash

from single_oracle_nonsparse import SingleOracle

from numpy.random import default_rng
import numpy as np
import networkx as nx
from sys import argv
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from algorithms.mip import mip

def get_matrix(university_game):
	target_values = university_game.target_vals
	paths_maximizer = find_all_paths_efficient(university_game.atk_adj)
	paths_minimizer = find_all_paths_efficient(university_game.def_adj)
	num_paths_maximizer = len(paths_maximizer)
	num_paths_minimizer = len(paths_minimizer)


	university_game_matrix = [[0.0 for _ in range(num_paths_minimizer)] for _ in range(num_paths_maximizer)]

	for path_student_idx in range(num_paths_maximizer):
		for path_prof_idx in range(num_paths_minimizer):
			university_game_matrix[path_student_idx][path_prof_idx] = 0.0 if path_clash(university_game, len(university_game.atk_adj),
					paths_maximizer[path_student_idx], paths_minimizer[path_prof_idx]) else target_values[paths_maximizer[path_student_idx][-1]]
	
	return np.array(university_game_matrix), paths_maximizer, paths_minimizer

def generate_university_patrolling_game(
		seed_num,
		depth, 
		starting_points_minimizer = None,
		high_value_targets = None,
		consolidate_tolerance = 10, 
		normalizing_distance = 10000,
		):

	university_world = StreetMapWorld("Redwood National and State Parks, California, USA", 
		consolidate_tolerance=consolidate_tolerance, normalizing_distance=normalizing_distance, 
		osmnx_nx_type = "all_private", directed=False, allow_stay=True)

	num_nodes_discretized = university_world.convert_to_networkx_graph().number_of_nodes()
	num_nodes_osm = university_world.get_underlying_osm().number_of_nodes()

	defender_graph = nx.Graph()
	defender_graph.add_nodes_from([i for i in range(num_nodes_osm)])
	for node in defender_graph.nodes():
		defender_graph.add_edge(node, node)

	### set up targets
	generator = default_rng(seed_num)
	target_values = [generator.integers(low=1,high=5) for _ in range(num_nodes_osm)] + [0.0 for _ in range(num_nodes_discretized - num_nodes_osm)]  

	if starting_points_minimizer == None:
		starting_points_minimizer = list(set([generator.integers(low=0,high=num_nodes_osm) for _ in range(generator.integers(low=4,high=9))]))


	### add some high value targets
	if consolidate_tolerance == 10:
		if high_value_targets == None:
			high_value_targets = list(set([generator.integers(low=0,high=num_nodes_osm) for _ in range(generator.integers(low=4,high=9))]))
		for h in high_value_targets: target_values[h] = generator.integers(low=6,high=10)
	else:
		assert False, 'Targets not chosen for this particular tolerance.'

	bpg = BinaryPursuitGame(university_world.convert_to_networkx_graph(), defender_graph, 
						starting_points_minimizer, [i for i in range(num_nodes_osm)], target_values, depth, 'exact_edge') # class closeness not used
	university_game = LayeredGame(bpg.atk_adj, bpg.def_adj, target_values,
						LayeredGame.MakeExactVertexClosenessFn(bpg.def_adj),
						LayeredGame.MakeExactVertexClosenessFn(bpg.atk_adj))

	return university_game

depth = int(argv[1])
seed = int(argv[2])
k = int(argv[3])

scenario = 0

if scenario == 0:
	high_value_targets = [9, 13, 39, 35] 
	sp_minimizer = [2, 4, 10, 24, 32, 47]
else:
	assert False, "Unknown scenario: " + str(scenario)

game = generate_university_patrolling_game(seed,depth,starting_points_minimizer = sp_minimizer,high_value_targets = high_value_targets)


solver = SingleOracle(game, k)
solver.run()

print(solver.value)


CHECK_AGAINST_MIP = False
if CHECK_AGAINST_MIP:
	matrix, _, _ = get_matrix(game)
	sol, mip_val = mip(matrix,k)
	print(mip_val)


