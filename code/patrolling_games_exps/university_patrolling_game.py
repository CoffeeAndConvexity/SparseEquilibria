from src.worlds.world import StreetMapWorld
from src.games.bomb_planting_game import BombPlantingGame
from src.games.binary_pursuit_game import BinaryPursuitGame
from src.games.layered_game import LayeredGame
from src.util.layered_graph import find_all_paths_efficient, get_neighboring_nodes, path_clash

from numpy.random import default_rng
import numpy as np
import networkx as nx


def generate_university_patrolling_game(
		seed_num,
		depth, 
		starting_points_minimizer = None,
		high_value_targets = None,
		consolidate_tolerance = 10, 
		normalizing_distance = 10000,
		plot_setting = False,
		):

	university_world = StreetMapWorld("Columbia University, USA", 
		consolidate_tolerance=consolidate_tolerance, normalizing_distance=normalizing_distance, 
		osmnx_nx_type = "all_private", directed=False, allow_stay=True)

	num_nodes_discretized = university_world.convert_to_networkx_graph().number_of_nodes()
	num_nodes_osm = university_world.get_underlying_osm().number_of_nodes()

	attacker_graph = nx.Graph()
	attacker_graph.add_nodes_from([i for i in range(num_nodes_osm)])
	for node in attacker_graph.nodes():
		attacker_graph.add_edge(node, node)

	### set up targets
	# seed_num = 0
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

	if plot_setting:
		university_world.draw_annotated_map(attacker_points = high_value_targets, defender_points = starting_points_minimizer)
	print(sorted(starting_points_minimizer))
	print(sorted(high_value_targets))

	bpg = BinaryPursuitGame(attacker_graph, university_world.convert_to_networkx_graph(), 
						[i for i in range(num_nodes_osm)], starting_points_minimizer, target_values, depth, 'exact_edge') # class closeness not used
	university_game = LayeredGame(bpg.atk_adj, bpg.def_adj, target_values,
						LayeredGame.MakeExactVertexClosenessFn(bpg.def_adj),
						LayeredGame.MakeExactVertexClosenessFn(bpg.atk_adj))

	target_values = university_game.target_vals


	paths_maximizer = find_all_paths_efficient(university_game.atk_adj)
	paths_minimizer = find_all_paths_efficient(university_game.def_adj)
	num_paths_maximizer = len(paths_maximizer)
	num_paths_minimizer = len(paths_minimizer)


	university_game_matrix = [[0.0 for _ in range(num_paths_minimizer)] for _ in range(num_paths_maximizer)]

	### make sure the matrix has the same format as your games
	for path_student_idx in range(num_paths_maximizer):
		for path_prof_idx in range(num_paths_minimizer):
			university_game_matrix[path_student_idx][path_prof_idx] = 0.0 if path_clash(university_game, len(university_game.atk_adj),
					paths_maximizer[path_student_idx], paths_minimizer[path_prof_idx]) else target_values[paths_maximizer[path_student_idx][-1]]
	
	return np.array(university_game_matrix), paths_maximizer, paths_minimizer, university_game