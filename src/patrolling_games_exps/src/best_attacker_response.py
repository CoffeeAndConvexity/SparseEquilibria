from __future__ import annotations 
import gurobipy as gp
from collections import defaultdict
from copy import deepcopy

from src.games.layered_game import LayeredGame
# from src.solvers.attacker_br_relaxation import AttackerBRRelaxation
# from src.util.layered_graph import find_all_paths, find_all_paths_efficient

class BestAttackerResponse(object):
    def __init__(self,
                 gurobi_env,
                 game: LayeredGame,
                 initial_subgame_d: list[list[int]],
                 initial_dist_d: list[int],
                 speedups: set = {},
                 verbose = False):
        """
        speedups: set of optimizations. Currently supports
            - 'SOS1'
            - 'min_x_relax'
            - 'atk_lifting1'
            - 'trim_impossible_pairs'
            - 'simple_atk_speedup'
            - 'binary_y'
        """

        self.gurobi_env = gurobi_env
        self.game = game
        self.subgame_d = deepcopy(initial_subgame_d)

        self.speedups = speedups
        self.verbose = verbose

        self._construct_initial_MIP(initial_subgame_d, initial_dist_d)

    def _construct_initial_MIP(self, subgame_d, dist_d: list[float], output_flag: int = 0):
        subgame_size_d = len(subgame_d)
        assert len(dist_d) == subgame_size_d
        br_a = gp.Model("br_a", env=self.gurobi_env)
        # br_a.Params.LogToConsole = 0
        if self.verbose:
            br_a.Params.OutputFlag = 1

        br_a_var_yq = self._construct_reweighted_y(br_a)
        br_a_var_y = self._construct_y_variables(br_a)
        br_a_var_z = self._construct_z_variables(br_a)
        br_a_var_x = self._construct_x_variables(br_a)
    
        br_a_cons_trg = self._construct_z_constraints(
            br_a, br_a_var_z, br_a_var_x)
        br_a_cons_yq = self._construct_yq_constraints(br_a, br_a_var_y, br_a_var_yq, dist_d)
        br_a_cons_trg_yq = self._construct_yq_z_constraints(br_a, br_a_var_z, br_a_var_yq)
        br_a_cons_x_y = self._construct_x_y_constraints(br_a, br_a_var_y, br_a_var_x)

        br_a_cons_flow, br_a_cons_source = self._flow_constraints(
            br_a, br_a_var_x)
        br_a.ModelSense = gp.GRB.MAXIMIZE

        self.br_a = br_a
        self.br_a_var_y = br_a_var_y  # Will be expanded in updates
        self.br_a_var_yq = br_a_var_yq  # Will be included in updates
        self.br_a_var_x = br_a_var_x  # Will be included in updates
        self.br_a_var_z = br_a_var_z # Will be included in updates
        self.br_a_cons_yq = br_a_cons_yq  # Will be included in updates
        self.br_a_cons_trg = br_a_cons_trg  # Won't be included in updates
        self.br_a_cons_trg_yq = br_a_cons_trg_yq # Won't be included in updates.
        self.br_a_cons_x_y = br_a_cons_x_y    # Won't be included in updates
        self.br_a_cons_flow = br_a_cons_flow  # Won't be included in updates
        self.br_a_cons_source = br_a_cons_source  # Won't be included in updates

        # if 'min_x_relax' in self.speedups:
        #     self._construct_min_x_constraints(br_a, br_a_var_x, br_a_var_y)
        if 'relax_hint' in self.speedups:
            self.lp_relaxation = AttackerBRRelaxation(self.gurobi_env, 
                self.game, 
                initial_subgame_d=self.subgame_d, 
                initial_dist_d=dist_d, verbose=False)
        if 'atk_lifting1' in self.speedups:
            self.br_a_subpaths_var_dict, self.subpath_marginal_cons = \
                self._construct_lifted_variables(br_a, br_a_var_x)
            self.br_a_cons_path_lifted = \
                self._caught_probability_constraints_lifted(br_a, br_a_var_y, self.br_a_subpaths_var_dict)
        if 'trim_impossible_pairs' in self.speedups:
            impossible_pairs = self._get_impossible_pairs()
            impossible_pair_constrs = self._construct_impossible_pair_constraints(br_a, br_a_var_x, impossible_pairs)
        if 'branch_priority_preterminal' in self.speedups:
            pass

        self.solve()

    def _construct_y_variables(self, br_a):
        """ A single `y` variable in [0, 1], where the lower bound is implicit 
            (unless the speedup `binary_y` is included) is added for each defender 
            path. y_p for path p is intended to be 1 when the attacker is caught
            by the defender path y_p (conditioned on the event that the defender chose to
            play path p).
        """
        if 'binary_y' in self.speedups:
            # Binary y variables allow for them to be chosen to branched on,
            # while their relaxation still remains the same.
            # TODO: what about branching priority compared to x variables?
            br_a_var_y = [br_a.addVar(ub=1.0, 
                                    vtype=gp.GRB.BINARY,
                                    name="y("+str(p)+")")
                        for p in range(len(self.subgame_d))]
        else:
            br_a_var_y = [br_a.addVar(ub=1.0, 
                                    name="y("+str(p)+")")
                        for p in range(len(self.subgame_d))]


        return br_a_var_y

    def _construct_reweighted_y(self, br_a):
        """ Construct sum of reweighted y's
        """
        br_a_var_yq = br_a.addVar(ub=1.0, name="yq")

        return br_a_var_yq

    def _construct_z_variables(self, br_a):
        br_a_var_z = [br_a.addVar(ub=1.0,
                                  obj=self.game.target_vals[target],
                                  name="z("+str(target)+")")
                      for target in range(len(self.game.atk_adj[-1]))]

        return br_a_var_z

    def _construct_x_variables(self, br_a):
        br_a_var_x = dict()
        for L in range(self.game.num_layers()-1):

            if 'SOS1' in self.speedups:
                SOS_list = []

            for v1 in range(len(self.game.atk_adj[L])):
                if len(self.game.atk_adj[L][v1]) == 0:
                    continue
                for v2 in self.game.atk_adj[L][v1]:

                    br_a_var_x[(L, v1, v2)] = br_a.addVar(
                            vtype=gp.GRB.BINARY,
                            name="x("+str(L)+","+str(v1)+","+str(v2)+")")
                    if 'branch_priority_preterminal' in self.speedups and \
                        L == self.game.num_layers()-2:
                        br_a_var_x[(L, v1, v2)].setAttr('BranchPriority', 1)

                    if 'SOS1' in self.speedups:
                        SOS_list.append(br_a_var_x[(L, v1, v2)])
            
            if 'SOS1' in self.speedups:
                # Method 1: Just include SOS directly in Gurobi
                br_a.addSOS(gp.GRB.SOS_TYPE1 , SOS_list)

                # Method 2: Add in SOS constraints via layers.
                # br_a.addConstr(gp.quicksum(SOS_list) <= 1)

        return br_a_var_x

    def _construct_z_constraints(self, br_a, br_a_var_z, br_a_var_x):
        br_a_cons_trg = [br_a_var_z[trg_idx]
                         for trg_idx in range(len(self.game.target_vals))]
        for preterminal_layer_node_idx in range(len(self.game.atk_adj[-2])):
            for target in self.game.atk_adj[-2][preterminal_layer_node_idx]:
                br_a_cons_trg[target] -= br_a_var_x[(
                    self.game.num_layers()-2, preterminal_layer_node_idx, target)]
        for target in range(len(self.game.target_vals)):
            br_a_cons_trg[target] = br_a.addConstr(
                br_a_cons_trg[target] <= 0.0)

        return br_a_cons_trg

    def _construct_yq_constraints(self, br_a, br_a_var_y, br_a_var_yq, dist_d):
        # Compute weighted summation over defender paths in subgame.
        yq_target = 0
        for path_idx in range(len(self.subgame_d)):
            yq_target += br_a_var_y[path_idx] * dist_d[path_idx]

        br_a_cons_yq = br_a.addConstr(br_a_var_yq - yq_target == 0.0)

        return br_a_cons_yq


    def _construct_yq_z_constraints(self, br_a, br_a_var_z, br_a_var_yq):
        br_a_cons_trg_yq = []
        for target in range(len(self.game.atk_adj[-1])):
            br_a_cons_trg_yq.append(br_a.addConstr(
                br_a_var_z[target] + br_a_var_yq <= 1.0))
        
        return br_a_cons_trg_yq


    def _construct_x_y_constraints(self, br_a, br_a_var_y, br_a_var_x):
        br_a_cons_x_y = []

        for path_idx in range(len(self.subgame_d)):
            interdicting_edges = self.game.closeness_fn_def_path(
                self.subgame_d[path_idx])
            
            if not 'simple_atk_speedup' in self.speedups:
                for L in range(self.game.num_layers()-1):
                    for edge in interdicting_edges[L]:
                        br_a_cons_x_y.append(br_a.addConstr(
                            br_a_var_y[path_idx] - br_a_var_x[(L, edge[0], edge[1])] >= 0.0))
            elif 'simple_atk_speedup' in self.speedups:
                # `simple_atk_speedup` strengthens the constraint on each y variable.
                # Instead of lower bounding y by every close edge to defender path p,
                # group these close edges into L (or L-1) layers. The lower bound of
                # that y variable is the *sum* of the x variables for edges in each 
                # group of close edges. The idea here is that the sum of x's in each group
                # will never be strictly greater than 1 (which would cause this inequality
                # to be unsatifiable always).                
                for L in range(self.game.num_layers()-1):
                    tmp = br_a_var_y[path_idx]
                    for edge in interdicting_edges[L]:
                        tmp -= br_a_var_x[(L, edge[0], edge[1])]
                    br_a_cons_x_y.append(br_a.addConstr(tmp >= 0))

        return br_a_cons_x_y

    def _caught_probability_constraints_lifted(self, 
            br_a, 
            br_a_var_y, 
            br_a_lifted_variables):
        """ Constructs the constraints governing the lifted variables.
        """
        br_a_cons_path = []
        for path_idx in range(len(self.subgame_d)):
            # br_d_cons_path.append(br_d_var_y[path_idx])
            interdicting_edges = self.game.closeness_fn_atk_path(
                self.subgame_d[path_idx])
            
            if 'simple_atk_speedup' in self.speedups:
                d = defaultdict(lambda: [])
                for (L, subpath), var in br_a_lifted_variables.items():
                    for t in range(len(subpath)-1):
                        if (subpath[t], subpath[t+1]) in interdicting_edges[L+t]:
                            # br_a_cons_path.append(br_a.addConstr(br_a_var_y[path_idx] - var >= 0))
                            d[L].append(var)
                            break 
                for loc, vars in d.items():
                    br_a_cons_path.append(br_a.addConstr(br_a_var_y[path_idx] - gp.quicksum(vars) >= 0))

            else:
                for (L, subpath), var in br_a_lifted_variables.items():
                    for t in range(len(subpath)-1):
                        if (subpath[t], subpath[t+1]) in interdicting_edges[L+t]:
                            br_a_cons_path.append(br_a.addConstr(br_a_var_y[path_idx] - var >= 0))
                            break # No need to continue, that megaedge has already intersected.
            

        return br_a_cons_path

    def _flow_constraints(self, br_a, br_a_var_x):
        """ Constructs the following constraints for the defender best response.
        1) Flow conservation constraints at all internal nodes.
        2) Sum-to-one constraint at the single source node.

        Returns tuple of:
            (1) in a dictionary mapping (layer_id, node)
        and (2) as a single constraint.
        """
        num_layers_a = len(self.game.atk_adj)

        # Flow constraints over internal vertices
        br_a_cons_flow = defaultdict(lambda: 0)
        for L in range(num_layers_a - 1):
            for node in range(len(self.game.atk_adj[L])):
                if L > 0:
                    for succ in self.game.atk_adj[L][node]:
                        # TODO: direct_sum more efficient?
                        br_a_cons_flow[(L, node)
                                       ] -= br_a_var_x[(L, node, succ)]
                if L < len(self.game.atk_adj)-2:
                    for succ in self.game.atk_adj[L][node]:
                        # TODO: direct_sum more efficient?
                        br_a_cons_flow[(L+1, succ)
                                       ] += br_a_var_x[(L, node, succ)]
        for node in br_a_cons_flow:
            br_a_cons_flow[node] = br_a.addConstr(br_a_cons_flow[node] == 0.0)

        # Special flow constraint for the source layer.
        br_a_cons_source = br_a_var_x[(0, 0, self.game.atk_adj[0][0][0])]
        for i in range(1, len(self.game.atk_adj[0][0])):
            br_a_cons_source += br_a_var_x[(0, 0, self.game.atk_adj[0][0][i])]
        br_a_cons_source = br_a.addConstr(br_a_cons_source == 1.0)

        return br_a_cons_flow, br_a_cons_source

    """
    def _construct_min_x_constraints(self, br_a, br_a_var_x, br_a_var_y):
        assert len(br_a_var_y) == len(self.subgame_d)

        br_a_cons_min_x_ub = []

        for path_idx in range(len(self.subgame_d)):
            interdicting_edges = self.game.closeness_fn_def_path(
                self.subgame_d[path_idx])
            num_interdicting_edges = sum([len(x) for x in interdicting_edges])

            cons_to_add = []
            for L in range(self.game.num_layers()-1):
                for edge in interdicting_edges[L]:
                    cons_to_add.append(br_a.addConstr(
                        br_a_var_x[(L, edge[0], edge[1])] - br_a_var_y[path_idx]/num_interdicting_edges >= 0.0))
                    br_a_cons_x_y.append(br_a.addConstr(
                        br_a_var_y[path_idx] - br_a_var_x[(L, edge[0], edge[1])] >= 0.0))

            br_a_cons_min_x_ub.append(cons_to_add)

            for L in range(self.game.num_layers()-1):
                for edge in interdicting_edges[L]:
                    br_a_cons_x_y.append(br_a.addConstr(
                        br_a_var_y[path_idx] - br_a_var_x[(L, edge[0], edge[1])] >= 0.0))
    """        

    def insert_new_defender_path(self, br_d_path):
        self.subgame_d.append(deepcopy(br_d_path))

        # Insert one additional y variable (probability of not being caught).
        self.br_a_var_y.append(self.br_a.addVar(
            ub=1.0, name="y("+str(len(self.subgame_d)-1)+")"))

        # Construct additional constraint for the newly inserted y variable
        # that relates y to x. For the defender, we will have one new constraint
        # for each interdicting edge.
        interdicting_edges = self.game.closeness_fn_def_path(br_d_path)
        for L in range(self.game.num_layers()-1):
            for edge in interdicting_edges[L]:
                self.br_a.addConstr(
                    self.br_a_var_y[-1] - self.br_a_var_x[(L, edge[0], edge[1])] >= 0.0)

        if 'relax_hint' in self.speedups:
            self.lp_relaxation.insert_new_defender_path(br_d_path)
        if 'atk_lifting1' in self.speedups:
            # Insert target y variable in constraint (we will replace this with 
            # an actual constraint later.)
            interdicting_edges = self.game.closeness_fn_atk_path(br_d_path)
            
            for (L, subpath), var in self.br_a_subpaths_var_dict.items():
                for t in range(len(subpath)-1):
                    if (subpath[t], subpath[t+1]) in interdicting_edges[L+t]:
                        self.br_a_cons_path_lifted.append(self.br_a.addConstr(self.br_a_var_y[-1] - var >= 0))
                        break

    def update_defender_distribution(self, dist_d):
        self.br_a.remove(self.br_a_cons_yq)
        yq_target = 0
        for path_idx in range(len(self.subgame_d)):
            yq_target += self.br_a_var_y[path_idx] * dist_d[path_idx]
        self.br_a_cons_yq = self.br_a.addConstr(
            self.br_a_var_yq - yq_target == 0.0)

        if 'relax_hint' in self.speedups:
            self.lp_relaxation.update_defender_distribution(dist_d)

    def _construct_lifted_variables(self, br_a, br_a_var_x):        
        num_layers = self.game.num_layers()
        atk_adj = self.game.atk_adj

        subpaths_var_dict = dict()
        subpaths_var_by_edge_containment = dict()
        for layer_id in range(0, num_layers-1, 2):
            for start_id in range(len(atk_adj[layer_id])):
                subpaths = find_all_paths_efficient(atk_adj, 
                    start=start_id, 
                    layer=layer_id,
                    max_layer=min([layer_id+2,num_layers-1]))
                for subpath in subpaths:
                    subpath_var = br_a.addVar(
                        lb=0.0,
                        ub=1.0,
                        name='lift('+str(layer_id)+','+str(subpath)+')')

                    # Add subpath into all edge containment lists
                    for layer_offset in range(len(subpath)-1):
                        intermediate_layer = layer_offset + layer_id
                        sid, eid = subpath[layer_offset], subpath[layer_offset+1]
                        if (intermediate_layer, sid, eid) not in subpaths_var_by_edge_containment:
                            subpaths_var_by_edge_containment[(intermediate_layer, sid, eid)] = []
                        subpaths_var_by_edge_containment[(intermediate_layer, sid, eid)].append(subpath_var)
                    
                    # Store subpath variables in dictionary for easy acccess
                    subpaths_var_dict[(layer_id, subpath)] = subpath_var

        print('number of subpaths', len(subpaths_var_dict))

        # Add product constraints.
        pdt_constraints = []
        for (layer_id, subpath), subpath_var in subpaths_var_dict.items():
            if len(subpath) <= 2: continue
            e1 = subpath[0], subpath[1]
            e2 = subpath[1], subpath[2]
            ell1, ell2 = layer_id, layer_id + 1
            v1 = br_a_var_x[(ell1, e1[0], e1[1])]
            v2 = br_a_var_x[(ell2, e2[0], e2[1])]

            # Method 1: manually linearize using 3 linear constraints.
            # See: https://www.gurobi.com/events/models-with-products-of-binary-variables/
            # for more details
            pdt_constraints.append(br_a.addConstr(subpath_var <= v1))
            pdt_constraints.append(br_a.addConstr(subpath_var <= v2))
            pdt_constraints.append(br_a.addConstr(subpath_var >= v1+v2-1))

            # Method 2: Use Gurobi to do it automatically
            # pdt_constraints.append(br_a.addConstr(subpath_var == v1 * v2))
            # br_a.setParam(gp.GRB.Param.PreQLinearize, 1) # PreQLinearize uses 3 linear constraints per product.

        # Add sum of flow constraints, i.e., 
        # Relevant flows must sum to give the edge flows given by the x variables (which
        # when relaxed are flows on a single edge.)
        sum_of_subpath_flows_constr = []
        # for (L, sid, eid), subpath_vars in subpaths_var_by_edge_containment.items():
        for L in range(num_layers-1):
            for start_vertex in range(len(atk_adj[L])):
                for end_vertex in atk_adj[L][start_vertex]:
                    constr = br_a_var_x[(L, start_vertex, end_vertex)] 
                    for subpath_var in subpaths_var_by_edge_containment[(L, start_vertex, end_vertex)]:
                        constr -= subpath_var

                    # Should this condition be here...?
                    if len(subpaths_var_by_edge_containment[(L, start_vertex, end_vertex)]) > 0:
                        pass
                        # TODO: put the flow constraints back? or remove them entirely?
                        # sum_of_subpath_flows_constr.append(
                        #     br_a.addConstr(constr == 0.)
                        # )


        print('Number of aux constr', len(sum_of_subpath_flows_constr))

        return subpaths_var_dict, sum_of_subpath_flows_constr

    def _construct_impossible_pair_constraints(self, br_a, br_a_var_x, impossible_pairs: dict):
        pairwise_forbidden_constraints = []
        for (L_target, idx_target), forbidden_vertices in impossible_pairs.items():
            if L_target >= self.game.num_layers()-1:
                continue
            for target_end in self.game.atk_adj[L_target][idx_target]:
                edge = (L_target, idx_target, target_end)
                edge_x_var = br_a_var_x[edge]
                for L_source, idx_source in forbidden_vertices:
                    for source_end in self.game.atk_adj[L_source][idx_source]:
                        edge_source = (L_source, idx_source, source_end)
                        edge_source_x_var = br_a_var_x[edge_source]
                        
                        pairwise_forbidden_constraints.append(br_a.addConstr(edge_x_var + edge_source_x_var <= 1))
        
        print('Number of forbidden pairwise constraints:', len(pairwise_forbidden_constraints))
        return pairwise_forbidden_constraints
                        

    def _get_impossible_pairs(self):
        """ Iterate over all target vertices v (could be internal), and 
            use DP to find the set S(v) = {u | path u->v exists} and 
            \bar{S}(v) = {u | path u->v does not exist}. 

            Returns \bar{S}(v) for all v, i.e.,
            ret[(target_layer, target_idx)] = set containing (L, idx) tuples
            where L < target layer and idx is the source vertex id in layer L.
        """
        ret = dict()
        for L_target in range(1, self.game.num_layers()):
            for v_target in range(len(self.game.atk_adj[L_target])):
                ret[(L_target, v_target)] = set()

                prev_reachable = [False] * len(self.game.atk_adj[L_target])
                prev_reachable[v_target] = True
                for L_source in reversed(range(L_target)):
                    reachable = [False] * len(self.game.atk_adj[L_source])
                    for v_source in range(len(self.game.atk_adj[L_source])):
                        for next_vertex in self.game.atk_adj[L_source][v_source]:
                            if prev_reachable[next_vertex]: 
                                reachable[v_source] = True

                    for v_source in range(len(self.game.atk_adj[L_source])):
                        if not reachable[v_source]:
                            ret[(L_target, v_target)].add((L_source, v_source))

                    prev_reachable = reachable

        return ret

    def _get_path_val(self, path: list[int]):
        final_target = path[-1]
        target_val = self.game.target_vals[final_target]

        return target_val

    def get_objective(self):
        return self.br_a.ObjVal

    def get_path(self, eps=1e-5):
        br_a_path = (0,)
        prev_node = 0
        for L in range(self.game.num_layers()-1):
            for node in self.game.atk_adj[L][prev_node]:
                if abs(1.0 - self.br_a_var_x[(L, prev_node, node)].X) < eps:
                    br_a_path += (node,)
                    prev_node = node
                    break
        return br_a_path

    def solve(self):
        if 'relax_hint' in self.speedups:
            hint_path = self.lp_relaxation.get_approximate_br()
            print(hint_path)

            """
            # Method using MIP starts
            for _, z in self.br_a_var_x.items():
                z.Start = 0
            for L in range(self.game.num_layers()-1):
                edge = hint_path[L], hint_path[L+1]
                self.br_a_var_x[(L, edge[0], edge[1])].Start = 1
            """

            # Method using Hints
            for L in range(self.game.num_layers()-1):
                edge = hint_path[L], hint_path[L+1]
                self.br_a_var_x[(L, edge[0], edge[1])].VarHintVal = 1
            
        self.br_a.optimize()
        
        if self.br_a.Status == gp.GRB.OPTIMAL:
            return 'exact'
        elif self.br_a.Status == gp.GRB.TIME_LIMIT:
            return 'inexact'
        elif self.br_a.Status == gp.GRB.INFEASIBLE:
            return 'infeasible'
        else:
            return 'unexpected_status'

    def set_time_limit(self, t: int):
        self.br_a.setParam('TimeLimit', t)