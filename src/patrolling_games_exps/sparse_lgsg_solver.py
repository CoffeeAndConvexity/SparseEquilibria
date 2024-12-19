from __future__ import annotations 
import gurobipy as gp
from collections import defaultdict

from src.games.layered_game import LayeredGame
from src.util.layered_graph import find_all_paths_efficient

class SparseLGSGSolver(object):
    def __init__(self,
                 game: LayeredGame,
                 support_bound: int,
                 verbose = False):

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        self.game = game

        self.support_bound = support_bound

        self.verbose = verbose

        self.model = gp.Model("model", env=env)

        self.utility_var = self.model.addVar(lb=-float("inf"))

        model_var_x, self.model_var_z, self.model_var_t = self._construct_stacked_MIP(self.model, self.support_bound)

        self._construct_interdiction_constraints(self.game, self.model, self.utility_var, self.support_bound, model_var_x, self.model_var_z, self.model_var_t)

    def solve(self):
        self.model.setObjective(self.utility_var, gp.GRB.MINIMIZE)
        self.model.optimize() 

        if self.model.Status == gp.GRB.OPTIMAL:
            return 'exact'
        elif self.model.Status == gp.GRB.TIME_LIMIT:
            return 'inexact'
        elif self.model.Status == gp.GRB.INFEASIBLE:
            return 'infeasible'
        else:
            return 'unexpected_status'

    def get_objective(self):
        return self.model.ObjVal

    def get_solution(self):
        solution_paths = []
        solution_distribution = []
        for path_idx in range(self.support_bound):
            br_d_path = (0,)
            prev_node = 0
            for L in range(self.game.num_layers()-1):
                for node in self.game.def_adj[L][prev_node]:
                    if abs(1- self.model_var_z[(path_idx, L, prev_node, node)].X) < 1e-5:
                        br_d_path += (node,)
                        prev_node = node
                        break
            solution_paths.append(br_d_path)
            solution_distribution.append(self.model_var_t[path_idx].X)
        return solution_paths, solution_distribution

    def _construct_interdiction_constraints(self, game, model, utility_var, support_bound, model_var_x, model_var_z, model_var_t):
        opp_paths = find_all_paths_efficient(game.atk_adj)

        model_var_c = {}
        for path_idx in range(support_bound):
            for opp_path_idx in range(len(opp_paths)):
                grb_varname = "c("+str(path_idx)+","+str(opp_path_idx)+")"
                model_var_c[(path_idx, opp_path_idx)] = model.addVar(
                            vtype=gp.GRB.CONTINUOUS, 
                            ub=1.0,
                            name=grb_varname)

        # BR constraints
        for p_idx, p in enumerate(opp_paths):
            opp_path_vars = [game.target_vals[p[-1]] * model_var_c[(path_idx, p_idx)] for path_idx in range(support_bound)]
            model.addConstr(utility_var >= gp.quicksum(opp_path_vars))

        # interdiction constraints
        for opp_path_idx, opp_path in enumerate(opp_paths):
            interdicting_edges = game.closeness_fn_atk_path(opp_path)
            for path_idx in range(support_bound):
                path_vars = []
                for L in range(game.num_layers()-1):
                    for edge in interdicting_edges[L]:
                        path_vars.append(model_var_x[(path_idx, L, edge[0], edge[1])])
                model.addConstr(model_var_t[path_idx] - gp.quicksum(path_vars) <= model_var_c[(path_idx, opp_path_idx)])



    def _construct_stacked_MIP(self, model, support_bound):
        model_var_t = []
        for path_idx in range(support_bound):
            grb_varname = "t("+str(path_idx)+")"
            model_var_t.append(model.addVar(
                vtype=gp.GRB.CONTINUOUS, 
                ub=1.0,
                name=grb_varname
                    ))
        model.addConstr(gp.quicksum(model_var_t) == 1.0)

        model_var_x, model_var_z = self._construct_x_variables(model, support_bound)

        self._flow_constraints(model, model_var_x, model_var_z, model_var_t, support_bound)

        return model_var_x, model_var_z, model_var_t


    def _construct_x_variables(self, model, support_bound):
        model_var_x = dict()
        model_var_z = dict()
        for L in range(self.game.num_layers() - 1):
            for v1 in range(len(self.game.def_adj[L])):
                if len(self.game.def_adj[L][v1]) == 0:
                    continue
                for v2 in self.game.def_adj[L][v1]:
                    for path_idx in range(support_bound):
                        grb_varname = "z("+str(path_idx)+","+str(L)+","+str(v1)+","+str(v2)+")"
                        model_var_z[(path_idx, L, v1, v2)] = model.addVar(
                            vtype=gp.GRB.BINARY,
                            name=grb_varname)
                        grb_varname = "x("+str(path_idx)+","+str(L)+","+str(v1)+","+str(v2)+")"
                        model_var_x[(path_idx, L, v1, v2)] = model.addVar(
                            vtype=gp.GRB.CONTINUOUS, 
                            ub=1.0,
                            name=grb_varname)
                        model.addConstr(model_var_x[(path_idx, L, v1, v2)] <= model_var_z[(path_idx, L, v1, v2)])

        return model_var_x, model_var_z


    def _flow_constraints(self, model, model_var_x, model_var_z, model_var_t, support_bound):
        """ Constructs the following constraints for the defender best response.
        1) Flow conservation constraints at all internal nodes.
        2) Sum-to-one constraint at the single source node.

        Returns tuple of:
            (1) in a dictionary mapping (layer_id, node)
        and (2) as a single constraint.
        """
        num_layers_d = len(self.game.def_adj)

        # Flow constraints over internal vertices
        model_cons_flow = defaultdict(lambda: [])
        for L in range(num_layers_d - 1):
            for node in range(len(self.game.def_adj[L])):
                if L > 0:
                    for succ in self.game.def_adj[L][node]:
                        model_cons_flow[(L, node)
                                       ].append((L, node, succ))
                if L < num_layers_d-2:
                    for succ in self.game.def_adj[L][node]:
                        model_cons_flow[(L+1, succ)
                                       ].append((L, node, succ))

        for node in model_cons_flow:
            for path_idx in range(support_bound):
                x_vars = [-model_var_x[(path_idx,)+var] if var[0]==node[0] else model_var_x[(path_idx,)+var] for var in model_cons_flow[node]]
                z_vars = [-model_var_z[(path_idx,)+var] if var[0]==node[0] else model_var_z[(path_idx,)+var] for var in model_cons_flow[node]]
                model.addConstr(gp.quicksum(x_vars) == 0)
                model.addConstr(gp.quicksum(z_vars) == 0)

        # Special flow constraint for the source layer
        for path_idx in range(support_bound):
            x_vars_source = [model_var_x[(path_idx, 0, 0, self.game.def_adj[0][0][i])] for i in range(len(self.game.def_adj[0][0]))]
            z_vars_source = [model_var_z[(path_idx, 0, 0, self.game.def_adj[0][0][i])] for i in range(len(self.game.def_adj[0][0]))]
            model.addConstr(gp.quicksum(x_vars_source) == model_var_t[path_idx])
            model.addConstr(gp.quicksum(z_vars_source) == 1)

