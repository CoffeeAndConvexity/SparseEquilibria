import gurobipy as gp
from gurobipy import GRB
import numpy as np

from random import Random

from src.util.layered_graph import find_all_paths_efficient, get_neighboring_nodes, path_clash
from src.best_attacker_response import BestAttackerResponse
from src.games.layered_game import LayeredGame


class SingleOracle():
    def __init__(self,
                 game: LayeredGame,
                 support_bound: int,
                 seed_num: int = 0,
                 epsilon: float = 1e-3,
                 init_size_subgame=10,
                 log_to_console=False,
                 callback = None,
                 verbose = False):
        self.game = game
        self.num_layers = len(self.game.atk_adj)

        self.support_bound = support_bound

        self.seed_num = seed_num
        
        self.random_generator = Random()
        self.random_generator.seed(seed_num)

        self.epsilon = epsilon
        self.current_gap = None
        self.final_gap = None
        # self.final_subgames = None
        self.init_size_subgame = init_size_subgame
        self.log_to_console = log_to_console

        self.callback = callback
        self.verbose = verbose

    def _sample_path(self, G):
        sampled = False
        while not sampled:  # Keep resampling until we get a path
            path = (0,)
            sampled = True
            for i in range(1, self.num_layers):
                num_suc = len(G[i-1][path[i-1]])
                if num_suc == 0:
                    sampled = False
                    break
                path += (G[i-1][path[i-1]][self.random_generator.randint(0, num_suc-1)],)
        return path


    def run(self):

        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()


        fullgame_d = find_all_paths_efficient(self.game.def_adj)
        subgame_a = list(set([self._sample_path(self.game.atk_adj) for _ in range(self.init_size_subgame)]))

        num_rows = len(subgame_a)
        num_cols = len(fullgame_d)

        # Create model
        m = gp.Model("NashLP_ColPlayer", env=env)
        # m.Params.outputFlag = 0
        # m.Params.MIPGap = 1e-12
        # Create variables
        y = [m.addVar(lb=0.0, ub=1.0, name=f"y{j}") for j in range(num_cols)]
        indicator_y = [m.addVar(vtype=GRB.BINARY, name=f"ind_y{j}") for j in range(num_cols)]
        z = m.addVar(lb=-float("inf"), name="z")
        m.setObjective(z, GRB.MINIMIZE) 
        for j in range(num_cols):
            m.addConstr(y[j] <= indicator_y[j])
        m.addConstr(gp.quicksum(y) == 1, "sum_y")
        m.addConstr(gp.quicksum(indicator_y) <= self.support_bound, "bounded_support")


        # Add constraints 
        for i in range(num_rows):
            utilities = [0.0 if path_clash(self.game, len(self.game.atk_adj),subgame_a[i], fullgame_d[j]) else self.game.target_vals[subgame_a[i][-1]] for j in range(num_cols)]
            m.addConstr(z >= gp.quicksum([utilities[j] * y[j] for j in range(num_cols)]), f"c_row_{i}")

        m.optimize()


        dist_d = [y[j].X for j in range(num_cols)]
        best_responder_a = BestAttackerResponse(env, self.game, fullgame_d, dist_d, verbose = self.verbose)


        self.current_gap = abs(m.ObjVal - best_responder_a.get_objective())

        while self.current_gap > self.epsilon:

            if self.callback:
                self.callback()

            # add constraint to MIP
            br_a_path = best_responder_a.get_path()
            subgame_a.append(br_a_path)
            utilities = [0.0 if path_clash(self.game, len(self.game.atk_adj),br_a_path, fullgame_d[j]) else self.game.target_vals[br_a_path[-1]] for j in range(num_cols)]
            m.addConstr(z >= gp.quicksum([utilities[j] * y[j] for j in range(num_cols)]), f"c_row_{len(subgame_a)}")

            # resolve MIP
            m.optimize()

            # update BR
            dist_d = [y[j].X for j in range(num_cols)]
            best_responder_a.update_defender_distribution(dist_d)

            # solve BR
            status_br_a = best_responder_a.solve()

            # compute gap
            self.current_gap = abs(m.ObjVal - best_responder_a.get_objective())

        self.final_gap = self.current_gap
        self.value = m.ObjVal


