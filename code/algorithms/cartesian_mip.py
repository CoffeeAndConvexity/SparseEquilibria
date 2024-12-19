import gurobipy as gp
from gurobipy import GRB
import numpy as np
from itertools import product

def cartesian_mip(
    A,
    S_hat, 
    K
    ):
    """_summary_
    Bounded support player is the row player

    Args:
        A (2-dim matrix): payoff matrix for zero-sum game 
        S_hat (array of arrays of sub-arrays): super-set containing sets of action sets (actions are represented by their indices)
        K (array): support bound for every set in S_hat

    Returns:
        x (array): probability distribution over the actions of the sparse player
        obj_val: value of the game
    """
    
    num_rows = len(A)
    num_cols = len(A[0])

    ## Create model
    model = gp.Model("cartesianMIP")
    model.Params.outputFlag = 0
    
    ## Create variables
    g = model.addVar(lb=-float("inf"), name="g")
    x = [model.addVar(lb=0.0, ub=1.0, name=f"x{i}") for i in range(num_rows)]
    ind = [model.addVars(len(Si), vtype=GRB.BINARY, name=f"S{i+1}") for i, Si in enumerate(S_hat)]

    ## Set objective
    model.setObjective(g, GRB.MINIMIZE) 

    ## Add constraints 
    for j in range(num_cols):
        model.addConstr(g >= sum(A[i][j] * x[i] for i in range(num_rows)), f"c_col_{j}")

    model.addConstr(sum(x) == 1, "sum_x")

    for i, S in enumerate(S_hat):
        for index, set in enumerate(S):
            for action in set:
                model.addConstr(x[action] <= ind[i][index])

    for i, set_indicators in enumerate(ind):
        model.addConstr(set_indicators.sum() <= K[i])

    try:
        model.optimize()

    except Exception:
        # Handle infeasible model
        print("Infeasible model encountered")
       
    return model.ObjVal

