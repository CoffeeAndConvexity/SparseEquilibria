import gurobipy as gp
from gurobipy import GRB
import numpy as np

def uniform_mip(
    utility_matrix,
    K
    ):
    '''_summary_
    Bounded support player is the row player

    Args:
        utility_matrix: payoff matrix for zero-sum game 
        K (int): support bound for the row player
        
    Returns:
        x (array): probability distribution over the actions of the sparse player
        obj_val: value of the game
    '''

    num_rows = len(utility_matrix)
    num_cols = len(utility_matrix[0])

    # Create model
    m = gp.Model("NashLP_ColPlayer")
    m.Params.outputFlag = 0
    
    # Create variables
    x = [m.addVar(lb=0.0, ub=1.0, name=f"x{i}") for i in range(num_rows)]
    z = [m.addVar(lb=0.0, ub=K, vtype=GRB.INTEGER, name=f"z{i}") for i in range(num_rows)]
    g = m.addVar(lb=-float("inf"), name="g")

    m.setObjective(g, GRB.MINIMIZE) 

    # Add constraints 
    for j in range(num_cols):
        m.addConstr(g >= sum(utility_matrix[i][j] * x[i] for i in range(num_rows)), f"c_col_{j}")

    for i in range(num_rows):
        m.addConstr(x[i] == z[i]/K)
    m.addConstr(sum(z) == K, "sum_z")
    m.addConstr(sum(x) == 1, "sum_x")

    m.optimize()

    return m.ObjVal
