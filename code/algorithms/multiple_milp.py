import gurobipy as gp
from gurobipy import GRB
import numpy as np

def multiple_milp(A, B, support_size):

    """_summary_
    Bounded support player is the row player

    Args:
        A (matrix): utility matrix of the defender (row player)
        B (matrix): utility matrix of the attacker (column player)
        support_size (int): support bound for the sparse player 

    Returns:
        decimal: value of the game 
    """
    n = len(A)
    m = len(B[0])
    max_obj = -np.inf

    for b in range(m):

        model = gp.Model(f"MILP_{b}")
        model.Params.outputFlag = 0

        # Set up decision variables
        x = [model.addVar(lb=0.0, ub=1, name=f"x_{a}") for a in range(n)]
        z = [model.addVar(vtype=GRB.BINARY, name=f"z_{a}") for a in range(n)]

        # Set objective function
        obj = sum(A[a][b] * x[a] for a in range(n))
        model.setObjective(obj, GRB.MAXIMIZE)

        # Add constraints
        s1 = sum(B[a][b] * x[a] for a in range(n))
        for b_prime in range(m):
            s2 = sum(B[a][b_prime] * x[a] for a in range(n))
            constr_expr = s2 - s1    
            model.addConstr(constr_expr <= 0)

        for a in range(n):
            model.addConstr(x[a] <= z[a])

        model.addConstr(sum(x) == 1, "sum_x")
        model.addConstr(sum(z) <= support_size, "bounded_support")
        model.addConstr(obj >= max_obj, "max_obj")
        
        try:
            # Optimize the model
            model.optimize()
            obj_val = model.ObjVal
            if obj_val > max_obj:
                max_obj = obj_val
                max_model= model
            
        except Exception:
            # Handle infeasible model
            print("Infeasible model encountered. Skipping to the next one.")
                   
    return max_obj 



