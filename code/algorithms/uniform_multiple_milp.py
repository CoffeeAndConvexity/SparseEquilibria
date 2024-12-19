import gurobipy as gp
from gurobipy import GRB
import numpy as np

def uniform_multiple_milp(A, B, K):

    """_summary_
    Bounded support player is the row player

    Args:
        A (matrix): utility matrix of the defender (row player)
        B (matrix): utility matrix of the attacker (column player)
        K (int): support bound for the sparse player 

    Returns:
        decimal: value of the game 
    """
    
    n, m = A.shape[0], B.shape[1]

    max_obj = -np.inf

    for b in range(m):

        model = gp.Model(f"MILP_{b}")
        model.Params.outputFlag = 0

        # Set up decision variables
        x = [model.addVar(lb=0.0, ub=1.0, name=f"x{i}") for i in range(n)]
        z = [model.addVar(lb=0.0, ub=K, vtype=GRB.INTEGER, name=f"z{i}") for i in range(n)]

        # Set objective function
        obj = sum(A[a][b] * x[a] for a in range(n))
        model.setObjective(obj, GRB.MAXIMIZE)

        # Add constraints
        s1 = sum(B[a][b] * x[a] for a in range(n))
        for b_prime in range(m):
            s2 = sum(B[a][b_prime] * x[a] for a in range(n))
            constr_expr = s2 - s1    
            model.addConstr(constr_expr <= 0)

        for i in range(n):
            model.addConstr(x[i] == z[i]/K)

        model.addConstr(sum(x) == 1, "sum_x")
        model.addConstr(sum(z) == K, "sum_z")
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


