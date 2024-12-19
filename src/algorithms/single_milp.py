import gurobipy as gp
from gurobipy import GRB


def single_milp(A, B, support_size):

    """_summary_
    Bounded support player is the row player

    Args:
        A (matrix): utility matrix of the defender (row player)
        B (matrix): utility matrix of the attacker (column player)
        support_size (int): support bound for the sparse player

    Returns:
        decimal: value of the game
    """

    n, m = len(A), len(A[0])

    model = gp.Model("single_milp")

    # Set up decision variables
    r = {}
    z = [model.addVar(vtype=GRB.BINARY, name=f"z_{a}") for a in range(n)]
    x = [model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=1, name=f"x_{a}") for a in range(n)]
    y = [model.addVar(vtype=GRB.BINARY, name=f"y_{b}") for b in range(m)]
    for a in range(n):
        for b in range(m):
            r[a, b] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"r_{a}_{b}")

    # Set objective function
    obj_expr = gp.LinExpr()
    for a in range(n):
        for b in range(m):
            obj_expr += A[a][b] * r[a, b] 
    model.setObjective(obj_expr, GRB.MAXIMIZE)

    # Add constraints
    for b_prime in range(m):
        constraint_expr = gp.LinExpr()
        for a in range(n):
            rhs_sum = B[a][b_prime] * x[a]
            lhs_sum = 0
            for b in range(m):
                lhs_sum += B[a][b] * r[a, b]
            constraint_expr += (lhs_sum - rhs_sum)
        model.addConstr(constraint_expr >= 0, f"constraint_lower_{b_prime}")
        model.addConstr(constraint_expr <= 100000 * (1 - y[b_prime]), f"constraint_upper_{b_prime}")
    
    for b in range(m):
        const_expr = gp.LinExpr()
        for a in range(n):
            const_expr += r[a,b]
        model.addConstr(const_expr == y[b], f"constraint_eq_{b}") 
    
    model.addConstr(sum(x) == 1, "sum_x")
    model.addConstr(sum(y) == 1, "sum_y")
    for a in range(n):
        model.addConstr(x[a] <= z[a], f"indicator_{a}")
    model.addConstr(sum(z) <= support_size, "bounded_support")

    for a in range(n):
        for b in range(m):
            model.addConstr(r[a, b] <= x[a])

    # Optimize the model
    model.optimize()
    obj_val = model.ObjVal

    return obj_val 
