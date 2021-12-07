import PSTN_class as PSTN
import sys
import convex
from gurobipy import GRB
import gurobipy as gp
from math import sqrt

inf = sys.float_info.max

b0 = PSTN.timePoint("b0", "Initial Time Point")
e0 = PSTN.timePoint("e0", "e0")
b1 = PSTN.timePoint("b1", "b1")

c00 = PSTN.constraint("c00", b0, e0, "pstc", {"lb": 0, "ub": inf, "value": 1}, distribution = {"type": "gaussian", "mean": 5, "variance": 2})
c01 = PSTN.constraint("c01", b0, b1, "stc", {"lb": 3, "ub": 5, "value": 1}, hard=False)
c10 = PSTN.constraint("c10", b1, e0, "stc", {"lb": 1, "ub": 2, "value": 1})

p = PSTN.PSTN("toy problem", [b0, e0, b1], [c00, c01, c10])
mat = convex.get_matrices(p, 10)
A, vars, b, c, T_l, T_u, q_l, q_u, mu, cov = mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8], mat[9]

m = convex.get_initial_points(A, vars, b, T_l, T_u, q_l, q_u)
if m.status == GRB.OPTIMAL:
    for var in m.getVars():
        print(var.varName, var.x)
    m.write("{}.sol".format("toy_problem"))
else:
    raise ValueError("Could not find initial points")
