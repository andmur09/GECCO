from cvxpy.constraints.nonpos import NonNeg
from cvxstoc import expectation, prob, CategoricalRandomVariable
from cvxpy import Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.transforms import partial_optimize
import numpy

# Create problem data.
c, s, r, b = 10, 25, 5, 150
d_probs = [0.3, 0.6, 0.1]
d_vals = [55, 139, 141]
d = CategoricalRandomVariable(d_vals, d_probs)

# Create optimization variables.
x = Variable(nonneg = True)
y, z = Variable(nonneg = True), Variable(nonneg = True)

# Create second stage problem.
obj = -s*y - r*z
constrs = [y+z<=x, y<=d, z<=d]
p2 = Problem(Minimize(obj), constrs)
Q = partial_optimize(p2, [y, z], [x])

# Create and solve first stage problem.
p1 = Problem(Minimize(c*x + expectation(Q, num_samples=100)), [x<=b])
p1.solve()