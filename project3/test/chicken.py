import numpy as np

from project3.utils.lp_util import LinProg

Q1 = np.array([[6, 2], [7, 0]])
Q2 = np.array([[6, 7], [2, 0]])

# A = [[6, 6], [2, 7], [7, 2], [0, 0]]

solver = LinProg()
sol = solver.ce(Q1, Q2, solver="glpk")
probs = sol
print(probs)
# [ 5.00e-01]
# [ 2.50e-01]
# [ 2.50e-01]
# [ 0.00e+00]
