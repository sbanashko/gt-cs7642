import numpy as np

from project3.utils import lp_util

# Test matrix for Maximin
A = np.array([[3, -2, 2], [-1, 0, 4], [-4, -3, 1]])

sol = lp_util.maxmin(A)
probs = sol
print(probs)
# [ 1.67e-01]
# [ 8.33e-01]
# [ 0.00e+00]

# Test matrices for CE
Q1 = np.array([[6, 2], [7, 0]])
Q2 = np.array([[6, 7], [2, 0]])

sol = lp_util.ce(Q1, Q2)
probs = sol
print(probs)
# [ 5.00e-01]
# [ 2.50e-01]
# [ 2.50e-01]
# [ 0.00e+00]
