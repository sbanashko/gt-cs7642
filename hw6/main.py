from cvxopt import matrix
from cvxopt.modeling import dot, op, variable

from hw6.problems import *

"""
Rock paper scissors example

     R   P   S
R [[ 0,  1, -1],
P  [-1,  0,  1],
S  [ 1, -1,  0]]


"""

# Variables
R = variable(1, 'R')
P = variable(1, 'P')
S = variable(1, 'S')
V = variable(1, 'V')

# Variable vector [R, P, S, V]
# x = variable(4)

# Objective function
# A = matrix(sample_problems[0])

# c = None
# G = matrix([[1, 0, -1, 1],
#             [1, 1, 0, -1],
#             [1, -1, 1, 0],
#             [0, 1, 1, 1],
#             [0, -1, -1, -1]])

# h = []

# Coefficients for each variable that sums to 1.0
# A = matrix([[1], [1], [1]])

# Not really sure what this means
# b = matrix(1)

# Linear constraints
c1 = (P - S >= V)
c2 = (-R + S >= V)
c3 = (R - P >= V)
c4 = (R + P + S == 1)
constraints = [c1, c2, c3, c4]

problem = op(-V, constraints)
problem.solve()
# print(problem.objective.value())
for var in problem.variables():
    print('{} = {}'.format(var.name, var.value))
