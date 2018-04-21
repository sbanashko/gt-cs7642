from cvxopt import matrix
from cvxopt.modeling import dot, op, variable
import numpy as np


def solve(problem):
    # Variable vector [R, P, S, V]
    x = variable(4)

    # minimize cTx == -V
    c = matrix([0., 0., 0., -1.])
    objective = dot(c, x)

    # Matrices
    Ainit = np.zeros((0, len(x)))

    # R = 0, P = 1, S = 2, V = V
    for row in problem:
        Ainit = np.vstack([Ainit, [row[0], row[1], row[2], 1]])

    # Other known constraints
    Ainit = np.vstack([Ainit, [1., 1., 1., 0.]])  # R + P + S <= 1

    A = matrix(Ainit)
    b = matrix([0., 0., 0., 1.])

    # Linear constraints embedded in matrices
    ineq = (A * x <= b)

    lp = op(objective, ineq)
    lp.solve()

    # [R, P, S, V] ^ T
    return ineq.multiplier.value
