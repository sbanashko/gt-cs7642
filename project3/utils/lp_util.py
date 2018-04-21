from cvxopt import matrix
from cvxopt.modeling import dot, op, variable
import numpy as np


def solve(Q_s):
    # Variable vector [pi_N, pi_E, pi_W, pi_S, pi_X, V]
    x = variable(Q_s.shape[0] + 1)

    # minimize cTx == -V
    obj_matrix = np.zeros(len(x))
    obj_matrix[-1] = -1
    c = matrix(obj_matrix)
    objective = dot(c, x)

    # Matrices
    ineq_matrix = np.zeros((0, len(x)))

    # R = 0, P = 1, S = 2, V = V
    for row in Q_s:
        new_row = [row[i] for i in range(len(row))]
        new_row.append(1.)
        ineq_matrix = np.vstack([ineq_matrix, new_row])

    # Other known constraints
    pi_vars = np.ones(len(x))
    np.append(pi_vars, 0.)
    ineq_matrix = np.vstack([ineq_matrix, pi_vars])  # pi <= 1

    A = matrix(ineq_matrix)
    pi_sum = np.zeros(len(x))
    np.append(pi_sum, 1.)
    b = matrix(pi_sum)

    # Linear constraints embedded in matrices
    ineq = (A * x <= b)

    lp = op(objective, ineq)
    lp.solve()

    # [pi_N, pi_E, pi_W, pi_S, pi_X, V] ^ T
    return ineq.multiplier.value
#
#
# sample = np.array([[1, 2, 3, 4, 5],
#                    [4, 3, 5, 2, 3],
#                    [5, 4, 9, 1, 0],
#                    [3, 8, 7, 6, 6],
#                    [9, 1, 0, 3, 4]])
#
# print(solve(sample))
