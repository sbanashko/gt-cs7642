from cvxopt import matrix, solvers
from cvxopt.modeling import dot, op, variable
import numpy as np


solvers.options['show_progress'] = False


def solve(Q_s, objective_fn=None, learner=None):

    # pi_N, pi_E, pi_W, pi_S, pi_X
    na = Q_s.shape[0]

    # Variable vector [pi_N, pi_E, pi_W, pi_S, pi_X, V]
    x = variable(na + 1)

    # if learner == 'foe':
    #     # Minimax
    #     op = min(range(Q_s.shape[1]), key=lambda a2: sum([Q_s[a1, a2] for a1 in range(Q_s.shape[0])]))
    #     player_vals = Q_s.T[op]
    #
    #     # Now use this to create constraints
    #     player_vals
    #
    #     # Maximize the probability distribution pi
    #     obj_matrix = np.zeros(len(x))
    #     obj_matrix[-1] = -1
    #     c = matrix(obj_matrix)
    #     objective_fn = dot(c, x)
    #
    # elif learner == 'uceq':
    #     # Default to minimize cTx == -V (same as maximize V)
    #     obj_matrix = np.zeros(len(x))
    #     obj_matrix[-1] = -1
    #     c = matrix(obj_matrix)
    #     objective_fn = dot(c, x)

    # Default to minimize cTx == -V (same as maximize V)
    obj_matrix = np.zeros(len(x))
    obj_matrix[-1] = -1.
    c = matrix(obj_matrix)
    objective_fn = dot(c, x)

    # Matrices
    ineq_matrix = np.zeros((0, len(x)))

    for row in Q_s:
        new_row = [row[i] for i in range(len(row))]
        new_row.append(1.)
        ineq_matrix = np.vstack([ineq_matrix, new_row])

    # ~~~ Other known constraints ~~~

    # sum_{pi_i} = 1
    pi_vars = np.ones(na)
    pi_vars = np.append(pi_vars, 0.)
    ineq_matrix = np.vstack([ineq_matrix, pi_vars])

    # each pi_i >= 0 (same as -pi_i <= 0)
    for i in range(na):
        new_row = np.zeros(len(x))
        new_row[i] = -1.
        ineq_matrix = np.vstack([ineq_matrix, new_row])

    A = matrix(ineq_matrix)

    # ~~~ Limits to other known constraints ~~~

    # sum_{pi_i} = 1
    limits = np.append(np.zeros(na), 1.)

    # each pi_i >= 0
    limits = np.append(limits, np.zeros(na))

    b = matrix(limits)

    # Linear constraints embedded in matrices
    ineq = (A * x <= b)

    lp = op(objective_fn, ineq)
    lp.solve()

    # [pi_N, pi_E, pi_W, pi_S, pi_X, V] ^ T
    probabilities = np.array(ineq.multiplier.value[:na].T)
    return probabilities / probabilities.sum(0)
