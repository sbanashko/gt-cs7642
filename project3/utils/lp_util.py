from cvxopt import matrix, solvers
from cvxopt.base import spmatrix, spdiag
from cvxopt.modeling import dot, op, variable
import numpy as np

solvers.options['show_progress'] = False
solvers.options['msg_lev'] = 'GLP_MSG_OFF'


# def solve_ce(Q_s):
#     # Num actions
#     na = Q_s.shape[0]
#
#     # Variable joint vector [pi_N*, pi_E*, pi_W*, pi_S*, pi_X*]
#     x = variable(pow(na, 2) + 1)
#
#     # Default to minimize cTx == -V (same as maximize V)
#     obj_matrix = np.zeros(len(x))
#     obj_matrix[-1] = -1.
#     c = matrix(obj_matrix)
#     objective_fn = dot(c, x)
#
#     # Inequalities
#     ineq_matrix = np.zeros((0, len(x)))
#
#     for a1 in Q_s:
#         for a1p in Q_s:
#             # No sense in checking inequality of the same (equal) row
#             if a1 != a1p:
#                 constraint = np.zeros(len(x))
#                 # TODO
#                 new_row = [row[i] for i in range(len(row))]
#                 new_row.append(1.)
#                 ineq_matrix = np.vstack([ineq_matrix, new_row])
#
#     # ~~~ Other known constraints ~~~
#
#     # sum_{pi_i} = 1
#     pi_vars = np.ones(na)
#     pi_vars = np.append(pi_vars, 0.)
#     ineq_matrix = np.vstack([ineq_matrix, pi_vars])
#
#     # each pi_i >= 0 (same as -pi_i <= 0)
#     for i in range(na):
#         new_row = np.zeros(len(x))
#         new_row[i] = -1.
#         ineq_matrix = np.vstack([ineq_matrix, new_row])
#
#     A = matrix(ineq_matrix)
#
#     # ~~~ Limits to other known constraints ~~~
#
#     # sum_{pi_i} = 1
#     limits = np.append(np.zeros(na), 1.)
#
#     # each pi_i >= 0
#     limits = np.append(limits, np.zeros(na))
#
#     b = matrix(limits)
#
#     # Linear constraints embedded in matrices
#     ineq = (A * x <= b)
#
#     lp = op(objective_fn, ineq)
#     lp.solve(solver='glpk', options={'glpk': {'msg_lev': 'GLP_MSG_OFF'}})
#
#     # [pi_N, pi_E, pi_W, pi_S, pi_X, V] ^ T
#     probabilities = np.array(ineq.multiplier.value[:na].T)
#     probabilities += + 0.
#     return probabilities / probabilities.sum(0)

def _foe_objective_fn(Q_s):
    # pi_N, pi_E, pi_W, pi_S, pi_X
    na = Q_s.shape[0]

    # Variable vector [pi_N, pi_E, pi_W, pi_S, pi_X, V]
    x = variable(na)

    # Maximize the probability distribution pi
    obj_matrix = np.zeros(len(Q_s))
    obj_matrix[-1] = -1
    c = matrix(obj_matrix)
    return dot(c, x)


def _uceq_objective_fn(A, b):
    m, n = A.size

    def F(x=None, z=None):
        if x is None:
            return 0, matrix(0.5, (n, 1))
        if min(x) <= 0.0:
            return None
        f = sum(x)
        Df = -(x ** -1).T
        if z is None:
            return f, Df
        H = spdiag(z[0] * x ** -2)
        return f, Df, H

    return solvers.cp(F, A=A, b=b)['x']


def solve(Q_s, algo_name):
    # Num actions
    na = Q_s.shape[0]

    # Probability distribution over Q_s (==pow(na, num_agents))
    sigma = variable(na ** 2)
    ineq_matrix = np.zeros((0, len(sigma)))

    # # TODO this is totally not right for CEQ
    # obj_matrix = np.zeros(len(sigma))
    # obj_matrix[-1] = -1.
    # c = matrix(obj_matrix)
    # objective_fn = dot(c, sigma)

    if algo_name == 'foe':
        objective_fn = _foe_objective_fn(Q_s)
    elif algo_name == 'uceq':
        objective_fn = _uceq_objective_fn(Q_s)
    else:
        print('Unsupported LP learner type')
        return

    # Player inequalities
    for ai in range(na):

        # Loop over "other" player actions
        for ai_prime in range(na):

            # Inequality of identical (equal) expressions tells us nothing
            if ai == ai_prime:
                pass

            # Create empty new row to be edited and added to inequality matrix
            inequality = np.zeros(len(sigma))

            # Loop over opponent actions
            for oi in range(na):
                # Add inequality to inequality matrix (A)
                inequality[ai * na + oi] = Q_s[ai, oi] - Q_s[ai_prime, oi]

            ineq_matrix = np.vstack([ineq_matrix, inequality])

    # Opponent inequalities
    for ai in range(na):

        # Loop over "other" opponent actions
        for ai_prime in range(na):

            # Inequality of identical (equal) expressions tells us nothing
            if ai == ai_prime:
                pass

            # Create empty new row to be edited and added to inequality matrix
            inequality = np.zeros(len(sigma))

            # Loop over player actions
            for oi in range(na):
                # Add inequality to inequality matrix (A)
                inequality[ai * na + oi] = Q_s.T[ai, oi] - Q_s.T[ai_prime, oi]

            ineq_matrix = np.vstack([ineq_matrix, inequality])

    # Set limits matrix/array (b)
    limits = np.zeros(len(ineq_matrix))

    ''' Set sum of probabilities to 1 '''
    ineq_matrix = np.vstack([ineq_matrix, np.ones(len(sigma))])
    limits = np.append(limits, 1.)

    ''' Set each probability >= 0 '''
    # identity_matrix = spmatrix(-1.0, range(len(sigma)), range(len(sigma)))
    identity_matrix = np.identity(len(sigma))
    ineq_matrix = np.vstack([ineq_matrix, identity_matrix])
    limits = np.append(limits, np.zeros(len(sigma)))

    # Set it up finally
    A = matrix(ineq_matrix)
    b = matrix(limits)

    ineq = (A * sigma >= b)

    lp = op(objective_fn, ineq)
    lp.solve()

    # pi ^ T
    probabilities = np.array(ineq.multiplier.value[:na].T)
    return probabilities / probabilities.sum(0)


# https://github.com/adam5ny/blogs/blob/master/cvxopt/cvxopt_examples.py
class LinProg(object):
    """
    LP solution for multiple agent zero-sum game with payoffs for PA as A
    """

    def __init__(self, verbose=False):
        if not verbose:
            solvers.options['show_progress'] = False  # disable solver output
            solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
            solvers.options['LPX_K_MSGLEV'] = 0  # previous versions

    def maxmin(self, A, solver='glpk'):
        na = A.shape[0]

        # Minimize matrix c
        c = np.append(-1, np.zeros(na))
        c = matrix(c)

        # Inequality constraints G*x <= h
        G = np.matrix(A).T
        G *= -1  # minimization constraint
        G = np.vstack([G, np.eye(na) * -1])  # > 0 constraint for all vars
        new_col = np.append(np.ones(na), np.zeros(na))
        G = np.insert(G, 0, new_col, axis=1)  # insert utility column
        G = matrix(G)
        h = matrix(np.zeros(na * 2, dtype='float'))

        # Equality constraints Ax = b
        A = np.append(0, np.ones(na))
        A = np.matrix(A, dtype='float')
        A = matrix(A)
        b = np.matrix(1, dtype='float')
        b = matrix(b)

        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)
        probs = np.array(sol['x'].H[1:].T)[0]

        # Scale and normalize to prevent negative probabilities
        probs -= probs.min() + 0.
        return probs / probs.sum(0)

    def ce(self, Q, opQ, solver=None):
        na = Q.shape[0]
        nvars = na ** 2

        Q_flat = Q.flatten()
        opQ_flat = opQ.flatten()

        # Minimize matrix c (*=-1 to maximize)
        c = -np.array(Q_flat + opQ_flat, dtype="float")
        c = matrix(c)

        # Inequality constraints G*x <= h
        G = np.empty((0, nvars))

        # Player constraints
        for i in range(na):  # action row i
            for j in range(na):  # action row j
                if i == j: continue
                constraint = np.zeros(nvars)
                base_idx = i * na
                comp_idx = j * na
                for _ in range(na):
                    constraint[base_idx + _] = Q_flat[comp_idx + _] - Q_flat[base_idx + _]
                G = np.vstack([G, constraint])

        # Opponent constraints
        Gopp = np.empty((0, nvars))
        for i in range(na):  # action row i
            for j in range(na):  # action row j
                if i == j: continue
                constraint = np.zeros(nvars)
                for _ in range(na):
                    # constraint[base_idx + j * _] = opQ_flat[comp_idx + _] - opQ_flat[base_idx + _]
                    constraint[i + _ * na] = opQ_flat[j + (_ * na)] - opQ_flat[i + (_ * na)]
                Gopp = np.vstack([Gopp, constraint])

        G = np.vstack([G, Gopp])
        G = np.matrix(G, dtype="float")
        G = np.vstack([G, -1. * np.eye(nvars)])
        h_size = len(G)
        G = matrix(G)
        h = np.array(np.zeros(h_size), dtype="float")
        h = matrix(h)

        # Equality constraints Ax = b
        A = np.matrix(np.ones(nvars), dtype="float")
        A = matrix(A)
        b = np.matrix(1, dtype="float")
        b = matrix(b)
        sol = solvers.lp(c=c, G=G, h=h, A=A, b=b, solver=solver)

        probs = np.array(sol['x'].T)[0]

        # Scale and normalize to prevent negative probabilities
        probs -= probs.min() + 0.
        return probs.reshape((na, na)) / probs.sum(0)
