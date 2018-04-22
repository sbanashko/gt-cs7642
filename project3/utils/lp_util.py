from cvxopt import matrix, solvers
from cvxopt.base import spmatrix, spdiag
from cvxopt.modeling import dot, op, variable
import numpy as np

solvers.options['show_progress'] = False
solvers.options['msg_lev'] = 'GLP_MSG_OFF'


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
    lp.solve(solver='glpk', options={'glpk':{'msg_lev':'GLP_MSG_OFF'}})

    # [pi_N, pi_E, pi_W, pi_S, pi_X, V] ^ T
    probabilities = np.array(ineq.multiplier.value[:na].T)
    return probabilities / probabilities.sum(0)


def _objective_fn(A, b):
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


def solve_v2(Q_s):
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

    objective_fn = _objective_fn(Q_s)

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
