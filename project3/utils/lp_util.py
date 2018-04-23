from cvxopt import matrix, solvers
from cvxopt.base import spmatrix, spdiag
from cvxopt.modeling import dot, op, variable
import numpy as np

solvers.options['show_progress'] = False
solvers.options['msg_lev'] = 'GLP_MSG_OFF'
solvers.options['show_progress'] = False  # disable solver output
solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
solvers.options['LPX_K_MSGLEV'] = 0  # previous versions


# https://github.com/adam5ny/blogs/blob/master/cvxopt/cvxopt_examples.py
def ce(Q, opQ, solver=None):
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


def maxmin(A, solver='glpk'):
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
