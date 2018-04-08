from cvxopt import matrix, solvers
from cvxopt.modeling import dot, op, variable
import numpy as np

from hw6.problems import *

"""
Rock paper scissors example

     R   P   S
R [[ 0,  1, -1],
P  [-1,  0,  1],
S  [ 1, -1,  0]]


"""
def solve(problem):
    # Variables
    R = variable(1, 'R')
    P = variable(1, 'P')
    S = variable(1, 'S')
    V = variable(1, 'V')

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
    # Ainit = np.vstack([Ainit, [-1., 0., 0., 0.]])  # -R <= 0
    # Ainit = np.vstack([Ainit, [0., -1., 0., 0.]])  # -P <= 0
    # Ainit = np.vstack([Ainit, [0., 0., -1., 0.]])  # -S <= 0

    A = matrix(Ainit)
    # b = matrix([0., 0., 0., 1., 0., 0., 0.])
    b = matrix([0., 0., 0., 1.])

    # Linear constraints embedded in matrices
    ineq = (A*x <= b)

    lp = op(objective, ineq)
    lp.solve()

    # [R, P, S, V] ^ T
    return ineq.multiplier.value


if __name__ == '__main__':
    solvers.options['show_progress'] = False

    for i, problem in enumerate(rldm_problems):
        print('Problem {}'.format(i + 1))
        solution = solve(problem)
        print([round(x, 5) for x in solution])
        # print('R={} P={} S={} V={}'.format(solution[0], solution[1], solution[2], solution[3]))

        if i + 1 < len(rldm_problems):
            print()
