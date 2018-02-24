import numpy as np


def get_sample_problem():
    # hw3_tester example as mdptoolbox setup
    T = np.array([[
        [0, 0.5, 0.5],
        [0, 1, 0],
        [0.9, 0, 0.1]
    ], [
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]])

    R = np.array([[
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]
    ], [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]])
    return T, R
