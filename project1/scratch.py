import numpy as np
import matplotlib.pyplot as plt

from mdptoolbox.mdp import ValueIteration

# from td import TD


def _build_transition_matrix(states):
    """
    Build matrix of dimensions (A, S, S)
    :param s: states array
    :return:
    """
    num_states = len(states)
    matrix = np.zeros((num_states, num_states))
    matrix[0, 0] = 1
    matrix[num_states - 1, num_states - 1] = 1
    for i in range(1, num_states - 1):
        matrix[i, i - 1] = 0.5
        matrix[i, i + 1] = 0.5
    return np.array([matrix])


def _build_reward_matrix(states):
    matrix = np.zeros((len(states), len(states)))
    matrix[5, 6] = 1  # only reward is from F to G
    return np.array([matrix])


# TODO implement random walk problem as MDP
S = range(7)

T = _build_transition_matrix(S)
R = _build_reward_matrix(S)

gamma = 1
epsilon = 1e-6
max_iter = 100

vi = ValueIteration(T, R, gamma, epsilon, max_iter)

vi.run()

print vi.V

# TODO replicate S&B Exercise 6.2
# Estimated state value updates over 100 iterations, initialized at 0.5
# TD(0) vs TD(1) RMS

# TODO Replicate figure 3
# error (using best alpha for each lambda) as a function of lambda
# lambda values: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# averaged over 100 training sets, 10 sequences each

# TODO Replicate figure 4

# TODO Replicate figure 5
