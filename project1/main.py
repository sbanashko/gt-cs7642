import numpy as np
import matplotlib.pyplot as plt

from mdptoolbox.mdp import MDP


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

print 'Transition matrix:', T.shape
print T

print '*' * 80

print 'Reward matrix:', R.shape
print R

gamma = 0.9
epsilon = 1e-6
max_iter = 100

mdp = MDP(T, R, gamma, epsilon, max_iter)

# TODO Replicate figure 3
# error (using best alpha for each lambda) as a function of lambda
# lambda values: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
# averaged over 100 training sets, 10 sequences each

# TODO Replicate figure 4

# TODO Replicate figure 5
