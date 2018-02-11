import numpy as np

from project1.models import states
from project1.settings import DEBUG

max_lookahead = 100  # maximum_lookahead
tol = 1e-3


def _step_weight(lambda_val, k):
    """
    The weight applied to each k-step estimate to calculate total change to state value
    :param lambda_val:
    :param k:
    :return:
    """
    # Just fill out remaining weight to equal 1 to avoid geometric series calculations...
    if k > max_lookahead:
        existing_weights = sum([_step_weight(lambda_val, i) for i in range(1, max_lookahead)])
        return 1 - existing_weights
    return (1 - lambda_val) * pow(lambda_val, k - 1)


def _step_estimate(state_seq, k):
    """
    The step estimate, Ek, is the calculated change in state value for a k-step lookahead
    :param state_seq:
    :param k:
    :return:
    """
    # Use final state's estimate if k is beyond state sequence length
    if k >= len(state_seq):
        k = len(state_seq) - 1
    return sum([state_seq[i].r for i in range(k)]) + state_seq[k].v - state_seq[0].v


def TD(lambda_val, alpha_decay_rate=1.0, gamma=1.0, episodes=10, history=True):
    """
    Temporal difference learner
    :param lambda_val:
    :param alpha_decay_rate:
    :param gamma:
    :param episodes:
    :param history: if True, return 2D array of state values after each
    episode, otherwise return 1D array of final state values
    :return:
    """
    if DEBUG:
        print 'TD({})'.format(lambda_val)

    # Store history state values after each episode
    V = np.ndarray((episodes, len(states) - 2))  # don't care about terminal states

    for T in range(episodes):

        # Update learning rate according to episode
        alpha = 1. / pow(T + 1, alpha_decay_rate)
        idx = 3
        state_sequence = [states[idx]]

        # Simulate episode
        while True:
            if state_sequence[len(state_sequence) - 1].terminal:
                break
            idx += 1 if np.random.choice(2) else -1
            state_sequence.append(states[idx])

        if DEBUG:
            print 'Episode', T + 1, ':', [s.name for s in state_sequence]

        # Execute iterative value updates
        for s in state_sequence:
            s.e = 0

        for t in range(1, len(state_sequence)):
            state_sequence[t - 1].e += 1

            for s2 in state_sequence:
                state_error = state_sequence[t].r + gamma * state_sequence[t].v - state_sequence[t - 1].v
                delta = alpha * s2.e * state_error
                s2.v += delta
                s2.e *= lambda_val * gamma

        V[T] = [states[i].v for i in range(1, len(states) - 1)]  # don't care about terminal states

    return V if history else V[len(V) - 1]
