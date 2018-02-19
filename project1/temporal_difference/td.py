import string

import numpy as np

from project1.models.state import State
from project1.settings import *
from project1.utils import plot_val_estimates

max_lookahead = 100  # maximum_lookahead
tol = 1e-3


def _reset_states():
    states = [State(string.ascii_uppercase[i], i + 1, v=0.5, r=0.0) for i in range(NSTATES)]
    states.insert(0, State('0', 0, v=0.0, r=0.0, terminal=True))
    states.append(State('1', NSTATES + 1, v=0.0, r=1.0, terminal=True))
    return states


def _reset_delta_v(nstates):
    return np.zeros(nstates)


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


def TD(lambda_val,
       episodes,
       alpha=0.3,
       alpha_decay_rate=0.9,
       gamma=1.0,
       max_iter=MAX_ITERATIONS,
       epsilon=0.001,
       history=False):
    """
    Temporal difference learner
    :param lambda_val:
    :param episodes:
    :param alpha:
    :param alpha_decay_rate:
    :param gamma:
    :param max_iter:
    :param epsilon:
    :param history: if True, return 2D array of state values after each
    episode, otherwise return 1D array of final state values
    :return:
    """
    # Store history state values after each episode
    V = np.ndarray((0, NSTATES))  # don't care about terminal states

    # Shit was carrying over from one TD calculation the next!
    states = _reset_states()

    # Flag convergence
    converged = False
    iterator = 0

    # Record state value estimates
    sv_estimates = []

    delta_v = _reset_delta_v(NSTATES)

    # Repeatedly present same episodes in training set until convergence
    while not converged:
        if WEIGHT_UPDATE_LOC == 'trainset':
            delta_v = _reset_delta_v(NSTATES)

        for T, sequence in enumerate(episodes):

            if SAVE_EX62 and T in EX62_T_VALS and len(sv_estimates) == 0:
                sv_estimates.append([states[i].v for i in range(1, len(states) - 1)])

            # Execute iterative value updates
            for s in states:
                s.e = 0

            # Reset and store deltas of value/weight vector
            if WEIGHT_UPDATE_LOC == 'episode':
                delta_v = _reset_delta_v(NSTATES)

            for t in range(1, len(sequence)):
                current_state = states[sequence[t].index]
                prev_state = states[sequence[t - 1].index]

                prev_state.e += 1

                if WEIGHT_UPDATE_LOC == 'timestep':
                    delta_v = _reset_delta_v(NSTATES)

                state_error = current_state.r + gamma * current_state.v - prev_state.v

                # Apply update to all states wrt eligibility
                for s in states:
                    delta = alpha * s.e * state_error

                    # Apply weight update after each TIMESTEP
                    if WEIGHT_UPDATE_LOC == 'timestep':
                        s.v += delta
                    elif not s.terminal:
                        delta_v[s.index - 1] += delta
                    s.e *= lambda_val * gamma

            # Apply weight update after each EPISODE
            if WEIGHT_UPDATE_LOC == 'episode':
                for i in range(len(delta_v)):
                    states[i + 1].v += delta_v[i]

            if SAVE_EX62 and T in EX62_T_VALS and len(sv_estimates) < len(EX62_T_VALS):
                sv_estimates.append([states[i].v for i in range(1, len(states) - 1)])

            # Update learning rate according to episode
            # alpha = 1. / (T + 1)
            alpha *= alpha_decay_rate

        iterator += 1

        # Apply weight update after each presentation of TRAINING SET
        if WEIGHT_UPDATE_LOC == 'trainset':
            for i in range(len(delta_v)):
                states[i + 1].v += delta_v[i]

        V = np.vstack([V, [states[i].v for i in range(1, len(states) - 1)]])  # don't care about terminal states

        # plot_val_estimates(T, sv_estimates, T, alpha)

        if iterator >= max_iter:
            if DEBUG:
                print 'Max iterations reached'
            break

        # Check Euclidean distance of gradient descent for convergence
        dv = np.sqrt(np.sum([pow(dv, 2) for dv in delta_v]))
        # dv = rmse([s.v for s in states[1:len(states)-1]], ACTUAL_STATE_VALUES)

        if dv < epsilon:
            converged = True

    if SAVE_EX62:
        plot_val_estimates(0, sv_estimates)
    # print V[len(V) - 1]
    return V if history else V[len(V) - 1]
