import numpy as np


def build_transition_matrix(num_states, N, B):
    # Build transition function/matrix
    # transitions for action = roll
    roll_t = np.empty((0, num_states))

    for s in range(num_states):
        state_row = np.zeros(num_states)
        for s_prime in range(1, num_states / 2):
            # s_prime is the new bankroll, which can only be 0 or between the
            # current state s and s + maximum roll, and we'll set probability
            # for s_prime = 0 afterward since it's the same for all states
            if s < s_prime <= s + N < num_states / 2:
                if not B[s_prime - s - 1]:
                    state_row[s_prime] = 1. / N

        roll_t = np.vstack([roll_t, state_row])

    # probability of ending in state 0 is len(B)/N for all states, but
    # we're cutting an infinite horizon and faking the last N -1 states to
    # always end up (p=1.0) in state 0 (bankrupt)
    # roll_t[:, num_states / 2] = np.full((len(states)), 1. * sum(B) / N)
    roll_t[:, num_states / 2] = 1. - np.sum(roll_t, axis=1)

    # transitions for action = quit
    quit_t = np.identity(num_states)  # 100% chance of ending in same/current state

    return np.array([roll_t, quit_t])


# def build_reward_matrix(num_states):
#     # Build reward function/matrix
#     roll_r = np.empty((0, num_states))
#
#     for s in range(num_states):
#         state_row = np.zeros(num_states)
#         for s_prime in range(num_states):
#             # reward for s_prime = 0 is LOSING everything currently in bankroll
#             state_row[s_prime] = s_prime - s
#
#         roll_r = np.vstack([roll_r, state_row])
#
#     # No additional reward for quitting
#     quit_r = np.zeros((num_states, num_states))
#
#     return np.array([roll_r, quit_r])


def build_reward_matrix(num_states):
    """ Office hours: use quitting as only way to get immediate reward """

    # No immediate reward for rolling
    roll_r = np.zeros((num_states, num_states))

    # Quitting gives reward equal to current state/bankroll, but only once
    # (from non-terminal to its respective terminal state)
    quit_r = np.zeros((num_states, num_states))
    for i in range(num_states / 2):
        quit_r[i, num_states / 2 + i] = i

    return np.array([roll_r, quit_r])
