import string

import numpy as np

from project1.models import State
from project1.settings import NSTATES


def reset_states():
    states = [State(string.ascii_uppercase[i], i + 1, v=0.5, r=0.0) for i in range(NSTATES)]
    states.insert(0, State('0', 0, v=0.0, r=0.0, terminal=True))
    states.append(State('1', NSTATES + 1, v=0.0, r=1.0, terminal=True))
    return states


def generate_episodes(nepisodes, states, limit=8):
    episodes = []

    for i in range(nepisodes):
        # Start in middle state C
        # 0 <-- A <--> B <--> C <--> D <--> E --> 1
        # 0     1      2      3      4      5     6
        idx = 3
        state_sequence = [states[idx]]

        # Simulate episode
        while True:
            # Slack experiment: restrict length of episode to control error
            # COOL
            if len(state_sequence) > limit:
                idx = 3
                state_sequence = [states[idx]]
            if state_sequence[len(state_sequence) - 1].terminal:
                break
            idx += 1 if np.random.choice(2) else -1
            state_sequence.append(states[idx])

        episodes.append(state_sequence)

    return episodes


def rmse(est, actual):
    """
    Helper method to calculate RMSE of TD values
    :param est:
    :param actual:
    :return:
    """
    # Create array (sets x states) of all state errors for all training sets
    errors = np.subtract(est, actual)

    # Calculate RMSE across state predictions for each state
    try:
        errors = np.sqrt(np.mean(pow(errors, 2), axis=1))
    except IndexError:
        errors = np.sqrt(np.mean(pow(errors, 2)))

    # Average RMSEs across all training sets to return scalar TD error for each lambda
    return np.mean(errors)
