import json

import numpy as np


def construct_TR(ns):
    """
    Reconstruct Figure 2.3 from Littman's thesis (page 42)
    with any (even) number of states > 3
    :param ns: int - number of states
    :return:
    """

    ''' Transition function '''
    T1 = np.zeros((ns, ns))
    T2 = np.zeros((ns, ns))

    # Initial state
    T1[0, 1] = 1.0
    T2[0, 2] = 1.0

    # Decision states
    for s in range(1, ns - 3):
        if s % 2 == 0:
            T1[s, s + 1] = 0.5
            T1[s, s + 2] = 0.5
            T2[s, s + 1] = 0.5
            T2[s, s + 2] = 0.5
        else:
            T1[s, s + 2] = 1.0
            T2[s, s + 3] = 1.0

    # Absorbing states
    for s in range(ns - 3, ns):
        T1[s, ns - 1] = 1.0
        T2[s, ns - 1] = 1.0

    T = np.array([T1, T2])

    ''' Reward function '''
    R1 = np.zeros((ns, ns))
    R2 = np.zeros((ns, ns))

    R1[ns - 3, ns - 1] = -100
    R2[ns - 3, ns - 1] = -100

    R = np.array([R1, R2])

    # TESTING
    # for a in range(2):
    #     for s in np.arange(1, ns - 3, 2):
    #         R[a, s, s + 3] = 0

    # print(T)
    # print('*' * 80)
    # print(R)

    return T, R


def construct_mdp_obj(T, R):
    mdp = {}
    states = []

    ns = T.shape[1]

    for s in range(len(T[0][0])):

        state = {'id': s}
        actions = []

        for a in range(len(T)):

            action = {'id': a}
            transitions = []
            transition_idx = 0  # to_idx not guaranteed if prob == 0

            for to_idx in range(ns):
                probability = round(T[a, s, to_idx], 2)
                if probability > 0:
                    transitions.append({
                        'id': transition_idx,
                        'probability': probability,
                        'reward': R[a, s, to_idx],
                        'to': to_idx
                    })
                    transition_idx += 1

            action['transitions'] = transitions
            actions.append(action)

        state['actions'] = actions
        states.append(state)
    mdp['gamma'] = 0.75
    mdp['states'] = states

    return mdp


def mdp_to_json(mdp_obj, save=False):
    if save:
        with open('problem.json', 'w') as outfile:
            json.dump(mdp_obj, outfile)
    else:
        json_data = json.dumps(mdp_obj)
        return json_data
