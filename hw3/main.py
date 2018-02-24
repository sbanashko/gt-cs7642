from problems import get_sample_problem
from util import mdp_to_json

mdp = {}

T, R = get_sample_problem()

na = len(T)
ns = len(T[0])

print ns, 'states and', na, 'actions'

states = []

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

print mdp_to_json(mdp)
