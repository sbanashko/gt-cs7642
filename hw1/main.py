"""
CS 7642 Homework 1
Dan Frakes | dfrakes3
"""
from mdptoolbox import mdp

from util import *
from problems import *


def run_program(N, B):
    # Arbitrary threshold to consider realistic horizon
    threshold = 0.01

    # Iterator variables
    max_timestep = 0

    # Calculate maximum likely horizon using arbitrary threshold
    while True:
        p = pow(1. * (N - sum(B)) / N, max_timestep)
        if p < threshold:
            break
        max_timestep += 1

    # State is bankroll, from 0 to N*max_timesteps (inclusive)
    states = range(N * max_timestep + 1)

    # Extra state for each possible bankroll to indicate terminal state
    states *= 2

    # Actions are always roll or quit, encoded to {0, 1}
    actions = [0, 1]

    T = build_transition_matrix(len(states), N, B)
    R = build_reward_matrix(len(states))

    # Gamma is 1 since we don't value future reward any less than immediate
    gamma = 1.0

    # Arbitrary threshold epsilon
    epsilon = 0.01

    vi = mdp.ValueIteration(T, R, gamma, epsilon, max_iter=1000)
    vi.run()

    print 'N={} ... output={}'.format(N, vi.V[0])


''' Let's do this thing'''
# e = examples[0]
# run_program(e['N'], e['B'])

# for e in examples:
#     run_program(e['N'], e['B'])

for p in problems:
    run_program(p['N'], p['B'])
