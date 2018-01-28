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
    timestep = 0
    max_timesteps = 0

    # Calculate maximum likely horizon using arbitrary threshold
    while True:
        p = pow(1. * (N - sum(B)) / N, timestep)
        if p < threshold:
            max_timesteps = timestep - 1
            break
        timestep += 1

    # print 'Threshold =', threshold, ': Max Timesteps =', max_timesteps

    # State is bankroll, from 0 to N*max_timesteps (inclusive)
    states = range(N * max_timesteps + 1)

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
    vi.setVerbose()
    vi.run()

    # last_roll = 0
    # i = 0
    # for r in vi.policy:
    #     if r == 1:
    #         last_roll = i
    #         break
    #     i += 1
    # print 'Optimal policy: roll until bankroll = {}'.format(last_roll)
    # print 'Expected value of first state:', vi.V[0]

    print 'N={} ... output={}'.format(N, vi.R[0][0])

    # print 'State values:'
    # for i in range(len(vi.V)):
    #     print 'state {} : {}'.format(i, vi.V[i])


''' Let's do this thing'''
e = dummy[0]
run_program(e['N'], e['B'])

# for e in examples:
#     run_program(e['N'], e['B'])

# for p in problems:
#     run_program(p['N'], p['B'])
