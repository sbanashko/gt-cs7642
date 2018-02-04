"""
CS 7642 Homework 2
Dan Frakes | dfrakes3

Given the following MDP:

   -> s1
s0       -> s3 -> s4 -> s5 -> s6
   -> s2

Calculate a value of lambda (such that lambda < 1) for which TD(lambda) = TD(1).

Given:
probToState - probability of transition from s0 to s1
valueEstimates - vector of initial value estimates for each state
rewards - vector of rewards for each transition
"""

import numpy as np
from scipy import optimize, isclose

from problems import problems


class State:
    def __init__(self, v=0, e=0):
        self.v = v
        self.e = e


max_lookahead = 100  # maximum_lookahead
tol = 1e-3


def step_weight(lambda_val, k):
    """
    The weight applied to each k-step estimate to calculate total change to state value
    :param lambda_val:
    :param k:
    :return:
    """
    # Just fill out remaining weight to equal 1 to avoid geometric series calculations...
    if k > max_lookahead:
        existing_weights = sum([step_weight(lambda_val, i) for i in range(1, max_lookahead)])
        return 1 - existing_weights
    return (1 - lambda_val) * pow(lambda_val, k - 1)


def step_estimate(state_seq, reward_seq, k):
    """
    The step estimate, Ek, is the calculated change in state value for a k-step lookahead
    :param state_seq:
    :param reward_seq:
    :param k:
    :return:
    """
    # Use final state's estimate if k is beyond state sequence length
    if k >= len(state_seq):
        k = len(state_seq) - 1
    return sum([reward_seq[i] for i in range(k)]) + state_seq[k].v - state_seq[0].v


def _generate_model_sequence(problem):
    """
    Specific to HW2 provided MDP, DON'T REUSE THIS CODE!
    :return:
    """
    probToState = problem.probToState
    rewards = problem.rewards
    states = [State(v) for v in problem.valueEstimates]

    state_sequence = [states[0],
                      State(v=probToState * states[1].v + (1 - probToState) * states[2].v),
                      states[3],
                      states[4],
                      states[5],
                      states[6],
                      State(0)]

    reward_sequence = [probToState * rewards[0] + (1 - probToState) * rewards[1],
                       probToState * rewards[2] + (1 - probToState) * rewards[3],
                       rewards[4],
                       rewards[5],
                       rewards[6],
                       0]

    return state_sequence, reward_sequence


def TD(lambda_val, problem, debug=False):

    state_seq, reward_seq = _generate_model_sequence(problem)

    # debugging
    if debug:
        for i in range(1, max_lookahead + 2):
            print '    E_{}={} * weight={}'.format(i,
                                                   step_estimate(state_seq, reward_seq, i),
                                                   step_weight(lambda_val, i))

        print '  weight sum:', sum([step_weight(lambda_val, j) for j in range(1, max_lookahead + 2)])
        print

    return sum([step_weight(lambda_val, k) * step_estimate(state_seq, reward_seq, k) for k in range(1, max_lookahead + 2)])


def fn(lambda_val, problem):
    return TD(lambda_val, problem) - TD(1, problem)


for idx, p in enumerate(problems):
    solved = False
    for guess in np.linspace(0.0, 0.9, 100):
        try:
            solution = optimize.newton(fn, guess, args=(p, ))
            if 0 <= solution < 0.99 and isclose(TD(1, p), TD(solution, p), tol):
                solved = True
                label = 'Unknown'
                if p.test:
                    label = 'Correct' if isclose(solution, p.solution, tol) else 'Failed'
                print 'Problem {}: {} ({})'.format(idx + 1, solution, label)
                break
        except RuntimeError:
            # Failed to converge after n iterations
            pass
    if not solved:
        print 'Problem {}: Failed'.format(idx + 1)
