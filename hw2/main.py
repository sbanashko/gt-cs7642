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

from scipy import optimize


class State:
    def __init__(self, v=0, e=0):
        self.v = v
        self.e = e


probToState = 0.5
valueEstimates = [0, 3, 8, 2, 1, 2, 0]
states = [State(v) for v in valueEstimates]
rewards = [0, 0, 0, 4, 1, 1, 1]

max_lookahead = 5  # maximum_lookahead
timesteps = range(1, 7)
alpha = 1
gamma = 1
episodes = 1000


def step_weight(lambda_val, k):
    """
    The weight applied to each k-step estimate to calculate total change to state value
    :param lambda_val:
    :param k:
    :return:
    """
    # Just fill out remaining weight to equal 1 to avoid geometric series calculations...
    if k >= max_lookahead:
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
    k = min(k, max_lookahead)
    return sum([reward_seq[i] for i in range(k)]) + state_seq[k].v - state_seq[0].v


def _generate_model_sequence():
    """
    Specific to HW2 provided MDP, DON'T REUSE THIS CODE!
    :return:
    """
    state_sequence = [states[0],
                      State(v=probToState * states[1].v + (1 - probToState) * states[2].v),
                      states[3],
                      states[4],
                      states[5],
                      states[6]]

    reward_sequence = [probToState * rewards[0] + (1 - probToState) * rewards[1],
                       probToState * rewards[2] + (1 - probToState) * rewards[3],
                       rewards[4],
                       rewards[5],
                       rewards[6]]

    return state_sequence, reward_sequence


def TD(lambda_val):
    state_seq, reward_seq = _generate_model_sequence()
    return sum([step_weight(lambda_val, k) * step_estimate(state_seq, reward_seq, k) for k in range(1, max_lookahead + 1)])


def fn(lambda_val):
    return TD(lambda_val) - TD(1)


# Example from Piazza @126
for ld in [0, 0.4, 1]:
    print 'TD({}) = {}'.format(ld, TD(ld))
    print '=' * 80

# print optimize.newton(fn, 0.0, maxiter=1000)
# print optimize.brentq(fn, 0.1, 0.9)
