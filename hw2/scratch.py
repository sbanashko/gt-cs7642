import numpy as np

# Inputs
probToState = 0.5
valueEstimates = [0, 3, 8, 2, 1, 2, 0]
rewards = [0, 0, 0, 4, 1, 1, 1]
gamma = 1


class State:
    def __init__(self, v=0):
        self.v = v


states = [State(valueEstimates[s]) for s in range(7)]


def execute_episode(ep):
    print 'episode', ep
    # Update learning rate
    alpha = 1. / (ep + 1)

    # Simulate sequence based on probToState
    idx_sequence = [0, 1, 3, 4, 5, 6]
    if np.random.choice(2, p=[probToState, 1 - probToState]):
        idx_sequence[1] = 2

    state_sequence = [states[i] for i in idx_sequence]

    # For each state, update state value with k-estimator. For this problem,
    # we only need k up to 5 because all state sequences have 6 unique states
    # (max look-ahead of 5 states)
    for t, state in enumerate(state_sequence):
        # print '  timestep', t
        for k_max in range(5 - t):
            Ek = sum([pow(gamma, k - 1) * rewards[t + k] for k in range(k_max)]) + \
                 pow(gamma, k_max) * state_sequence[t + k_max].v - state.v

            # print '    Ek', Ek

            state.v += alpha * Ek

    print '*{}*'.format([s.v for s in states])


for ep in range(10):
    execute_episode(ep)


# def _generate_episode_sequence():
#     """
#     Specific to HW2 provided MDP, DON'T REUSE THIS CODE!
#     :return:
#     """
#     stochastic_state = np.random.choice(2, p=[probToState, 1 - probToState]) + 1
#     state_indexes = [0, stochastic_state, 3, 4, 5, 6]
#     state_sequence = [states[i] for i in state_indexes]
#
#     stochastic_reward = 0 if stochastic_state == 1 else 1
#     reward_indexes = [stochastic_reward, stochastic_reward + 2, 4, 5, 6]
#     reward_sequence = [rewards[i] for i in reward_indexes]
#     reward_sequence.append(0)
#
#     return state_sequence, reward_sequence

# def TD(lambda_val, seq):
#     return sum([step_weight(lambda_val, k) * step_estimate(seq, k) for k in range(max_step)]) + step_estimate(max_step)

# def iterative_TD(lambda_val):
#     for T in range(episodes):
#
#         # print 'Episode', T
#         # for s in states:
#         #     print '  State {}: {}'.format(s.index, s.v)
#
#         # 1-base T to simplify translation from algorithms to Python
#         T += 1
#
#         # Generate state sequence
#         state_seq, reward_seq = _generate_episode_sequence()
#
#         for state in state_seq:
#             state.e = 0
#
#         # Execute timestep
#         for t in timesteps:
#             prev_state = state_seq[t - 1]
#             prev_state.e += 1
#
#             for state in state_seq:
#                 # state.v += alpha * state.e * (reward_seq[t - 1] + gamma * state.v - prev_state.v)
#                 print 'DPF TESTING'
#                 total_w = 0
#                 for k in range(1, max_lookahead + 1):
#                     w = step_weight(lambda_val, k)
#                     print 'Weight for {}-step estimate:'.format(k), w
#                     total_w += w
#                 print total_w
#                 print 'but my sum here was', sum([step_weight(lambda_val, a) for a in range(1, max_lookahead + 1)])
#                 print 'END TESTING'
#                 exit(4)
#                 state.v += alpha * state.e * sum(
#                     [step_weight(lambda_val, k) * step_estimate(state_seq, reward_seq, k) for k in
#                      range(max_lookahead)])
#                 state.e *= gamma * lambda_val
#
#     return states