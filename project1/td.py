import os
import numpy as np
import matplotlib.pyplot as plt


class State:
    def __init__(self, name, v=0, r=0, e=0, terminal=False):
        self.name = name
        self.v = v
        self.r = r
        self.e = e
        self.terminal = terminal


states = [State('L', v=0.0, r=0.0, terminal=True),  # left terminal state
          State('A', v=0.5, r=0.0),  # A
          State('B', v=0.5, r=0.0),  # B
          State('C', v=0.5, r=0.0),  # C
          State('D', v=0.5, r=0.0),  # D
          State('E', v=0.5, r=0.0),  # E
          State('R', v=0.0, r=1.0, terminal=True)]  # right terminal state

episodes = 100
alpha = 1.0
gamma = 1.0
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


def step_estimate(state_seq, k):
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


def _save_plot(episode):
    xvals = range(1, len(states) - 1)
    plt.plot(xvals, [y / 6.0 for y in range(1, 6)], color='black', label='actual')
    plt.plot(xvals, [s.v for s in states[1:6]], color='blue', label='estimate')
    plt.xticks(xvals, [s.name for s in states[1:6]])
    plt.ylim(0.0, 1.0)
    plt.text(4.3, 0.05, r'T = {}'.format(episode), fontsize=15)
    plt.legend()
    plt.savefig(os.path.join('output', '{}.png'.format(str(episode).zfill(4))))
    plt.close()


def _save_error_plot(episode):
    xvals = range(1, len(states) - 1)
    plt.plot(xvals, [y / 6.0 for y in range(1, 6)], color='black', label='actual')
    plt.plot(xvals, [s.v for s in states[1:6]], color='blue', label='estimate')
    plt.xticks(xvals, [s.name for s in states[1:6]])
    plt.ylim(0.0, 1.0)
    plt.text(4.3, 0.05, r'T = {}'.format(episode), fontsize=15)
    plt.legend()
    plt.savefig(os.path.join('output', '{}.png'.format(str(episode).zfill(4))))
    plt.close()


def TD(lambda_val, debug=False):

    print 'TD({})'.format(lambda_val)

    # Save initial plot
    _save_plot(0)

    # Store errors for one plot at end of all episodes
    td_error = []

    for T in range(episodes):

        # Update learning rate according to episode
        alpha = 1. / (T + 1)
        episode_error = 0
        idx = 3
        state_sequence = [states[idx]]

        # Simulate episode
        while True:
            if state_sequence[len(state_sequence) - 1].terminal:
                break
            idx += 1 if np.random.choice(2) else -1
            state_sequence.append(states[idx])

        # DEBUG
        if debug:
            print 'Episode', T + 1, ':', [s.name for s in state_sequence]

        # Execute iterative value updates
        for s in state_sequence:
            s.e = 0

        for t in range(1, len(state_sequence)):
            state_sequence[t - 1].e += 1

            for s2 in state_sequence:
                state_error = state_sequence[t].r + gamma * state_sequence[t].v - state_sequence[t - 1].v
                delta = alpha * s2.e * state_error
                s2.v += delta
                s2.e *= lambda_val * gamma
                episode_error += np.sqrt(pow(state_error, 2))

        # _save_plot(T + 1)

        td_error.append(episode_error)

    # Plot error over episodes
    plt.plot(range(episodes), td_error)
    plt.savefig(os.path.join('output', 'error.png'))
    plt.close()

    return sum([step_weight(lambda_val, k) * step_estimate(state_sequence, k) for k in range(1, max_lookahead + 2)]), \
           np.mean(td_error)


results = [TD(ld) for ld in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]]

# print [results[i][0] for i in range(len(results))]

plt.plot([results[i][0] for i in range(len(results))],
         [results[i][1] for i in range(len(results))])
plt.show()
