import os
from datetime import datetime as dt
import string
import numpy as np
import matplotlib.pyplot as plt


class State:
    def __init__(self, name, index, v=0.0, r=0.0, e=0.0, terminal=False):
        self.name = name
        self.index = index
        self.v = v
        self.r = r
        self.e = e
        self.terminal = terminal


# configs
RUN_ID = dt.now().strftime('%H%M%S')
os.mkdir(RUN_ID)
NSTATES = 5
ACTUAL_VALUES = [1. * (x + 1) / (NSTATES + 1) for x in range(NSTATES)]
states = [State(string.ascii_uppercase[i], i + 1, v=0.5, r=0.0) for i in range(NSTATES)]
states.insert(0, State('0', 0, v=0.0, r=0.0, terminal=True))
states.append(State('1', NSTATES + 1, v=0.0, r=1.0, terminal=True))


def _reset_delta_v(nstates):
    return [0.0 for _ in range(nstates)]


def _generate_episodes(nepisodes):
    episodes = []

    for i in range(nepisodes):
        # Start in middle state C
        # 0 <-- A <--> B <--> C <--> D <--> E --> 1
        # 0     1      2      3      4      5     6
        idx = 3
        state_sequence = [states[idx]]

        # Simulate episode
        while True:
            if state_sequence[len(state_sequence) - 1].terminal:
                break
            idx += 1 if np.random.choice(2) else -1
            state_sequence.append(states[idx])

        episodes.append(state_sequence)

    return episodes


def TD(lambda_val,
       alpha=0.1,
       alpha_decay_rate=0.999,
       gamma=1.0,
       num_episodes=100):
    """
    Temporal difference learner
    :param lambda_val:
    :param alpha:
    :param alpha_decay_rate:
    :param gamma:
    :param num_episodes:
    :param history: if True, return 2D array of state values after each
    episode, otherwise return 1D array of final state values
    :return:
    """
    # Store episodes to repeatedly present
    episodes = _generate_episodes(num_episodes)

    # Counter for ordered plot filenames
    file_counter = 0

    # Store history state values after each episode
    V = np.empty((0, NSTATES))  # don't care about terminal states

    # Store specific value to match example 6.2
    example_episodes = [0, 1, 10, 100]
    example_ep_vals = []

    # Initial plot
    plot_val_estimates(0, alpha)
    example_ep_vals.append([s.v for s in states][1:NSTATES + 1])

    for T, sequence in enumerate(episodes):

        if (T + 1) % 10 == 0:
            print 'Running episode {}...'.format(T + 1)

        # Execute iterative value updates
        for s in sequence:
            s.e = 0

        # Store value changes to apply after complete episode
        delta_v = np.zeros(len(states) - 2) # don't care about nonterminal states

        for t in range(1, len(sequence)):
            sequence[t - 1].e += 1

            for s in sequence:
                state_error = sequence[t].r + gamma * sequence[t].v - sequence[t - 1].v
                delta = alpha * s.e * state_error
                # s.v += delta
                if not s.terminal:
                    delta_v[s.index - 1] += delta
                s.e *= lambda_val * gamma

        # Update values after episode
        for i, dv in enumerate(delta_v):
            states[i + 1].v += dv

        V = np.vstack([V, [states[i].v for i in range(1, len(states) - 1)]])  # don't care about terminal states

        plot_val_estimates(T + 1, alpha)
        if (T + 1) in example_episodes:
            example_ep_vals.append([s.v for s in states][1:NSTATES + 1])

        # Update learning rate according to episode
        alpha *= alpha_decay_rate

    # Plot combined chart
    fig = plt.figure()
    x = range(1, NSTATES + 1)
    plt.plot(x, ACTUAL_VALUES, label='Actual')
    for e in range(len(example_episodes)):
        plt.plot(x, example_ep_vals[e], label=example_episodes[e])
    plt.xticks(x, ['A', 'B', 'C', 'D', 'E'])
    plt.title('State value estimates')
    plt.xlim((0.5, NSTATES + 0.25))
    plt.ylim((0, 0.85))
    plt.legend(loc=2)
    fig.savefig(os.path.join(RUN_ID, 'combined.png'))
    plt.close()


def plot_val_estimates(episode, alpha):
    fig = plt.figure()
    x = range(1, NSTATES + 1)
    plt.plot(x, [s.v for s in states][1:NSTATES + 1], label='Estimated')
    plt.plot(x, ACTUAL_VALUES, label='Actual')
    plt.xticks(x, ['A', 'B', 'C', 'D', 'E'])
    plt.title('State value estimates')
    plt.xlim((0.5, NSTATES + 0.25))
    plt.ylim((0, 0.85))
    plt.text(4.2, 0.14, 'T = {}'.format(episode))
    plt.text(4.2, 0.10, u'$\\alpha$ = {}'.format(round(alpha, 4)))
    plt.legend(loc=2)
    fig.savefig(os.path.join('{}'.format(RUN_ID), '{}.png'.format(str(episode).zfill(4))))
    plt.close()


TD(0.5)
