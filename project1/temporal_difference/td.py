import numpy as np

from project1.models import states
from project1.plots import plot_val_estimates
from project1.settings import DEBUG, NSTATES, MAX_ITERATIONS

max_lookahead = 100  # maximum_lookahead
tol = 1e-3


def _reset_w(nstates):
    return [0.5 for _ in range(nstates)]


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


def _step_weight(lambda_val, k):
    """
    The weight applied to each k-step estimate to calculate total change to state value
    :param lambda_val:
    :param k:
    :return:
    """
    # Just fill out remaining weight to equal 1 to avoid geometric series calculations...
    if k > max_lookahead:
        existing_weights = sum([_step_weight(lambda_val, i) for i in range(1, max_lookahead)])
        return 1 - existing_weights
    return (1 - lambda_val) * pow(lambda_val, k - 1)


def _step_estimate(state_seq, k):
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


def TD(lambda_val, alpha=0.1, alpha_decay_rate=0.98, gamma=1.0, num_episodes=10, history=False):
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
    if DEBUG:
        print 'TD({})'.format(lambda_val)

    # Store history state values after each episode
    V = np.ndarray((num_episodes, NSTATES))  # don't care about terminal states

    # Store episodes to repeatedly present
    episodes = _generate_episodes(num_episodes)

    # Flag convergence
    converged = False
    iterator = 0

    # Repeatedly present same episodes in training set until convergence
    while not converged:

        # Reset and store deltas of value/weight vector
        # delta_v = _reset_delta_v(NSTATES)

        file_counter = 0

        for T, sequence in enumerate(episodes):

            # Update learning rate according to episode
            # alpha = 1. / pow(T + 1, alpha_decay_rate)
            alpha *= alpha_decay_rate

            # Execute iterative value updates
            for s in sequence:
                s.e = 0

            for t in range(1, len(sequence)):
                sequence[t - 1].e += 1

                # Equation (1)
                # State sequence indices = range(t=1...m)
                for s in sequence:
                    state_error = sequence[t].r + gamma * sequence[t].v - sequence[t - 1].v
                    delta = alpha * s.e * state_error

                    # if not s.terminal:
                    #     delta_v[s.index - 1] += delta

                    # Uncomment to apply weight update after each TIMESTEP
                    s.v += delta
                    # delta_w = _reset_delta_w()
                    s.e *= lambda_val * gamma

                # Uncomment to apply weight update after each EPISODE
                # for i in range(nstates):
                #     w[i] += delta_w[i]
                # delta_w = _reset_delta_w()

            V[T] = [states[i].v for i in range(1, len(states) - 1)]  # don't care about terminal states
            # plot_val_estimates(file_counter, [s.v for s in states][1:NSTATES + 1], T, alpha)
            # file_counter += 1

        # Uncomment to apply weight update after each presentation of TRAINING SET
        # for i in range(len(delta_v)):
        #     states[i + 1].v += delta_v[i]

        # Scalar combination of delta_v (i.e. total step size)
        # print '   ', delta_v
        # print np.sqrt(sum([pow(dv, 2) for dv in delta_v]))

        iterator += 1

        if iterator >= MAX_ITERATIONS:
            # print 'Max iterations reached, faking convergence'
            break

        # fuck it!
        converged = True

    return V if history else V[len(V) - 1]
