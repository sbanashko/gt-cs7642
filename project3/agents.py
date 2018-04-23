import numpy as np

from project3.environment import Player

# Littman (1994): alpha = 1.0, alpha_decay = 0.9999954
# Greenwald (2003): alpha = ?, alpha_decay = ?, alpha_min = 0.001
from project3.utils import lp_util
from project3.utils.log_util import logger


class QLearner(Player):
    """
    See Greenwald (2008)
    Table 1, page 3
    """

    def __init__(self, player_info, ns, na,
                 alpha=1.0, alpha_decay=0.999997, alpha_min=0.0,
                 epsilon=0.75, epsilon_decay=0.99995, epsilon_min=0.01,
                 gamma=0.9):
        self.s = 0
        self.a = 0
        self.ns = ns
        self.na = na

        # Q table
        # Initialized to [-1, 1) uniformly at random
        self.Q = np.random.randn(self.ns, self.na)
        # Initialized to [0, 1) uniformly at random
        # self.Q = np.random.random((self.ns, self.na))
        # Initialized to 0
        # self.Q = np.zeros((self.ns, self.na))

        # Learning rate
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.alpha_min = alpha_min

        # Random action rate
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Discount rate
        self.gamma = gamma

        # For graphing, logging, etc
        self.algo_name = 'Q-Learner'

        x, y, has_ball, name = player_info
        super(QLearner, self).__init__(x, y, has_ball, name)

    def query_initial(self, s):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            action = np.argmax(self.Q[s])

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, o, sp, r, op_Q):
        delta_Q = self.update_Q((s, a, sp, r))

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            action = np.argmax(self.Q[sp])

        # Update current state and action
        self.s = sp
        self.a = action

        return action, delta_Q

    def update_Q(self, experience_tuple):
        s, a, sp, r = experience_tuple
        action = np.argmax(self.Q[s])
        prev_Q = self.Q[s, a]
        updated_Q = (1 - self.alpha) * prev_Q + \
                    self.alpha * ((1 - self.gamma) * r + self.gamma * self.Q[sp, action] - prev_Q)
        self.Q[s, a] = updated_Q
        # print('{} -> {}\t\talpha={}'.format(prev_Q, updated_Q, self.alpha))
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        return abs(updated_Q - prev_Q)


class FriendQLearner(QLearner):
    """
    See Littman (2001)
    Equation (6), page 2
    Equation (7), page 4
    """

    def __init__(self, *args):
        super(FriendQLearner, self).__init__(*args)
        self.Q = np.random.randn(self.ns, self.na, self.na)
        self.V = np.ones(self.ns)
        self.alpha_decay = 0.9999
        self.algo_name = 'Friend-Q'

    def query_initial(self, s):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon = min(self.epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            action, op_action = np.unravel_index(np.argmax([self.Q[s]]), self.Q[s].shape)

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, o, sp, r, op_Q):
        """
        Vi(s) = max_{a \in A} Qi(s, a, o)

        Friend-Q works cooperatively, making the assumption that the opponent wants
        to maximized the sum of players' rewards. The Friend-Q player therefore selects
        the action corresponding to the player's Q[s] table's maximum sum of rewards.
        Opponent Q table is not considered in FFQ updates.
        """
        delta_Q = self.update_Q((s, a, o, sp, r))

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon = min(self.epsilon * self.epsilon_decay, self.epsilon_min)
        else:
            max_Qs = np.argmax([self.Q[sp]], axis=None)
            action, op_action = np.unravel_index(max_Qs, self.Q[s].shape)

        # Update current state and action
        self.s = sp
        self.a = action

        return action, delta_Q

    def update_Q(self, experience_tuple):
        s, a, o, sp, r = experience_tuple
        prev_Q = self.Q[s, a, o]

        # Calculate Nash_i(s, Q_1, Q_2)
        max_Qs = np.max(self.Q[s])
        updated_Q = (1 - self.alpha) * prev_Q + self.alpha * (r + self.gamma * self.V[sp])

        self.Q[s, a, o] = updated_Q

        # Update V[s] with Nash_i(s, Q_1, Q_2)
        self.V[s] = max_Qs

        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        return abs(updated_Q - prev_Q)


class FoeQLearner(QLearner):
    """
    See Littman (2001)
    Equation (6), page 2
    Equation (8), page 4

    See Littman (1994)
    Figure 1, page 4
    Section 6.2:
    epsilon = 0.2
    epsilon_decay = 0.9999954
    """

    def __init__(self, *args):
        super(FoeQLearner, self).__init__(*args)
        # self.Q = np.ones((self.ns, self.na, self.na))
        self.Q = np.random.randn(self.ns, self.na, self.na)
        self.V = np.zeros(self.ns)
        self.pi = np.empty((self.ns, self.na))
        self.pi.fill(1. / self.na)
        # self.alpha_decay = 0.999
        # self.alpha_min = 0.01
        self.algo_name = 'Foe-Q'

    def query_initial(self, s):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon *= self.epsilon_decay
        else:
            # Select new action with new probability distribution of action space at state s
            try:
                action = np.random.choice(range(self.na), p=self.pi[s])
            except ValueError:
                # Stupid float precision... Normalize probabilities to sum to 1
                logger.warn('Stupid float precision... Normalize probabilities to sum to 1')
                self.pi[s] = self.pi[s] / self.pi[s].sum(0)
                action = np.random.choice(range(self.na), p=self.pi[s])

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, o, sp, r, op_Q):
        delta_Q = self.update_Q((s, a, o, sp, r))

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon *= self.epsilon_decay
        else:
            # Select new action with new probability distribution of action space at state s
            try:
                action = np.random.choice(range(self.na), p=self.pi[sp])
            except ValueError:
                # Stupid float precision... Normalize probabilities to sum to 1
                self.pi[s] = self.pi[s] / self.pi[s].sum(0)
                action = np.random.choice(range(self.na), p=self.pi[sp])

        # Update current state and action
        self.s = sp
        self.a = action

        return action, delta_Q

    def update_Q(self, experience_tuple):
        s, a, o, sp, r = experience_tuple

        # Update pi (maximizing the minimum value V[s])
        self.pi[sp] = lp_util.maxmin(self.Q[sp])

        # 4a. Update V[s']
        # See Greenwald (2005) Table 2, page 12
        self.V[sp] = sum([self.pi[sp, a_] * self.Q[sp, a_, o] for a_ in range(self.na)])
        # See Littman (1994) Figure 1, page 4
        # self.V[s] = min(range(self.na), key=lambda op: op)
        # See idk who to believe anymore
        # objective_fn = lambda op: sum([self.pi[s, ap] * self.Q[s, ap, op] for ap in range(self.na)])
        # self.V[sp] = min(range(self.na), key=objective_fn)

        prev_Q = self.Q[s, a, o]

        # 4b. Update Q[s, a]
        updated_Q = (1 - self.alpha) * prev_Q + self.alpha * ((1 - self.gamma) * r + self.gamma * self.V[sp])
        self.Q[s, a, o] = updated_Q

        # Decay alpha
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        return abs(updated_Q - prev_Q)


class CEQLearner(QLearner):
    """
    Based on Greenwald (2003)
    Section 3.1, Equation (9), page 3
    """

    def __init__(self, *args):
        super(CEQLearner, self).__init__(*args)
        # self.Q = np.ones((self.ns, self.na, self.na))
        self.Q = np.random.randn(self.ns, self.na, self.na)
        self.V = np.ones(self.ns)
        self.pi = np.zeros((self.ns, self.na))
        self.pi.fill(1. / self.na)
        self.algo_name = 'Correlated-Q'

    def query_initial(self, s):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon *= self.epsilon_decay
        else:
            # Select new action with new probability distribution of action space at state s
            try:
                action = np.random.choice(self.na, p=self.pi[s])
            except ValueError:
                # Stupid float precision... Normalize probabilities to sum to 1
                self.pi[s] = self.pi[s] / self.pi[s].sum(0)
                action = np.random.choice(self.na)

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, o, sp, r, op_Q):
        delta_Q = self.update_Q((s, a, o, sp, r), op_Q)

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.na)
            self.epsilon *= self.epsilon_decay
        else:
            # Select new action with new probability distribution of action space at state s
            try:
                action = np.random.choice(self.na, p=self.pi[sp])
            except ValueError:
                # Stupid float precision... Normalize probabilities to sum to 1
                self.pi[s] = self.pi[sp] / self.pi[sp].sum(0)
                action = np.random.choice(self.na, p=self.pi[sp])

        # Update current state and action
        self.s = s
        self.a = action

        return action, delta_Q

    def update_Q(self, experience_tuple, op_Q):

        s, a, o, sp, r = experience_tuple
        prev_Q = self.Q[s, a, o]

        # Update pi (maximizing the minimum value V[s])
        joint_dist = lp_util.ce(self.Q[sp], op_Q[sp])
        self.pi[s] = np.sum(np.array(joint_dist).reshape((self.na, self.na)), axis=1)
        # self.pi[s] = lp_util.solve(self.Q[sp], self.algo_name)

        # Update V[sp]
        # See Greenwald (2005) page 10
        # objective_fn = lambda op: sum([self.pi[s, ap] * self.Q[s, ap, op] for ap in range(self.na)])
        # self.V[sp] = min(range(self.na), key=objective_fn)
        self.V[sp] = sum([self.pi[sp, a_] * self.Q[sp, a_, o] for a_ in range(self.na)])

        # Update Q
        updated_Q = (1 - self.alpha) * prev_Q + self.alpha * (r + self.gamma * self.V[sp])
        self.Q[s, a, o] = updated_Q

        # Decay alpha
        self.alpha = max(self.alpha * self.alpha_decay, self.alpha_min)
        return abs(updated_Q - prev_Q)


class RandomAgent(Player):
    def __init__(self, player_info, ns, na):
        self.player_info = player_info
        self.na = na
        self.Q = np.zeros((ns, na, na))
        self.algo_name = 'Random Agent'

        x, y, has_ball, name = player_info
        super(RandomAgent, self).__init__(x, y, has_ball, name)

    def query_initial(self, *args):
        return np.random.choice(self.na)

    def query(self, *args):
        # Fake delta_Q = 0
        return np.random.choice(self.na), 0
