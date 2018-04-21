import numpy as np

from project3.environment import Player


# Littman (1994): alpha = 0.001, alpha_decay = 0.9999954
# Greenwald (2008): alpha = ?, alpha_decay = ?, alpha_min = 0.001

class QLearner(Player):
    """
    Based on pseudocode in Greenwald (2008)
    Table 1, page 3
    """
    def __init__(self, player_info, ns, na, alpha=1.0, alpha_decay_rate=0.999993,
                 random_action_rate=0.5, random_action_rate_decay=0.99,
                 gamma=0.9):
        self.s = 0
        self.a = 0
        self.ns = ns
        self.na = na

        # S x A
        self.Q = np.random.random((self.ns, self.na)) * 2 - 1

        self.alpha = alpha
        self.alpha_decay_rate = alpha_decay_rate
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.gamma = gamma
        self.memory = []

        # For graphing, logging, etc
        self.algo_name = 'Q-Learner'

        x, y, has_ball, name = player_info
        super(QLearner, self).__init__(x, y, has_ball, name)

    def query_initial(self, s):
        if np.random.random() < self.random_action_rate:
            action = np.random.choice(self.na)
            self.random_action_rate *= self.random_action_rate_decay
        else:
            action = np.argmax(self.Q[s])

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, sp, r):
        delta_Q = self.update_Q((s, a, sp, r))

        if np.random.random() < self.random_action_rate:
            action = np.random.choice(self.na)
            self.random_action_rate *= self.random_action_rate_decay
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
        self.alpha *= self.alpha_decay_rate
        return abs(updated_Q - prev_Q)


class FriendQLearner(QLearner):
    """
    Based on equations in Littman (2001)
    Equation 6, page 2
    Equation 7, page 4
    """
    def __init__(self, *args):
        super(FriendQLearner, self).__init__(*args)
        self.Q = np.ones((self.ns, self.na, self.na))
        self.algo_name = 'Friend-Q'

    def query_initial(self, s):
        action, op_action = np.unravel_index(np.argmax([self.Q[s]]), self.Q[s].shape)

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, o, sp, r):
        """
        Vi(s) = max_{a \in A} Qi(s, a, o)

        Friend-Q works cooperatively, so each action set (player action "a" and opponent
        action "o" combo) is selected in order to achieve the highest possible value V(s)
        from the Q table at Q(s, a, o)
        """
        delta_Q = self.update_Q((s, a, o, sp, r))
        action, op_action = np.unravel_index(np.argmax([self.Q[sp]]), self.Q[s].shape)

        # Update current state and action
        self.s = sp
        self.a = action

        return action, delta_Q

    def update_Q(self, experience_tuple):
        s, a, o, sp, r = experience_tuple
        prev_Q = self.Q[s, a, o]

        # Calculate Nash_i(s, Q_1, Q_2)
        max_Qs = np.argmax(self.Q[s], axis=None)
        a_ind, o_ind = np.unravel_index(max_Qs, self.Q[s].shape)
        updated_Q = prev_Q + self.alpha * (r + self.gamma * self.Q[s, a_ind, o_ind] - prev_Q)

        self.Q[s, a_ind, o_ind] = updated_Q
        self.alpha *= self.alpha_decay_rate
        return abs(updated_Q - prev_Q)


class FoeQLearner(QLearner):
    """
    Based on equations in Littman (2001)
    Equation 6, page 2
    Equation 8, page 4

    Also based on minimax-Q pseudocode in Littman (1994)
    Figure 1, page 4
    """
    def __init__(self, *args):
        super(FoeQLearner, self).__init__(*args)
        self.Q = np.ones((self.ns, self.na, self.na))
        self.V = np.ones(self.ns)
        self.pi = np.empty((self.ns, self.na))
        self.pi.fill(1./self.na)
        self.alpha = 1.0
        self.algo_name = 'Foe-Q'

    def query_initial(self, s):

        return 0

    def query(self, s, a, o, sp, r):
        return 0, 0

    def update_Q(self, experience_tuple):
        s, a, o, sp, r = experience_tuple
        prev_Q = self.Q[s, a, o]

        # Calculate Nash_i(s, Q_1, Q_2)
        max_Qs = np.argmax(self.Q[s], axis=None)
        a_ind, o_ind = np.unravel_index(max_Qs, self.Q[s].shape)
        updated_Q = prev_Q + self.alpha * (r + self.gamma * self.Q[s, a_ind, o_ind] - prev_Q)

        self.Q[s, a_ind, o_ind] = updated_Q
        self.alpha *= self.alpha_decay_rate
        return abs(updated_Q - prev_Q)


class CEQLearner(QLearner):
    def __init__(self, *args):
        super(CEQLearner, self).__init__(*args)
        self.Q = np.ones((self.ns, self.na, self.na))
        self.V = np.ones(self.ns)
        self.pi = np.zeros((self.ns, self.na))
        self.pi.fill(1./self.na)
        self.algo_name = 'Correlated-Q'

    def query_initial(self, s):
        pass

    def query(self, s, a, o, sp, r):
        pass


class RandomAgent(Player):
    def __init__(self, player_info, ns, na):
        self.player_info = player_info
        self.na = na
        self.Q = np.zeros((ns, na, na))
        self.algo_name = 'Random Agent'

        x, y, has_ball, name = player_info
        super(RandomAgent, self).__init__(x, y, has_ball, name)

    def query_initial(self, s):
        return np.random.choice(self.na)

    def query(self, s, a, o, sp, r):
        # Fake delta_Q = 0
        return np.random.choice(self.na), 0
