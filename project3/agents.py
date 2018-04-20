import numpy as np

from project3.environment import Player


class QLearner(Player):
    def __init__(self, player_info, ns, na, alpha=0.2, alpha_decay_rate=1.0,
                 random_action_rate=0.75, random_action_rate_decay=0.999,
                 gamma=0.9, dyna=0):
        self.s = 0
        self.a = 0
        self.ns = ns
        self.na = na

        # S x A (non-collaborative learner)
        self.Q = np.zeros((ns, na))

        self.alpha = alpha
        self.alpha_decay_rate = alpha_decay_rate
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.gamma = gamma
        self.memory = []
        self.dyna = dyna

        # FFQ, CEQ are collaborative (take other agent actions into consideration)
        self.collaborative = False

        # For graphing, logging, etc
        self.algo_name = 'Q-Learner'

        x, y, has_ball, name = player_info
        super(QLearner, self).__init__(x, y, has_ball, name)

    def query_initial(self, s):
        """
        Select action without updating the Q-table
        :param s:
        :return:
        """
        if np.random.random() < self.random_action_rate:
            action = np.random.choice(self.na)
            self.random_action_rate *= self.random_action_rate_decay
        else:
            action = np.argmax([self.Q[self.s, a] for a in range(self.na)])

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, sp, r):
        """
        Select action and update Q-table
        :param s: previous state
        :param a: selected action
        :param sp: new state
        :param r: immediate reward
        :return:
        """
        delta_Q = self.update_Q((s, a, sp, r))

        # Dyna-Q
        if self.dyna > 0:

            # Replace T and R models with in-memory historical data
            self.memory.append((self.s, self.a, sp, r))

            # Hallucinate
            for d in range(self.dyna):
                # Update Q-table
                self.update_Q(self.memory[np.random.choice(len(self.memory))])

        if np.random.random() < self.random_action_rate:
            action = np.random.choice(self.na)
            self.random_action_rate *= self.random_action_rate_decay
        else:
            action = np.argmax([self.Q[sp, a] for a in range(self.na)])

        # Update current state and action
        self.s = sp
        self.a = action

        return action, delta_Q

    def update_Q(self, experience_tuple):
        """
        Update Q table
        :param experience_tuple: s, a, s', r
        :return:
        """
        s, a, sp, r = experience_tuple
        prev_Q = self.Q[s, a]
        updated_Q = prev_Q + self.alpha * (
                r + self.gamma * self.Q[sp, np.argmax([self.Q[sp, i] for i in range(self.na)])] - prev_Q)
        self.Q[s, a] = updated_Q
        self.alpha *= self.alpha_decay_rate
        return abs(updated_Q - prev_Q)


class FriendQLearner(QLearner):

    def __init__(self, *args):
        super(FriendQLearner, self).__init__(*args)

        # S x A x O (assumes 2-player game and same number of actions
        # available to each player)
        self.Q = np.zeros((self.ns, self.na, self.na))

        # self.random_action_rate = 0.0

        self.collaborative = True
        self.algo_name = 'Friend-Q'

    def query_initial(self, s):
        """
        Select action without updating the Q-table
        :param s:
        :return:
        """
        if np.random.random() < self.random_action_rate:
            action = np.random.choice(self.na)
            self.random_action_rate *= self.random_action_rate_decay
        else:
            action, op_action = np.unravel_index(np.argmax([self.Q[s]]), self.Q[s].shape)

        # Update current state and action
        self.s = s
        self.a = action

        return action

    def query(self, s, a, o, sp, r):
        """
        Select action and update Q-table
        :param s: previous state
        :param a: selected action
        :param o: opponent action
        :param sp: new state
        :param r: immediate reward
        :return:
        """
        delta_Q = self.update_Q((s, a, o, sp, r))

        if np.random.random() < self.random_action_rate:
            action = np.random.choice(self.na)
            self.random_action_rate *= self.random_action_rate_decay
        else:
            action, op_action = np.unravel_index(np.argmax([self.Q[sp]]), self.Q[s].shape)

        # Update current state and action
        self.s = sp
        self.a = action

        return action, delta_Q

    def update_Q(self, experience_tuple):
        """
        Friend Q update rule
        see equations (6) and (7) from Littman (2001)
        :param experience_tuple:
        :return:
        """
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
    Equivalently, minimax-Q
    """
    def __init__(self, *args):
        super(FoeQLearner, self).__init__(*args)

        # S x A x O (assumes 2-player game and same number of actions
        # available to each player)
        self.Q = np.zeros((self.ns, self.na, self.na))

        self.collaborative = True
        self.algo_name = 'Foe-Q'


class CEQLearner(QLearner):
    def __init__(self, *args):
        super(CEQLearner, self).__init__(*args)

        # In this case we only care about probability distribution sigma over
        # action space from state s (theoretically there'd be one of these for
        # every state)
        # Initialize to equal probabilities == uniform random
        self.Ps = [1. / self.na for _ in range(self.na)]

        # S x A x O (assumes 2-player game and same number of actions
        # available to each player)
        self.Q = np.zeros((self.ns, self.na, self.na))

        self.collaborative = True
        self.algo_name = 'Correlated-Q'

    def query_initial(self, s):
        # S x A x O (assumes 2-player game and same number of actions
        # available to each player)
        self.Q = np.zeros((self.ns, self.na))

        pass

    def query(self, s, a, o, sp, r):
        pass


class RandomAgent(Player):
    def __init__(self, player_info, na):
        self.player_info = player_info
        self.na = na
        self.algo_name = 'Random Agent'

        x, y, has_ball, name = player_info
        super(RandomAgent, self).__init__(x, y, has_ball, name)

    def query_initial(self):
        return np.random.choice(self.na)

    def query(self):
        return np.random.choice(self.na)
