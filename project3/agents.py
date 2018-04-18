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

        # S x A x O (assumes 2-player game and same number of actions
        # available to each player)
        self.Q = np.zeros((ns, na))

        self.alpha = alpha
        self.alpha_decay_rate = alpha_decay_rate
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.gamma = gamma
        self.memory = []
        self.dyna = dyna

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
        else:
            action = np.argmax([self.Q[self.s, a] for a in range(self.na)])

        self.random_action_rate *= self.random_action_rate_decay

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

    def update_Q(self, experience_tuple):
        """
        Friend Q update rule
        see equations (6) and (7) from Littman (2001)
        :param experience_tuple:
        :return:
        """
        s, a, a_opponent, sp, r, r_opponent = experience_tuple
        prev_Q = self.Q[s, a, a_opponent]

        # Calculate Nash_i(s, Q_a, Q_b)
        max_Qs = np.argmax(self.Q[s], axis=None)
        a_index, a_opponent_index = np.unravel_index(max_Qs, self.Q[s].shape)
        nash = max_Qs[s, a_index, a_opponent_index]
        updated_Q = prev_Q + self.alpha * (r + self.gamma * nash - prev_Q)

        self.Q[s, a, a_opponent_index] = updated_Q
        self.alpha *= self.alpha_decay_rate
        return abs(updated_Q - prev_Q)


class FoeQLearner(QLearner):
    """
    Equivalently, minimax-Q
    """
    pass


class CEQLearner(QLearner):
    def __init__(self, *args):
        super(CEQLearner, self).__init__(*args)

        # In this case we only care about probability distribution sigma over
        # action space from state s (theoretically there'd be one of these for
        # every state)
        # Initialize to equal probabilities == uniform random
        self.Ps = [1. / self.na for _ in range(self.na)]

        # TODO other stuff

    def query_initial(self, s):
        pass

    def query(self, s, a, sp, r):
        pass


class RandomAgent(Player):
    def __init__(self, player_info, na):
        self.player_info = player_info
        self.na = na

        x, y, has_ball, name = player_info
        super(RandomAgent, self).__init__(x, y, has_ball, name)

    def query_initial(self):
        return np.random.choice(self.na)

    def query(self):
        return np.random.choice(self.na)
