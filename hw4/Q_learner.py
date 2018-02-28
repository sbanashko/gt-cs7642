import numpy as np


class QLearningAgent(object):
    """Q"""

    def __init__(self, ns, na, alpha=0.2, random_action_rate=0.5,
                 random_action_rate_decay=0.999, gamma=0.9, dyna=200):
        self.s = 0
        self.a = 0
        self.ns = ns
        self.na = na
        # Initialize Q table (because it's a finite problem and we can)
        self.Q = np.random.uniform(-1.0, 1.0, (ns, na))
        self.alpha = alpha
        self.random_action_rate = random_action_rate
        self.random_action_rate_decay = random_action_rate_decay
        self.gamma = gamma
        self.memory = []
        self.dyna = dyna

    def query_initial(self, s):
        """
        Select action without updating the Q-table
        :param s:
        :return:
        """
        # if np.random.random() < self.alpha:
        #     return np.random.choice(self.na)
        # return max(range(self.na), key=lambda x: self.Q[s, x])
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
        else:
            action = np.argmax([self.Q[sp, a] for a in range(self.na)])

        self.random_action_rate *= self.random_action_rate_decay

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
        updated_Q = (1 - self.alpha) * prev_Q + self.alpha * (
                r + self.gamma * self.Q[sp, np.argmax([self.Q[sp, i] for i in range(self.na)])])
        self.Q[s, a] = updated_Q
        return abs(updated_Q - prev_Q)
